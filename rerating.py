import json
import os
from unicodedata import category
from config import get_args
import torch
import torch.nn as nn
import numpy as np
import logging

from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer
from encoder import get_encoder

from data import DPC2022, DPC2022_for_generate
from torch.utils.data import Dataset, DataLoader
import pandas as pd
# from transformers import AdamW
from models.transformer import Transformer_visualgpt, VisualEncoder, ScaledDotProductAttentionMemory, ScaledDotProductAttention
from loss import BatchWeightedCE
import random
from train import train_for_epoch
from eval import eval_for_epoch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models.comment_score_model import CommentScoreNet
from tqdm import tqdm
import clip
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from metrics.captions_metrics import EvalCap


def data_norm(x):

    return (x-x.min())/(x.max()-x.min())

def comment_rerating(model, dataloader, args):
    
    rank = dist.get_rank()

    save_gen_caption_dir = os.path.join(args.log_dir, 'eval_gen_caption')
    if not os.path.exists(save_gen_caption_dir):
        os.mkdir(save_gen_caption_dir)

    clip_tokenizer = clip.tokenize
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_checkpoint = torch.load('saved_models/comment_assessment/%s_comment_assessment.pth' % (args.comment_assessment_model), map_location={'cuda:0':f'cuda:{rank}'})
    clip_model.load_state_dict(clip_checkpoint['state_dict'], strict=False)
    clip_model = clip_model.cuda()
    clip_model = DDP(clip_model, device_ids=[rank], find_unused_parameters=True)
    

    sentence_similarity_model = SentenceTransformer(os.path.join(args.huggingface_model_root, 'all-MiniLM-L6-v2'))
    sentence_similarity_model = sentence_similarity_model.cuda()
    sentence_similarity_model = DDP(sentence_similarity_model, device_ids=[rank], find_unused_parameters=True)


    CANet = CommentScoreNet(args)
    CANet_checkpoint = torch.load('saved_models/comment_assessment/%s_comment_assessment.pth' % (args.comment_assessment_model), map_location={'cuda:0':f'cuda:{rank}'})
    CANet.load_state_dict(CANet_checkpoint['state_dict'], strict=False)
    CANet = CANet.cuda()
    CANet = DDP(CANet, device_ids=[rank], find_unused_parameters=True)
    dist.barrier()


    eval_cap = EvalCap()

    candidates_captions_dict = {}
    references_captions_dict = {}

    dataset = dataloader.dataset
    gpt2_tokenizer = dataset.gpt2_tokenizer
 
    num_beams = 32
    output_size = 32
    desc_template = 'rerating training data'


    
    model.eval()
    with torch.no_grad():
        
        rank = dist.get_rank()
        if rank == 0:
            pbar = tqdm(desc=desc_template, unit='it', total=len(dataloader))
            
        for i, batch_data_dict in enumerate(dataloader):
            
            imgfeats = batch_data_dict['imgfeats'].cuda()
            images = batch_data_dict['images'].cuda()
            IDs = batch_data_dict['IDs']
            ascores = batch_data_dict['ascores']

            # batch_outs, log_probs = model.beam_search(imgfeats, args.max_sentence_length, text_tokenizer.eos_token_id,
            #                                         num_beams, out_size=output_size)
            batch_outs, log_probs = model.module.beam_search(imgfeats, args.max_sentence_length, gpt2_tokenizer.eos_token_id,
                                                    num_beams, out_size=output_size)
            
            for k, ID in enumerate(IDs):
                generation_comment_list = []
                ascore = ascores[k]
                outs = batch_outs[k]
                image = images[k]
                references_captions_dict[ID], ref_cs, ref_norm_cs = dataset.readData_from_id(ID)
                
                clip_ref_comment_token = clip_tokenizer(references_captions_dict[ID], truncate=True).cuda()
                clip_ref_logits_per_image, clip_logits_per_ref_comment = clip_model(image.unsqueeze(0), clip_ref_comment_token)
                clip_ref_probs_per_image = clip_ref_logits_per_image.softmax(dim=-1).cpu().numpy()
                norm_clip_ref_logits_per_image = data_norm(clip_ref_logits_per_image)
                candidate_captions = gpt2_tokenizer.batch_decode(outs, skip_special_tokens=False)

                # remove eos token
                for k, c in enumerate(candidate_captions):
                    c = ' '.join(c.split(' ')[:-1])
                    # print('ID:%s, candidate captions:%s, scores: %.6f'%(ID, c, scores[k].item()))  
                    generation_comment_list.append(c)

                clip_can_comment_token = clip_tokenizer(generation_comment_list, truncate=True).cuda()
                clip_can_logits_per_image, clip_logits_per_can_comment_ = clip_model(image.unsqueeze(0), clip_can_comment_token)
                
                generation_comment_similarity_embeding = sentence_similarity_model.module.encode(generation_comment_list)
                ref_comment_similarity_embeding = sentence_similarity_model.module.encode(references_captions_dict[ID])

                ref_gen_similarity_array = cosine_similarity(ref_comment_similarity_embeding, generation_comment_similarity_embeding)

                references_captions_encoding = CANet.module.tokenizer.batch_encode_plus(
                list(references_captions_dict[ID]),
                add_special_tokens=True,
                max_length=args.max_sentence_length,
                return_token_type_ids=False,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
                )

                ref_as_preds, ref_cs_preds = CANet(references_captions_encoding['input_ids'].cuda(), references_captions_encoding['attention_mask'].cuda())
                
                
                sorted, can_indices = torch.sort(clip_can_logits_per_image[0], descending=True)
                for idx in can_indices.tolist():
                    print({
                        'can_comment':generation_comment_list[idx], 'clip':clip_can_logits_per_image[0][idx].item()
                    })

                sorted, ref_indices = torch.sort(clip_ref_logits_per_image[0], descending=True)
                for idx in ref_indices.tolist():
                    print({
                        'ref_comment':references_captions_dict[ID][idx], 'clip':clip_ref_logits_per_image[0][idx].item(), 'clip_prob':clip_ref_probs_per_image[0][idx].item()
                    })

                as_diffs = torch.abs(ref_as_preds-ascore)
                
                weight_list = []
                for idx in range(len(references_captions_dict[ID])):
                    
                    cl = norm_clip_ref_logits_per_image[0][idx].item()
                    cs = ref_cs_preds[idx].item()
                    diff = as_diffs[idx].item()
                    max_sim_idx = np.argmax(ref_gen_similarity_array[idx])
                    sim = ref_gen_similarity_array[idx][max_sim_idx]

                    # weigth = ((cs + 1e-6)/( diff * sim +1e-6)).item()
                    # weigth = (max(cs - diff + cl, 0)/(sim + 1)).item()
                    weight = (cs/sim).item()
                    # eval_cap.evaluate_for_rerating({ID: [references_captions_dict[ID][idx]]}, {ID: generation_comment_list})
                    weight_list.append(weight)
                
                weights = torch.tensor(weight_list)
                sorted, indices = torch.sort(weights, descending=True)
                for idx in indices.tolist():

                    cl = norm_clip_ref_logits_per_image[0][idx].item()
                    cs = ref_cs_preds[idx].item()
                    diff = as_diffs[idx].item()
                    max_sim_idx = np.argmax(ref_gen_similarity_array[idx])
                    sim = ref_gen_similarity_array[idx][max_sim_idx]

                    print({
                        'comment':references_captions_dict[ID][idx],
                        'max_sim_gen_comment':generation_comment_list[max_sim_idx],
                        'clip':cl,
                        'cs':cs,
                        'diff':diff, 
                        'sim':sim,
                        'weigth':weights[idx].item(),
                        # 'metric':eval_cap.eval,
                    })
                print('end')
                # for j, c in enumerate(candidate_captions):
                #     print('%d, captions:%s, %.6f'%(j, c, output_seq_scores[j].item()))

 
            rank = dist.get_rank() 
            if rank == 0:
                pbar.update()

            # if i > 3:
            #     break



    
    return 



if __name__ == '__main__':

  args = get_args()
  print(args)

  dist.init_process_group("nccl")
  rank = dist.get_rank()
  device = torch.device(f'cuda:{rank}')
  torch.cuda.set_device(rank)

  # Model and dataloaders
  encoder =VisualEncoder(3, 0, attention_module=ScaledDotProductAttention)
  model = Transformer_visualgpt(encoder)

  train_df = pd.read_csv(os.path.join(args.train_root, 'train.csv'))
  val_df = pd.read_csv(os.path.join(args.test_and_val_root, 'val.csv'))
  
  train_dataset = DPC2022_for_generate(train_df, model.tokenizer, args)
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
  batch_trainDataLoader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=4, collate_fn=train_dataset.collate_fn, drop_last=True, sampler=train_sampler)

  val_dataset = DPC2022_for_generate(val_df, model.tokenizer , args)
  val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
  batch_valDataLoader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=4, collate_fn=val_dataset.collate_fn, drop_last=True, sampler=val_sampler)

  model_fname = '%s/%s/%s_image_caption_last_model.pth' % (args.models_dir, args.batch_weighted_ce ,args.use_model)

  if os.path.exists(model_fname):
      # model_checkpoint = torch.load(model_fname, map_location='cuda:0')
      model_checkpoint = torch.load(model_fname, map_location={'cuda:0':f'cuda:{rank}'})
      torch.set_rng_state(model_checkpoint['torch_rng_state'].cpu())
      torch.cuda.set_rng_state(model_checkpoint['cuda_rng_state'].cpu())
      np.random.set_state(model_checkpoint['numpy_rng_state'])
      random.setstate(model_checkpoint['random_rng_state'])
      model.load_state_dict(model_checkpoint['state_dict'], strict=False)
      

      start_epoch = model_checkpoint['epoch']
      eval_result_dict = model_checkpoint['eval_result_dict']
      print(f"start_epoch:{start_epoch}, eval_result_dict:{eval_result_dict}")

  # 等待加载完成
  dist.barrier()

  # 多gpu
  model = model.cuda()
  model = DDP(model, device_ids=[rank], find_unused_parameters=True)

  comment_rerating(model, batch_trainDataLoader, args)
