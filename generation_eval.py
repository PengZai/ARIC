from config import get_args
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
import json
import os
from data import DPC2022_for_generate
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics.captions_metrics import EvalCap
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from models.comment_score_model import CommentScoreNet
import clip
from models.multimodal_aesthetic_assessment_model import MAANet
from metrics.score_metrics import score_metrics
import numpy as np


def data_norm(x):

    return (x-x.min())/(x.max()-x.min())


def get_score_metric_from_list(as_preds_list, all_ascores):

    as_preds = torch.cat(as_preds_list, dim=0)
    all_as_preds_list = [torch.zeros_like(as_preds) for _ in range(dist.get_world_size())]
    dist.all_gather(all_as_preds_list, as_preds)
    all_as_preds = torch.cat(all_as_preds_list, dim=0)
    
    metric_results_dict = score_metrics(all_as_preds.cpu(), all_ascores.cpu(), mean=5.)

    return metric_results_dict

if __name__ == '__main__':

  gen_data_root = 'cs_cscore_generation_comment_for_eval/epoch=14/num_beams=5'

  if not os.path.exists(os.path.join(gen_data_root, 'selected')):
    os.mkdir(os.path.join(gen_data_root, 'selected'))
  
  args = get_args()
  print(args)

  dist.init_process_group("nccl")
  rank = dist.get_rank()
  device = torch.device(f'cuda:{rank}')
  torch.cuda.set_device(rank)

  gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
  gpt2_tokenizer.add_special_tokens({'pad_token': "+="})
  gpt2_tokenizer.add_special_tokens({'bos_token': "<?"})

  with open('models/bad_comment.json') as f:
    bad_comment_dict = json.load(f)
  bad_comment_list = bad_comment_dict['bad_comments']

  val_df = pd.read_csv(os.path.join(args.test_and_val_root, 'val.csv'))
  val_dataset = DPC2022_for_generate(val_df, gpt2_tokenizer , args)
  val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
  batch_valDataLoader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=4, collate_fn=val_dataset.collate_fn, drop_last=True, sampler=val_sampler)

  references_captions_dict = {}
  candidates_best_prob_comment_dict = {}
  candidates_best_cs_comment_dict = {}
  candidates_best_clip_comment_dict = {}
  candidates_best_clip_cs_comment_dict = {}

  eval_cap = EvalCap()


  sentence_similarity_model = SentenceTransformer(os.path.join(args.huggingface_model_root, 'all-MiniLM-L6-v2'))
  sentence_similarity_model = sentence_similarity_model.cuda()
  sentence_similarity_model.eval()
  SSNet = DDP(sentence_similarity_model, device_ids=[rank], find_unused_parameters=True)

  bad_comment_ss_embeddings = SSNet.module.encode(bad_comment_list)


  CANet = CommentScoreNet(args)
  CANet_checkpoint = torch.load('saved_models/comment_assessment/%s_comment_assessment.pth' % (args.comment_assessment_model), map_location={'cuda:0':f'cuda:{rank}'})
  CANet.load_state_dict(CANet_checkpoint['state_dict'], strict=False)
  CANet = CANet.cuda()
  CANet.eval()
  CANet = DDP(CANet, device_ids=[rank], find_unused_parameters=True)

  clip_tokenizer = clip.tokenize
  clip_model, preprocess = clip.load("ViT-B/32", device=device)
  clip_checkpoint = torch.load('saved_models/comment_assessment/%s_comment_assessment.pth' % (args.comment_assessment_model), map_location={'cuda:0':f'cuda:{rank}'})
  clip_model.load_state_dict(clip_checkpoint['state_dict'], strict=False)
  clip_model = clip_model.cuda()
  clip_model = DDP(clip_model, device_ids=[rank], find_unused_parameters=True)


  single_maa_model = MAANet(args)
  single_maa_model_checkpoint = torch.load('saved_models/maa_single/vit_bert_mlp.pth', map_location={'cuda:0':f'cuda:{rank}'})
  single_maa_model.load_state_dict(single_maa_model_checkpoint['state_dict'], strict=False)
  single_maa_model = single_maa_model.cuda()
  single_maa_model = DDP(single_maa_model, device_ids=[rank], find_unused_parameters=True)

  multiple_maa_model = MAANet(args)
  # multiple_maa_model_checkpoint = torch.load('saved_models/maa_multiple/vit_bert_mlp_64.pth', map_location={'cuda:0':f'cuda:{rank}'})
  multiple_maa_model_checkpoint = torch.load('saved_models/maa_multiple/test/vit_bert_mlp.pth', map_location={'cuda:0':f'cuda:{rank}'})

  multiple_maa_model.load_state_dict(multiple_maa_model_checkpoint['state_dict'], strict=False)
  multiple_maa_model = multiple_maa_model.cuda()
  multiple_maa_model = DDP(multiple_maa_model, device_ids=[rank], find_unused_parameters=True)

  # 美学得分list
  ascores_list = []
  iaa_as_preds_list= []
  best_prob_comment_taa_as_preds_list= []
  best_prob_comment_maa_as_preds_list= []
  best_cs_comment_taa_as_preds_list = []
  best_cs_comment_maa_as_preds_list = []
  best_clip_comment_taa_as_preds_list = []
  best_clip_comment_maa_as_preds_list = []
  best_clip_cs_comment_taa_as_preds_list = []
  best_clip_cs_comment_maa_as_preds_list = []

  multi_iaa_as_preds_list= []
  multi_true_taa_as_preds_list= []
  multi_true_maa_as_preds_list= []
  all_multi_comment_taa_as_preds_list = []
  all_multi_comment_maa_as_preds_list = []
  best_multi_comment_taa_as_preds_list = []
  best_multi_comment_maa_as_preds_list = []
  best_multi_comment_ungroup_taa_as_preds_list = []
  best_multi_comment_ungroup_maa_as_preds_list = []
  best_multi_cs_ungroup_comment_taa_as_preds_list = []
  best_multi_cs_ungroup_comment_maa_as_preds_list = []
  best_multi_cs_comment_taa_as_preds_list = []
  best_multi_cs_comment_maa_as_preds_list = []
  best_multi_clip_ungroup_comment_taa_as_preds_list = []
  best_multi_clip_ungroup_comment_maa_as_preds_list = []
  best_multi_clip_comment_taa_as_preds_list = []
  best_multi_clip_comment_maa_as_preds_list = []
  best_multi_clip_cs_ungroup_comment_taa_as_preds_list = []
  best_multi_clip_cs_ungroup_comment_maa_as_preds_list = []
  best_multi_clip_cs_comment_taa_as_preds_list = []
  best_multi_clip_cs_comment_maa_as_preds_list = []

  # 挑选评论的保存
  

  # 等待加载完成
  dist.barrier()

  with torch.no_grad():
    if dist.get_rank()==0:
      pbar = tqdm(desc=" ", unit="it", total=len(batch_valDataLoader))

    for i, batch_data_dict in enumerate(batch_valDataLoader):
      
      # if i > 2:
      #   break

      imgfeats = batch_data_dict['imgfeats'].cuda()
      IDs = batch_data_dict['IDs']
      images = batch_data_dict['images'].cuda()
      clip_images = batch_data_dict['clip_images'].cuda()
      ascores = batch_data_dict['ascores'].cuda()

    

      # 保留具有有效评论的图像
      image_list = []
      batch_ascores_list = []

      # single
      batch_best_prob_comment_list = []
      batch_best_cs_comment_list = []
      batch_best_clip_comment_list = []
      batch_best_clip_cs_comment_list = []

      # multiple
      batch_all_multi_comment_list = []
      batch_multi_true_comment_list = []
      batch_best_multi_comment_list = []
      batch_best_multi_comment_ungroup_list = []
      batch_best_multi_cs_ungroup_comment_list = []
      batch_best_multi_cs_comment_list = []
      batch_best_multi_clip_ungroup_comment_list = []
      batch_best_multi_clip_comment_list = []
      batch_best_multi_clip_cs_ungroup_comment_list = []
      batch_best_multi_clip_cs_comment_list = []
      
      for k, ID in enumerate(IDs):
        
        raw_gen_comments = batch_valDataLoader.dataset.readGenData_from_id(os.path.join(gen_data_root, 'all'), ID)
        
        # remove bad comment
        raw_gen_comment_ss_embeddings = SSNet.module.encode(raw_gen_comments)
        raw_bad_similarity_matrix = cosine_similarity(raw_gen_comment_ss_embeddings, bad_comment_ss_embeddings)
        mask = torch.ones(len(raw_gen_comments)).cuda()
        for i in range(len(raw_gen_comments)):
          if np.max(raw_bad_similarity_matrix[i]) < 0.7:
            mask[i]=0
        
        gen_comments = []
        for i in range(len(raw_gen_comments)):
          if mask[i] == 0:
            gen_comments.append(raw_gen_comments[i])
        
        # 如果没有产生评论, 那就跳过
        if len(gen_comments) == 0:
          gen_comments.append(raw_gen_comments[0])

        

        references_captions_dict[ID], ref_cs, ref_norm_cs = batch_valDataLoader.dataset.readData_from_id(ID)
        image = images[k]
        ascore = ascores[k]
        image_list.append(image)
        batch_ascores_list.append(ascore)

        clip_image = clip_images[k]
        selector_result = {
          "ID":ID
        }
        groups = []

        gen_comment_encoding = CANet.module.tokenizer.batch_encode_plus(
            gen_comments,
            add_special_tokens=True,
            max_length=args.max_sentence_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        as_preds, cs_preds = CANet(gen_comment_encoding['input_ids'].cuda(), gen_comment_encoding['attention_mask'].cuda())
        norm_cs_preds = data_norm(cs_preds)
        clip_gen_comment_token = clip_tokenizer(gen_comments, truncate=True).cuda()
        clip_gen_logits_per_image, clip_logits_per_ref_comment = clip_model(clip_image.unsqueeze(0), clip_gen_comment_token)
        norm_clip_gen_logits_per_image = data_norm(clip_gen_logits_per_image)

        # single comment
        best_prob_comment = gen_comments[0]
        candidates_best_prob_comment_dict[ID] = [best_prob_comment]
        selector_result['best_prob_comment'] = best_prob_comment
        batch_best_prob_comment_list.append(best_prob_comment)

        _, cs_indices = torch.sort(norm_cs_preds, descending=True)
        best_cs_comment = gen_comments[cs_indices[0]]
        selector_result['best_cs_comment'] = best_cs_comment
        candidates_best_cs_comment_dict[ID] = [best_cs_comment]
        batch_best_cs_comment_list.append(best_cs_comment)

        _, clip_indices = torch.sort(norm_clip_gen_logits_per_image[0], descending=True)
        best_clip_comment = gen_comments[clip_indices[0]]
        selector_result['best_clip_comment'] = best_clip_comment
        candidates_best_clip_comment_dict[ID] = [best_clip_comment]
        batch_best_clip_comment_list.append(best_clip_comment)

        _, clip_cs_indices = torch.sort((norm_clip_gen_logits_per_image[0]+norm_cs_preds)/2, descending=True)
        best_clip_cs_comment = gen_comments[clip_cs_indices[0]]
        selector_result['best_clip_cs_comment'] = best_clip_cs_comment
        candidates_best_clip_cs_comment_dict[ID] = [best_clip_cs_comment]
        batch_best_clip_cs_comment_list.append(best_clip_cs_comment)

        ss_embeddings = SSNet.module.encode(gen_comments)
        similarity_matrix = cosine_similarity(ss_embeddings, ss_embeddings)
        mask = torch.ones(len(gen_comments)).cuda()


        
        for i in range(similarity_matrix.shape[0]):
            group = []
            row = similarity_matrix[i]
            if mask[i] != 0:
                mask[i] = 0
                group.append({'comment':gen_comments[i], 'as_pred':as_preds[i].item(), 'cs_pred':cs_preds[i].item(), 'norm_cs_pred':norm_cs_preds[i].item(), 'clip_pred':clip_gen_logits_per_image[0][i].item(), 'norm_clip_pred':norm_clip_gen_logits_per_image[0][i].item()})

                for j, similar_value in enumerate(row):
                    
                    if similar_value >= args.similarity_threshold and mask[j] != 0:
                        group.append({'comment':gen_comments[j], 'as_pred':as_preds[j].item(), 'cs_pred':cs_preds[j].item(), 'norm_cs_pred':norm_cs_preds[j].item(), 'clip_pred':clip_gen_logits_per_image[0][j].item(), 'norm_clip_pred':norm_clip_gen_logits_per_image[0][j].item()})
                        mask[j] = 0

            if len(group) > 0:
                groups.append(group)

        # group comment
        best_multi_cs_ungroup_comment_list = []
        scored_gen_comments = [{'comment':gen_comments[i], 'as_pred':as_preds[i].item(), 'cs_pred':cs_preds[i].item(), 'norm_cs_pred':norm_cs_preds[i].item(), 'clip_pred':clip_gen_logits_per_image[0][i].item(), 'norm_clip_pred':norm_clip_gen_logits_per_image[0][i].item()} for i in range(len(gen_comments))]
        for item in sorted(scored_gen_comments, key=lambda d: d['cs_pred'], reverse=True):
          if item['cs_pred'] > 1:
            best_multi_cs_ungroup_comment_list.append(item['comment'])
        
        best_multi_clip_ungroup_comment_list = []
        for item in sorted(scored_gen_comments, key=lambda d: d['norm_clip_pred'], reverse=True):
          if item['norm_clip_pred'] > 0.1:
            best_multi_clip_ungroup_comment_list.append(item['comment'])

        best_multi_clip_cs_ungroup_comment_list = []
        for item in sorted(scored_gen_comments, key=lambda d: (d['norm_clip_pred']+d['norm_cs_pred'])/2, reverse=True):
          if item['cs_pred'] > 1 or item['norm_clip_pred'] > 0.1:
            best_multi_clip_cs_ungroup_comment_list.append(item['comment'])


        best_multi_comment_list = []
        best_multi_comment_ungroup_list = gen_comments[:len(groups)]
        best_multi_cs_comment_list = []
        best_multi_clip_comment_list = []
        best_multi_clip_cs_comment_list = []

        for group in groups:
          cs_sorted_group = sorted(group, key=lambda d: d['norm_cs_pred'], reverse=True)
          clip_sorted_group = sorted(group, key=lambda d: d['norm_clip_pred'], reverse=True) 
          clip_cs_sorted_group = sorted(group, key=lambda d: (d['norm_clip_pred']+d['norm_cs_pred'])/2, reverse=True)  

          best_multi_comment_list.append(group[0])
          if cs_sorted_group[0]['cs_pred'] >= 0.5:
            best_multi_cs_comment_list.append(cs_sorted_group[0])
          if clip_sorted_group[0]['norm_clip_pred']>= 0.1:
            best_multi_clip_comment_list.append(clip_sorted_group[0])
          if (clip_cs_sorted_group[0]['norm_clip_pred']>0.1 or clip_cs_sorted_group[0]['cs_pred']) >= 0.5:
            best_multi_clip_cs_comment_list.append(clip_cs_sorted_group[0])

        
        selector_result['best_multi_comment_list'] = [item['comment'] for item in best_multi_comment_list]
        selector_result['best_multi_comment_ungroup_list'] = [item for item in best_multi_comment_ungroup_list]
        selector_result['best_multi_cs_ungroup_comment_list'] = [item for item in best_multi_cs_ungroup_comment_list]
        selector_result['best_multi_cs_comment_list'] = [item['comment'] for item in best_multi_cs_comment_list]
        selector_result['best_multi_clip_ungroup_comment_list'] = [item for item in best_multi_clip_ungroup_comment_list]
        selector_result['best_multi_clip_comment_list'] = [item['comment'] for item in best_multi_clip_comment_list]
        selector_result['best_multi_clip_cs_ungroup_comment_list'] = [item for item in best_multi_clip_cs_ungroup_comment_list]
        selector_result['best_multi_clip_cs_comment_list'] = [item['comment'] for item in best_multi_clip_cs_comment_list]
        selector_result['gen_comments'] = gen_comments
        # 保存选择器的结果
        json.dump(selector_result, open(os.path.join(gen_data_root, 'selected', f'{ID}.json'), 'w'), indent=4)

        batch_multi_true_comment_list.append(' '.join(references_captions_dict[ID]))
        batch_all_multi_comment_list.append(' '.join(gen_comments))
        batch_best_multi_comment_list.append(' '.join([item['comment'] for item in best_multi_comment_list]))
        batch_best_multi_comment_ungroup_list.append(' '.join([item for item in best_multi_comment_ungroup_list]))
        batch_best_multi_cs_ungroup_comment_list.append(' '.join(best_multi_cs_ungroup_comment_list))
        batch_best_multi_cs_comment_list.append(' '.join([item['comment'] for item in best_multi_cs_comment_list]))
        batch_best_multi_clip_ungroup_comment_list.append(' '.join(best_multi_clip_ungroup_comment_list))
        batch_best_multi_clip_comment_list.append(' '.join([item['comment'] for item in best_multi_clip_comment_list]))
        batch_best_multi_clip_cs_ungroup_comment_list.append(' '.join(best_multi_clip_cs_ungroup_comment_list))
        batch_best_multi_clip_cs_comment_list.append(' '.join([item['comment'] for item in best_multi_clip_cs_comment_list]))

        
      
      # single comment
      batch_best_prob_comment_encoding = single_maa_model.module.tokenizer.batch_encode_plus(
            batch_best_prob_comment_list,
            add_special_tokens=True,
            max_length=args.max_sentence_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

      images = torch.stack(image_list, dim=0)
      ascores = torch.stack(batch_ascores_list, dim=0)
      ascores_list.append(ascores)
      zero_images = torch.zeros((images.size(0), 3, args.imgSize, args.imgSize)).cuda()

      empty_comment = ['' for i in range(images.size(0))]
      empty_comment_encoding = CANet.module.tokenizer.batch_encode_plus(
          empty_comment,
          add_special_tokens=True,
          max_length=args.max_sentence_length,
          return_token_type_ids=False,
          padding="max_length",
          truncation=True,
          return_attention_mask=True,
          return_tensors='pt',
      )

      multi_empty_comment_encoding = CANet.module.tokenizer.batch_encode_plus(
          empty_comment,
          add_special_tokens=True,
          max_length=args.max_multi_sentence_length,
          return_token_type_ids=False,
          padding="max_length",
          truncation=True,
          return_attention_mask=True,
          return_tensors='pt',
      )

      # baseline
      iaa_as_preds = single_maa_model(images, empty_comment_encoding['input_ids'].cuda(), empty_comment_encoding['attention_mask'].cuda())
      iaa_as_preds_list.append(iaa_as_preds.squeeze(dim=1))


      best_prob_comment_taa_as_preds = single_maa_model(zero_images, batch_best_prob_comment_encoding['input_ids'].cuda(), batch_best_prob_comment_encoding['attention_mask'].cuda())
      best_prob_comment_maa_as_preds = single_maa_model(images, batch_best_prob_comment_encoding['input_ids'].cuda(), batch_best_prob_comment_encoding['attention_mask'].cuda())
      best_prob_comment_taa_as_preds_list.append(best_prob_comment_taa_as_preds.squeeze(dim=1))
      best_prob_comment_maa_as_preds_list.append(best_prob_comment_maa_as_preds.squeeze(dim=1))

      batch_best_cs_comment_encoding = single_maa_model.module.tokenizer.batch_encode_plus(
            batch_best_cs_comment_list,
            add_special_tokens=True,
            max_length=args.max_sentence_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

      best_cs_comment_taa_as_preds = single_maa_model(zero_images, batch_best_cs_comment_encoding['input_ids'].cuda(), batch_best_cs_comment_encoding['attention_mask'].cuda())
      best_cs_comment_maa_as_preds = single_maa_model(images, batch_best_cs_comment_encoding['input_ids'].cuda(), batch_best_cs_comment_encoding['attention_mask'].cuda())
      best_cs_comment_taa_as_preds_list.append(best_cs_comment_taa_as_preds.squeeze(dim=1))
      best_cs_comment_maa_as_preds_list.append(best_cs_comment_maa_as_preds.squeeze(dim=1))

      batch_best_clip_comment_encoding = single_maa_model.module.tokenizer.batch_encode_plus(
            batch_best_clip_comment_list,
            add_special_tokens=True,
            max_length=args.max_sentence_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

      best_clip_comment_taa_as_preds = single_maa_model(zero_images, batch_best_clip_comment_encoding['input_ids'].cuda(), batch_best_clip_comment_encoding['attention_mask'].cuda())
      best_clip_comment_maa_as_preds = single_maa_model(images, batch_best_clip_comment_encoding['input_ids'].cuda(), batch_best_clip_comment_encoding['attention_mask'].cuda())
      best_clip_comment_taa_as_preds_list.append(best_clip_comment_taa_as_preds.squeeze(dim=1))
      best_clip_comment_maa_as_preds_list.append(best_clip_comment_maa_as_preds.squeeze(dim=1))

      batch_best_clip_cs_comment_encoding = single_maa_model.module.tokenizer.batch_encode_plus(
            batch_best_clip_cs_comment_list,
            add_special_tokens=True,
            max_length=args.max_sentence_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

      best_clip_cs_comment_taa_as_preds = single_maa_model(zero_images, batch_best_clip_cs_comment_encoding['input_ids'].cuda(), batch_best_clip_cs_comment_encoding['attention_mask'].cuda())
      best_clip_cs_comment_maa_as_preds = single_maa_model(images, batch_best_clip_cs_comment_encoding['input_ids'].cuda(), batch_best_clip_cs_comment_encoding['attention_mask'].cuda())
      best_clip_cs_comment_taa_as_preds_list.append(best_clip_cs_comment_taa_as_preds.squeeze(dim=1))
      best_clip_cs_comment_maa_as_preds_list.append(best_clip_cs_comment_maa_as_preds.squeeze(dim=1))

      # multi comment
      batch_best_multi_comment_encoding = single_maa_model.module.tokenizer.batch_encode_plus(
            batch_best_multi_comment_list,
            add_special_tokens=True,
            max_length=args.max_multi_sentence_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
      )

      batch_all_multi_comment_encoding = single_maa_model.module.tokenizer.batch_encode_plus(
            batch_all_multi_comment_list,
            add_special_tokens=True,
            max_length=args.max_multi_sentence_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
      )


      batch_multi_true_comment_encoding = CANet.module.tokenizer.batch_encode_plus(
          batch_multi_true_comment_list,
          add_special_tokens=True,
          max_length=args.max_multi_sentence_length,
          return_token_type_ids=False,
          padding="max_length",
          truncation=True,
          return_attention_mask=True,
          return_tensors='pt',
      )

      # baseline
      multi_iaa_as_preds = multiple_maa_model(images, multi_empty_comment_encoding['input_ids'].cuda(), multi_empty_comment_encoding['attention_mask'].cuda())
      multi_true_taa_as_preds = multiple_maa_model(zero_images, batch_multi_true_comment_encoding['input_ids'].cuda(), batch_multi_true_comment_encoding['attention_mask'].cuda())
      multi_true_maa_as_preds = multiple_maa_model(images, batch_multi_true_comment_encoding['input_ids'].cuda(), batch_multi_true_comment_encoding['attention_mask'].cuda())

      all_multi_comment_taa_as_preds = multiple_maa_model(zero_images, batch_all_multi_comment_encoding['input_ids'].cuda(), batch_all_multi_comment_encoding['attention_mask'].cuda())
      all_multi_comment_maa_as_preds = multiple_maa_model(images, batch_all_multi_comment_encoding['input_ids'].cuda(), batch_all_multi_comment_encoding['attention_mask'].cuda())

      best_multi_comment_taa_as_preds = multiple_maa_model(zero_images, batch_best_multi_comment_encoding['input_ids'].cuda(), batch_best_multi_comment_encoding['attention_mask'].cuda())
      best_multi_comment_maa_as_preds = multiple_maa_model(images, batch_best_multi_comment_encoding['input_ids'].cuda(), batch_best_multi_comment_encoding['attention_mask'].cuda())
      multi_iaa_as_preds_list.append(multi_iaa_as_preds.squeeze(dim=1))
      multi_true_taa_as_preds_list.append(multi_true_taa_as_preds.squeeze(dim=1))
      multi_true_maa_as_preds_list.append(multi_true_maa_as_preds.squeeze(dim=1))
      all_multi_comment_taa_as_preds_list.append(all_multi_comment_taa_as_preds.squeeze(dim=1))
      all_multi_comment_maa_as_preds_list.append(all_multi_comment_maa_as_preds.squeeze(dim=1))
      best_multi_comment_taa_as_preds_list.append(best_multi_comment_taa_as_preds.squeeze(dim=1))
      best_multi_comment_maa_as_preds_list.append(best_multi_comment_maa_as_preds.squeeze(dim=1))



      batch_best_multi_comment_ungroup_encoding = single_maa_model.module.tokenizer.batch_encode_plus(
            batch_best_multi_comment_ungroup_list,
            add_special_tokens=True,
            max_length=args.max_multi_sentence_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
      )

      best_multi_comment_ungroup_taa_as_preds = multiple_maa_model(zero_images, batch_best_multi_comment_ungroup_encoding['input_ids'].cuda(), batch_best_multi_comment_ungroup_encoding['attention_mask'].cuda())
      best_multi_comment_ungroup_maa_as_preds = multiple_maa_model(images, batch_best_multi_comment_ungroup_encoding['input_ids'].cuda(), batch_best_multi_comment_ungroup_encoding['attention_mask'].cuda())
      best_multi_comment_ungroup_taa_as_preds_list.append(best_multi_comment_ungroup_taa_as_preds.squeeze(dim=1))
      best_multi_comment_ungroup_maa_as_preds_list.append(best_multi_comment_ungroup_maa_as_preds.squeeze(dim=1))

      batch_best_multi_cs_comment_encoding = single_maa_model.module.tokenizer.batch_encode_plus(
            batch_best_multi_cs_comment_list,
            add_special_tokens=True,
            max_length=args.max_multi_sentence_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
      )
      best_multi_cs_comment_taa_as_preds = multiple_maa_model(zero_images, batch_best_multi_cs_comment_encoding['input_ids'].cuda(), batch_best_multi_cs_comment_encoding['attention_mask'].cuda())
      best_multi_cs_comment_maa_as_preds = multiple_maa_model(images, batch_best_multi_cs_comment_encoding['input_ids'].cuda(), batch_best_multi_cs_comment_encoding['attention_mask'].cuda())
      best_multi_cs_comment_taa_as_preds_list.append(best_multi_cs_comment_taa_as_preds.squeeze(dim=1))
      best_multi_cs_comment_maa_as_preds_list.append(best_multi_cs_comment_maa_as_preds.squeeze(dim=1))

      batch_best_multi_cs_ungroup_comment_encoding = single_maa_model.module.tokenizer.batch_encode_plus(
            batch_best_multi_cs_ungroup_comment_list,
            add_special_tokens=True,
            max_length=args.max_multi_sentence_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
      )
      best_multi_cs_ungroup_comment_taa_as_preds = multiple_maa_model(zero_images, batch_best_multi_cs_ungroup_comment_encoding['input_ids'].cuda(), batch_best_multi_cs_ungroup_comment_encoding['attention_mask'].cuda())
      best_multi_cs_ungroup_comment_maa_as_preds = multiple_maa_model(images, batch_best_multi_cs_ungroup_comment_encoding['input_ids'].cuda(), batch_best_multi_cs_ungroup_comment_encoding['attention_mask'].cuda())
      best_multi_cs_ungroup_comment_taa_as_preds_list.append(best_multi_cs_ungroup_comment_taa_as_preds.squeeze(dim=1))
      best_multi_cs_ungroup_comment_maa_as_preds_list.append(best_multi_cs_ungroup_comment_maa_as_preds.squeeze(dim=1))


      batch_best_multi_clip_comment_encoding = single_maa_model.module.tokenizer.batch_encode_plus(
            batch_best_multi_clip_comment_list,
            add_special_tokens=True,
            max_length=args.max_multi_sentence_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
      )

      best_multi_clip_comment_taa_as_preds = multiple_maa_model(zero_images, batch_best_multi_clip_comment_encoding['input_ids'].cuda(), batch_best_multi_clip_comment_encoding['attention_mask'].cuda())
      best_multi_clip_comment_maa_as_preds = multiple_maa_model(images, batch_best_multi_clip_comment_encoding['input_ids'].cuda(), batch_best_multi_clip_comment_encoding['attention_mask'].cuda())
      best_multi_clip_comment_taa_as_preds_list.append(best_multi_clip_comment_taa_as_preds.squeeze(dim=1))
      best_multi_clip_comment_maa_as_preds_list.append(best_multi_clip_comment_maa_as_preds.squeeze(dim=1))

      batch_best_multi_clip_ungroup_comment_encoding = single_maa_model.module.tokenizer.batch_encode_plus(
            batch_best_multi_clip_ungroup_comment_list,
            add_special_tokens=True,
            max_length=args.max_multi_sentence_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
      )
      best_multi_clip_ungroup_comment_taa_as_preds = multiple_maa_model(zero_images, batch_best_multi_clip_ungroup_comment_encoding['input_ids'].cuda(), batch_best_multi_clip_ungroup_comment_encoding['attention_mask'].cuda())
      best_multi_clip_ungroup_comment_maa_as_preds = multiple_maa_model(images, batch_best_multi_clip_ungroup_comment_encoding['input_ids'].cuda(), batch_best_multi_clip_ungroup_comment_encoding['attention_mask'].cuda())
      best_multi_clip_ungroup_comment_taa_as_preds_list.append(best_multi_clip_ungroup_comment_taa_as_preds.squeeze(dim=1))
      best_multi_clip_ungroup_comment_maa_as_preds_list.append(best_multi_clip_ungroup_comment_maa_as_preds.squeeze(dim=1))

      batch_best_multi_clip_cs_comment_encoding = single_maa_model.module.tokenizer.batch_encode_plus(
            batch_best_multi_clip_cs_comment_list,
            add_special_tokens=True,
            max_length=args.max_multi_sentence_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
      )

      best_multi_clip_cs_comment_taa_as_preds = multiple_maa_model(zero_images, batch_best_multi_clip_cs_comment_encoding['input_ids'].cuda(), batch_best_multi_clip_cs_comment_encoding['attention_mask'].cuda())
      best_multi_clip_cs_comment_maa_as_preds = multiple_maa_model(images, batch_best_multi_clip_cs_comment_encoding['input_ids'].cuda(), batch_best_multi_clip_cs_comment_encoding['attention_mask'].cuda())
      best_multi_clip_cs_comment_taa_as_preds_list.append(best_multi_clip_cs_comment_taa_as_preds.squeeze(dim=1))
      best_multi_clip_cs_comment_maa_as_preds_list.append(best_multi_clip_cs_comment_maa_as_preds.squeeze(dim=1))


      batch_best_multi_clip_cs_ungroup_comment_encoding = single_maa_model.module.tokenizer.batch_encode_plus(
            batch_best_multi_clip_cs_ungroup_comment_list,
            add_special_tokens=True,
            max_length=args.max_multi_sentence_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
      )
      best_multi_clip_cs_ungroup_comment_taa_as_preds = multiple_maa_model(zero_images, batch_best_multi_clip_cs_ungroup_comment_encoding['input_ids'].cuda(), batch_best_multi_clip_cs_ungroup_comment_encoding['attention_mask'].cuda())
      best_multi_clip_cs_ungroup_comment_maa_as_preds = multiple_maa_model(images, batch_best_multi_clip_cs_ungroup_comment_encoding['input_ids'].cuda(), batch_best_multi_clip_cs_ungroup_comment_encoding['attention_mask'].cuda())
      best_multi_clip_cs_ungroup_comment_taa_as_preds_list.append(best_multi_clip_cs_ungroup_comment_taa_as_preds.squeeze(dim=1))
      best_multi_clip_cs_ungroup_comment_maa_as_preds_list.append(best_multi_clip_cs_ungroup_comment_maa_as_preds.squeeze(dim=1))


      if dist.get_rank()==0:
        pbar.update()



    # captions metrics
    eval_cap.evaluate(candidates_best_prob_comment_dict, references_captions_dict)

    # GT
    ascores = torch.cat(ascores_list, dim=0)
    all_ascores_list = [torch.zeros_like(ascores) for _ in range(dist.get_world_size())]
    dist.all_gather(all_ascores_list, ascores)
    all_ascores = torch.cat(all_ascores_list, dim=0)

    # single 
    iaa_metric_results_dict = get_score_metric_from_list(iaa_as_preds_list, all_ascores)

    best_prob_comment_taa_metric_results_dict = get_score_metric_from_list(best_prob_comment_taa_as_preds_list, all_ascores)
    best_prob_comment_maa_metric_results_dict = get_score_metric_from_list(best_prob_comment_maa_as_preds_list, all_ascores)

    best_cs_comment_taa_metric_results_dict = get_score_metric_from_list(best_cs_comment_taa_as_preds_list, all_ascores)
    best_cs_comment_maa_metric_results_dict = get_score_metric_from_list(best_cs_comment_maa_as_preds_list, all_ascores)

    best_clip_comment_taa_metric_results_dict = get_score_metric_from_list(best_clip_comment_taa_as_preds_list, all_ascores)
    best_clip_comment_maa_metric_results_dict = get_score_metric_from_list(best_clip_comment_maa_as_preds_list, all_ascores)

    best_clip_cs_comment_taa_metric_results_dict = get_score_metric_from_list(best_clip_cs_comment_taa_as_preds_list, all_ascores)
    best_clip_cs_comment_maa_metric_results_dict = get_score_metric_from_list(best_clip_cs_comment_maa_as_preds_list, all_ascores)

    # multi
    multi_iaa_metric_results_dict = get_score_metric_from_list(multi_iaa_as_preds_list, all_ascores)
    multi_true_taa_metric_results_dict = get_score_metric_from_list(multi_true_taa_as_preds_list, all_ascores)
    multi_true_maa_metric_results_dict = get_score_metric_from_list(multi_true_maa_as_preds_list, all_ascores)

    all_multi_comment_taa_metric_results_dict = get_score_metric_from_list(all_multi_comment_taa_as_preds_list, all_ascores)
    all_multi_comment_maa_metric_results_dict = get_score_metric_from_list(all_multi_comment_maa_as_preds_list, all_ascores)

    best_multi_comment_taa_metric_results_dict = get_score_metric_from_list(best_multi_comment_taa_as_preds_list, all_ascores)
    best_multi_comment_maa_metric_results_dict = get_score_metric_from_list(best_multi_comment_maa_as_preds_list, all_ascores)

    best_multi_comment_taa_metric_results_dict = get_score_metric_from_list(best_multi_comment_taa_as_preds_list, all_ascores)
    best_multi_comment_maa_metric_results_dict = get_score_metric_from_list(best_multi_comment_maa_as_preds_list, all_ascores)

    best_multi_comment_ungroup_taa_metric_results_dict = get_score_metric_from_list(best_multi_comment_ungroup_taa_as_preds_list, all_ascores)
    best_multi_comment_ungroup_maa_metric_results_dict = get_score_metric_from_list(best_multi_comment_ungroup_maa_as_preds_list, all_ascores)

    best_multi_cs_comment_taa_metric_results_dict = get_score_metric_from_list(best_multi_cs_comment_taa_as_preds_list, all_ascores)
    best_multi_cs_comment_maa_metric_results_dict = get_score_metric_from_list(best_multi_cs_comment_maa_as_preds_list, all_ascores)

    best_multi_cs_ungroup_comment_taa_metric_results_dict = get_score_metric_from_list(best_multi_cs_ungroup_comment_taa_as_preds_list, all_ascores)
    best_multi_cs_ungroup_comment_maa_metric_results_dict = get_score_metric_from_list(best_multi_cs_ungroup_comment_maa_as_preds_list, all_ascores)

    best_multi_clip_comment_taa_metric_results_dict = get_score_metric_from_list(best_multi_clip_comment_taa_as_preds_list, all_ascores)
    best_multi_clip_comment_maa_metric_results_dict = get_score_metric_from_list(best_multi_clip_comment_maa_as_preds_list, all_ascores)

    best_multi_clip_ungroup_comment_taa_metric_results_dict = get_score_metric_from_list(best_multi_clip_ungroup_comment_taa_as_preds_list, all_ascores)
    best_multi_clip_ungroup_comment_maa_metric_results_dict = get_score_metric_from_list(best_multi_clip_ungroup_comment_maa_as_preds_list, all_ascores)

    best_multi_clip_cs_comment_taa_metric_results_dict = get_score_metric_from_list(best_multi_clip_cs_comment_taa_as_preds_list, all_ascores)
    best_multi_clip_cs_comment_maa_metric_results_dict = get_score_metric_from_list(best_multi_clip_cs_comment_maa_as_preds_list, all_ascores)

    best_multi_clip_cs_ungroup_comment_taa_metric_results_dict = get_score_metric_from_list(best_multi_clip_cs_ungroup_comment_taa_as_preds_list, all_ascores)
    best_multi_clip_cs_ungroup_comment_maa_metric_results_dict = get_score_metric_from_list(best_multi_clip_cs_ungroup_comment_maa_as_preds_list, all_ascores)


    eval_result_dict = {
      'cap_metric':eval_cap.eval, 
      'iaa_metric_results_dict':iaa_metric_results_dict,
      'best_prob_comment_taa_metric_results_dict':best_prob_comment_taa_metric_results_dict, 
      'best_prob_comment_maa_metric_results_dict':best_prob_comment_maa_metric_results_dict, 
      'best_cs_comment_taa_metric_results_dict':best_cs_comment_taa_metric_results_dict, 
      'best_cs_comment_maa_metric_results_dict':best_cs_comment_maa_metric_results_dict, 
      'best_clip_comment_taa_metric_results_dict':best_clip_comment_taa_metric_results_dict, 
      'best_clip_comment_maa_metric_results_dict':best_clip_comment_maa_metric_results_dict, 
      'best_clip_cs_comment_taa_metric_results_dict':best_clip_cs_comment_taa_metric_results_dict, 
      'best_clip_cs_comment_maa_metric_results_dict':best_clip_cs_comment_maa_metric_results_dict, 
      'multi_iaa_metric_results_dict':multi_iaa_metric_results_dict,
      'multi_true_taa_metric_results_dict':multi_true_taa_metric_results_dict, 
      'multi_true_maa_metric_results_dict':multi_true_maa_metric_results_dict,  
      'all_multi_comment_taa_metric_results_dict':all_multi_comment_taa_metric_results_dict,
      'all_multi_comment_maa_metric_results_dict':all_multi_comment_maa_metric_results_dict,
      'best_multi_comment_taa_metric_results_dict':best_multi_comment_taa_metric_results_dict, 
      'best_multi_comment_maa_metric_results_dict':best_multi_comment_maa_metric_results_dict, 
      'best_multi_comment_ungroup_taa_metric_results_dict':best_multi_comment_ungroup_taa_metric_results_dict, 
      'best_multi_comment_ungroup_maa_metric_results_dict':best_multi_comment_ungroup_maa_metric_results_dict, 
      'best_multi_cs_comment_taa_metric_results_dict':best_multi_cs_comment_taa_metric_results_dict, 
      'best_multi_cs_comment_maa_metric_results_dict':best_multi_cs_comment_maa_metric_results_dict, 
      'best_multi_cs_ungroup_comment_taa_metric_results_dict':best_multi_cs_ungroup_comment_taa_metric_results_dict,
      'best_multi_cs_ungroup_comment_maa_metric_results_dict':best_multi_cs_ungroup_comment_maa_metric_results_dict,
      'best_multi_clip_comment_taa_metric_results_dict':best_multi_clip_comment_taa_metric_results_dict, 
      'best_multi_clip_comment_maa_metric_results_dict':best_multi_clip_comment_maa_metric_results_dict, 
      'best_multi_clip_ungroup_comment_taa_metric_results_dict':best_multi_clip_ungroup_comment_taa_metric_results_dict,
      'best_multi_clip_ungroup_comment_maa_metric_results_dict':best_multi_clip_ungroup_comment_maa_metric_results_dict,     
      'best_multi_clip_cs_comment_taa_metric_results_dict':best_multi_clip_cs_comment_taa_metric_results_dict, 
      'best_multi_clip_cs_comment_maa_metric_results_dict':best_multi_clip_cs_comment_maa_metric_results_dict, 
      'best_multi_clip_cs_ungroup_comment_taa_metric_results_dict':best_multi_clip_ungroup_comment_taa_metric_results_dict,
      'best_multi_clip_cs_ungroup_comment_maa_metric_results_dict':best_multi_clip_ungroup_comment_maa_metric_results_dict,    
    }

    # convert to tensor and reduce to gpu0
    for metric_dict in eval_result_dict.values():
        for metric, score in metric_dict.items():
            metric_dict[metric]= torch.tensor(score, device='cuda')
            dist.reduce(metric_dict[metric], dst=0, op=dist.ReduceOp.SUM)

    dist.barrier()
    if dist.get_rank() == 0:  
      for metric_dict in eval_result_dict.values():
          for metric, score in metric_dict.items():
              # 由于做了reduce, 这里除以线程数
              if torch.is_tensor(score):
                metric_dict[metric] = (score/dist.get_world_size()).item()
              else:
                metric_dict[metric] = score/dist.get_world_size()

              print(f'{metric}: {metric_dict[metric]:.4f}')
      print(eval_result_dict)
      with open(os.path.join(gen_data_root, 'result.json'), 'w') as f:
        json.dump(eval_result_dict, f, indent=4)
    
    
    print(f'result dict has saved in {gen_data_root}')
    print('end')
