import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from models.comment_score_model import CommentScoreNet
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import copy
import json
import pandas as pd


class Rollout(nn.Module):

    "Roll-out policy"

    def __init__(self, image_caption_model, args):
        super(Rollout, self).__init__()
        
        self.args = args
        self.generate_train_list = []
        self.MAX_SAMPLE_NUM = 4
        self.gpt2_tokenizer = image_caption_model.module.tokenizer
        rank = dist.get_rank()
        self.image_caption_model = image_caption_model
        sentence_similarity_model = SentenceTransformer(os.path.join(args.huggingface_model_root, 'all-MiniLM-L6-v2'))
        sentence_similarity_model = sentence_similarity_model.cuda()
        sentence_similarity_model.eval()
        self.SSNet = DDP(sentence_similarity_model, device_ids=[rank], find_unused_parameters=True)


        CANet = CommentScoreNet(args)
        CANet_checkpoint = torch.load('saved_models/comment_assessment/%s_comment_assessment.pth' % (args.comment_assessment_model), map_location={'cuda:0':f'cuda:{rank}'})
        CANet.load_state_dict(CANet_checkpoint['state_dict'], strict=False)
        CANet = CANet.cuda()
        CANet.eval()
        CANet = DDP(CANet, device_ids=[rank], find_unused_parameters=True)
        self.CANet = CANet


        # 生成训练数据保存路径
        if not os.path.exists(args.generation_train_comment_root):
            os.mkdir(args.generation_train_comment_root)
            
        self.gen_train_best_comment_df_dir = os.path.join(args.generation_train_comment_root, 'gen_train_best_comment_df')
        if not os.path.exists(self.gen_train_best_comment_df_dir):
            os.mkdir(self.gen_train_best_comment_df_dir)

        self.gen_train_raw_comment_dir = os.path.join(args.generation_train_comment_root, 'gen_train_raw_comment')
        if not os.path.exists(self.gen_train_raw_comment_dir):
            os.mkdir(self.gen_train_raw_comment_dir)
        
        self.gen_train_best_comment_dir = os.path.join(args.generation_train_comment_root, 'gen_train_best_comment')
        if not os.path.exists(self.gen_train_best_comment_dir):
            os.mkdir(self.gen_train_best_comment_dir)

    def grouping(self, comments):

        # 挑选相似的comment为一组, 已经被挑选成组的comment不会再次被挑选
       
        groups = []
        comment_encoding = self.CANet.module.tokenizer.batch_encode_plus(
            comments,
            add_special_tokens=True,
            max_length=self.args.max_sentence_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        as_preds, cs_preds = self.CANet(comment_encoding['input_ids'].cuda(), comment_encoding['attention_mask'].cuda())
        ss_embeddings = self.SSNet.module.encode(comments)
        similarity_matrix = cosine_similarity(ss_embeddings, ss_embeddings)
        mask = torch.ones_like(as_preds)


        
        for i in range(similarity_matrix.shape[0]):
            group = []
            row = similarity_matrix[i]
            if mask[i] != 0:
                mask[i] = 0
                group.append({'comment':comments[i], 'as_pred':as_preds[i].item(), 'cs_pred':cs_preds[i].item()})

                for j, similar_value in enumerate(row):
                    
                    if similar_value >= self.args.similarity_threshold and mask[j] != 0:
                        group.append({'comment':comments[j], 'as_pred':as_preds[j].item(), 'cs_pred':cs_preds[j].item()})
                        mask[j] = 0

            if len(group) > 0:
                groups.append(group)
            
        return groups

    def comment_selector(self, ascore, groups):

        comments = []
        best_comments = []
        for i, group in enumerate(groups):
            best_comprehensive_score = 0
            for j, comment_dict in enumerate(group):
                comprehensive_score = comment_dict['cs_pred']/(np.abs(comment_dict['as_pred']-ascore.item()) + 1e-6)
                comment_dict['abs_diff'] = np.abs(comment_dict['as_pred']-ascore.item())
                comment_dict['com_score'] = comprehensive_score
                comments.append(comment_dict)
                if best_comprehensive_score <= comprehensive_score:
                    best_comprehensive_score = comprehensive_score
                    best_comment_dict = comment_dict
                    
            # 超过阈值才保存
            if best_comment_dict['cs_pred'] >= 0 and best_comment_dict['abs_diff'] >= 0:
                best_comments.append(best_comment_dict)

        return best_comments, comments

    # sample+beamsearch还是有点问题, 明明大家输入都是I, 为什么sample后输出是rubbish,
    # 直接beamsearch的就是正常的?
    def test_single(self, imgfeats, IDs, ascores):
        num_beams = 16
        # bs_out_size > 1 and bs_out_size < num_beams
        bs_out_size = 16
        batch_sample_outputs = None
        with torch.no_grad():

            
            # play a game using sample
            # prev_ids = self.image_caption_model.module.sample(imgfeats, i)
            prev_ids = self.gpt2_tokenizer.batch_encode_plus([
                'Nice detail here and', 
                'Nice detail here and',
            ], return_attention_mask=False, return_tensors='pt')['input_ids'].cuda()


            beam_search_outs, beam_search_log_probs = self.image_caption_model.module.beam_search(imgfeats, self.args.max_sentence_length, self.gpt2_tokenizer.eos_token_id,
                                num_beams, prev_ids, out_size=bs_out_size)

            if prev_ids == None:
                    sample_outputs = beam_search_outs
                    batch_sample_outputs = sample_outputs
            else:
                prev_ids = prev_ids.unsqueeze(dim=1).expand(prev_ids.size(0), bs_out_size, prev_ids.size(-1))
                sample_outputs = torch.cat([prev_ids, beam_search_outs], dim=-1)
            
            if batch_sample_outputs == None:
                batch_sample_outputs = sample_outputs
            else:
                batch_sample_outputs = torch.cat((batch_sample_outputs, sample_outputs), dim=1)


            for i, sample_outputs in enumerate(batch_sample_outputs):
                candidate_captions = []
                ID = IDs[i]
                ascore = ascores[i]
                raw_candidate_captions = self.gpt2_tokenizer.batch_decode(sample_outputs, skip_special_tokens=False)

                # remove eos token
                for k, c in enumerate(raw_candidate_captions):
                    c = ' '.join(c.split(' ')[:-1])
                    # print('ID:%s, candidate captions:%s, scores: %.6f'%(ID, c, scores[k].item()))  
                    candidate_captions.append(c)
                print('end')


    def test(self, imgfeats, IDs, ascores):
        

        num_beams = 5
        # bs_out_size > 1 and bs_out_size < num_beams
        bs_out_size = 5
        repeat_num = 2
        batch_sample_outputs = None
        with torch.no_grad():

            for i in range(0, 10):
                # play a game using sample
                if i == 0:
                    prev_ids = self.image_caption_model.module.sample(imgfeats, 0)
                else:
                    prev_ids = self.image_caption_model.module.sample(imgfeats, 1)
                beam_search_outs, beam_search_log_probs = self.image_caption_model.module.beam_search(imgfeats, self.args.max_sentence_length, self.gpt2_tokenizer.eos_token_id,
                                    num_beams, prev_ids, out_size=bs_out_size)

                if prev_ids == None:
                    sample_outputs = beam_search_outs
                else:
                    prev_ids = prev_ids.unsqueeze(dim=1).expand(prev_ids.size(0), bs_out_size, prev_ids.size(-1))
                    sample_outputs = torch.cat([prev_ids, beam_search_outs], dim=-1)
                
                if batch_sample_outputs == None:
                    batch_sample_outputs = sample_outputs
                else:
                    batch_sample_outputs = torch.cat((batch_sample_outputs, sample_outputs), dim=1)


            for i, sample_outputs in enumerate(batch_sample_outputs):
                candidate_captions = []
                ID = IDs[i]
                ascore = ascores[i]
                raw_candidate_captions = self.gpt2_tokenizer.batch_decode(sample_outputs, skip_special_tokens=False)

                # remove eos token
                for k, c in enumerate(raw_candidate_captions):
                    c = ' '.join(c.split(' ')[:-1])
                    # print('ID:%s, candidate captions:%s, scores: %.6f'%(ID, c, scores[k].item()))  
                    candidate_captions.append(c)
                print('end')


    def out(self, imgfeats, IDs, ascores):
        

        num_beams = 5
        # bs_out_size > 1 and bs_out_size < num_beams
        bs_out_size = 5
        with torch.no_grad():

            for i in range(self.MAX_SAMPLE_NUM):
                # play a game using sample
                prev_ids = self.image_caption_model.module.sample(imgfeats, i)
                beam_search_outs, beam_search_log_probs = self.image_caption_model.module.beam_search(imgfeats, self.args.max_sentence_length, self.gpt2_tokenizer.eos_token_id,
                                    num_beams, prev_ids, out_size=bs_out_size)
                if prev_ids == None:
                    sample_outputs = beam_search_outs
                    batch_sample_outputs = sample_outputs
                else:
                    prev_ids = prev_ids.unsqueeze(dim=1).expand(prev_ids.size(0), bs_out_size, prev_ids.size(-1))
                    sample_outputs = torch.cat([prev_ids, beam_search_outs], dim=-1)
                    batch_sample_outputs = torch.cat((batch_sample_outputs, sample_outputs), dim=1)

            for i, sample_outputs in enumerate(batch_sample_outputs):
                candidate_captions = []
                ID = IDs[i]
                ascore = ascores[i]
                raw_candidate_captions = self.gpt2_tokenizer.batch_decode(sample_outputs, skip_special_tokens=False)

                # remove eos token
                for k, c in enumerate(raw_candidate_captions):
                    c = ' '.join(c.split(' ')[:-1])
                    # print('ID:%s, candidate captions:%s, scores: %.6f'%(ID, c, scores[k].item()))  
                    candidate_captions.append(c)

                groups = self.grouping(candidate_captions)
                best_comments, comments = self.comment_selector(ascore, groups)
                
                self.generate_train_list.append({
                  'ID':ID,
                  'ascore':ascore.item(),
                  'best_comments':best_comments,
                  'comments':comments,
                })


            return self.generate_train_list


    def save(self, epoch):
        
        best_comment_training_list = []
        gen_train_raw_comment_for_epoch_dir = os.path.join(self.gen_train_raw_comment_dir, f'epoch_{epoch}')
        if not os.path.exists(gen_train_raw_comment_for_epoch_dir):
            os.mkdir(gen_train_raw_comment_for_epoch_dir)
        gen_train_best_comment_for_epoch_dir = os.path.join(self.gen_train_best_comment_dir, f'epoch_{epoch}')
        if not os.path.exists(gen_train_best_comment_for_epoch_dir):
            os.mkdir(gen_train_best_comment_for_epoch_dir)

        # index,ID,comment,avg_all_users_score,sum_score
        for row in self.generate_train_list:

            # raw comment
            json.dump({
                'ID':row['ID'],
                'comments':row['comments'],
                'avg_all_users_score':row['ascore'],
            }
            , open(os.path.join(gen_train_raw_comment_for_epoch_dir, f'{row["ID"]}.json'), 'w'), indent=4)
            
            # best comment
            json.dump(row, open(os.path.join(gen_train_best_comment_for_epoch_dir, f'{row["ID"]}.json'), 'w'), indent=4)
            
            # best comment df
            for best_comments_dict in row['best_comments']:
                best_comment_training_list.append({
                    'ID':row['ID'],
                    'comments':best_comments_dict['comment'],
                    'avg_all_users_score':row['ascore'],
                    'cs_pred':best_comments_dict['cs_pred'],
                    'as_pred':best_comments_dict['as_pred'],
                })

        best_comment_training_df = pd.DataFrame(best_comment_training_list)
        best_comment_training_df.to_csv(os.path.join(self.gen_train_best_comment_df_dir, 'gen_train_best_comment_df.csv'))

        print('end')


def rollout_for_epoch(image_caption_model, dataloader, epoch, args):

    generate_train_list = []
    rank = dist.get_rank()
    image_caption_model.eval()

    rollout = Rollout(image_caption_model, args)

    desc_template = 'rollout epoch - %d ' %(epoch)

    with torch.no_grad():
        
        if rank == 0:
            pbar = tqdm(desc=desc_template, unit='it', total=len(dataloader))

        for i, batch_data_dict in enumerate(dataloader):
            imgfeats = batch_data_dict['imgfeats'].cuda()
            IDs = batch_data_dict['IDs']
            ascores = batch_data_dict['ascores']

            # rollout.out(imgfeats, IDs, ascores)
            
            rollout.test(imgfeats, IDs, ascores)

            if i > 1:
                break
            if rank == 0:
                pbar.update()
        
        rollout.save(epoch)
        
            