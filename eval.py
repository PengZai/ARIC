from tqdm import tqdm
import torch
from metrics.captions_metrics import EvalCap
from metrics.score_metrics import score_metrics
import os
import torch.distributed as dist
from models.comment_score_model import CommentScoreNet
from torch.nn.parallel import DistributedDataParallel as DDP
import json

def eval_for_epoch(model, dataloader, epoch, num_beams, args):
    
    rank = dist.get_rank()

    output_size = num_beams

    
    save_gen_caption_dir =  args.batch_weighted_ce+'_generation_comment_for_eval'

    if rank == 0:
        if not os.path.exists(save_gen_caption_dir):
            os.mkdir(save_gen_caption_dir)
        if not os.path.exists(os.path.join(save_gen_caption_dir, f'epoch={epoch}')):
            os.mkdir(os.path.join(save_gen_caption_dir, f'epoch={epoch}'))
        if not os.path.exists(os.path.join(save_gen_caption_dir, f'epoch={epoch}')):
            os.mkdir(os.path.join(save_gen_caption_dir, f'epoch={epoch}'))
        if not os.path.exists(os.path.join(save_gen_caption_dir, f'epoch={epoch}', f'num_beams={num_beams}')):
            os.mkdir(os.path.join(save_gen_caption_dir, f'epoch={epoch}', f'num_beams={num_beams}'))

    save_single_gen_caption_dir = os.path.join(save_gen_caption_dir, f'epoch={epoch}', f'num_beams={num_beams}', 'single')
    if rank == 0:
        if not os.path.exists(save_single_gen_caption_dir):
            os.mkdir(save_single_gen_caption_dir)

    save_all_gen_caption_dir = os.path.join(save_gen_caption_dir, f'epoch={epoch}', f'num_beams={num_beams}', 'all')
    if rank == 0:
        if not os.path.exists(save_all_gen_caption_dir):
            os.mkdir(save_all_gen_caption_dir)

    dist.barrier()

    eval_cap = EvalCap()
    CANet = CommentScoreNet(args)
    CANet_checkpoint = torch.load('saved_models/comment_assessment/%s_comment_assessment.pth' % (args.comment_assessment_model), map_location={'cuda:0':f'cuda:{rank}'})
    CANet.load_state_dict(CANet_checkpoint['state_dict'], strict=False)
    CANet = CANet.cuda()
    CANet = DDP(CANet, device_ids=[rank], find_unused_parameters=True)
    dist.barrier()

    candidates_captions_dict = {}
    references_captions_dict = {}

    dataset = dataloader.dataset
    gpt2_tokenizer = dataset.gpt2_tokenizer
 
    
    desc_template = 'Epoch %d - eval ' %(epoch)

    ascores_list = []
    as_preds_list = []
    cs_preds_list = []
    
    model.eval()
    with torch.no_grad():
        
        rank = dist.get_rank()
        if rank == 0:
            pbar = tqdm(desc=desc_template, unit='it', total=len(dataloader))
            
        for i, batch_data_dict in enumerate(dataloader):
            
            generation_comment_list = []
            imgfeats = batch_data_dict['imgfeats'].cuda()
            IDs = batch_data_dict['IDs']
            images = batch_data_dict['images'].cuda()
            ascores = batch_data_dict['ascores'].cuda()

            # batch_outs, log_probs = model.beam_search(imgfeats, args.max_sentence_length, text_tokenizer.eos_token_id,
            #                                         num_beams, out_size=output_size)

            batch_outs, log_probs = model.module.beam_search(imgfeats, args.max_sentence_length, gpt2_tokenizer.eos_token_id,
                                                    num_beams, out_size=output_size)
            
            for k, ID in enumerate(IDs):

                candidate_captions = []
                ascore = ascores[k]
                outs = batch_outs[k]
                image = images[k]
                
                raw_candidate_captions = gpt2_tokenizer.batch_decode(outs, skip_special_tokens=False)

                # remove eos token
                for k, c in enumerate(raw_candidate_captions):
                    c = ' '.join(c.split(' ')[:-1])
                    # print('ID:%s, candidate captions:%s, scores: %.6f'%(ID, c, scores[k].item()))  
                    candidate_captions.append(c)

                json.dump({'ID':ID, 'gen_comments':candidate_captions}, open(os.path.join(save_all_gen_caption_dir, f'{ID}.json'), 'w'), indent=4)
                generation_comment_list.append(candidate_captions[0])

            candidate_captions_encoding = CANet.module.tokenizer.batch_encode_plus(
            list(generation_comment_list),
            add_special_tokens=True,
            max_length=args.max_sentence_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
            )

            as_preds, cs_preds = CANet(candidate_captions_encoding['input_ids'].cuda(), candidate_captions_encoding['attention_mask'].cuda())
           

            as_preds_list.append(as_preds)
            ascores_list.append(ascores)
            cs_preds_list.append(cs_preds)

            for k, ID in enumerate(IDs):
                candidates_captions_dict[ID] = [generation_comment_list[k]]
                references_captions_dict[ID], ref_cs, ref_norm_cs = dataset.readData_from_id(ID)

            rank = dist.get_rank() 
            if rank == 0:
                pbar.update()
 
            if i > 2:
                break

    cs_preds = torch.cat(cs_preds_list, dim=0)
    as_preds = torch.cat(as_preds_list, dim=0)
    ascores = torch.cat(ascores_list, dim=0)

    all_as_preds_list = [torch.zeros_like(as_preds) for _ in range(dist.get_world_size())]
    all_ascores_list = [torch.zeros_like(ascores) for _ in range(dist.get_world_size())]

    dist.all_gather(all_as_preds_list, as_preds)
    dist.all_gather(all_ascores_list, ascores)
    all_as_preds = torch.cat(all_as_preds_list, dim=0)
    all_ascores = torch.cat(all_ascores_list, dim=0)

    
    eval_cap.evaluate(candidates_captions_dict, references_captions_dict)

    # 保存用于eval的结果
    for i, (ID, comment) in enumerate(candidates_captions_dict.items()):
        candidates_captions_dict[ID] = {'gen_comment':comment, 'ascore':ascores[i].item(), 'as_pred':as_preds[i].item(), 'cs_pred':cs_preds[i].item()}
    json.dump(candidates_captions_dict, open(os.path.join(save_single_gen_caption_dir, f'gen_single_comments_rank_{rank}.json'), 'w'), indent=4)


    as_metric_results_dict = score_metrics(all_as_preds.cpu(), all_ascores.cpu(), mean=5.)


    eval_result_dict = {'cap_metric':eval_cap.eval, 'as_metric':as_metric_results_dict, 'cs_metric':{'cs_mean':cs_preds.mean().item()}}

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
                metric_dict[metric] = (score/dist.get_world_size()).item()
                print(f'{metric}: {metric_dict[metric]:.4f}')

        
    return eval_result_dict

 

