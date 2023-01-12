import os
from config import get_args
import torch
import torch.nn as nn
import numpy as np
import logging

from data import DPC2022, DPC2022_for_generate
from torch.utils.data import DataLoader
import pandas as pd
# from transformers import AdamW
from models.transformer import Transformer_visualgpt, VisualEncoder, ScaledDotProductAttention
from loss import BatchWeightedCE
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from train import train_for_epoch
from eval import eval_for_epoch

if __name__ == '__main__':

    args = get_args()
    print(args)

    # logging 初始化要在distribute之前
    logging.basicConfig(filename=os.path.join(args.log_dir, args.batch_weighted_ce, args.use_model+'_image_caption.txt'),
         level=logging.INFO,
         datefmt='%a, %d %b %Y %H:%M:%S'
    )
    logging.info(args)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(rank)


    # gpt2_tokenizer.add_special_tokens({'sep_token': "<|sepftext|>"})
    # gpt2_tokenizer.add_special_tokens({'pad_token': "+="})
    

    # Model and dataloaders
    encoder =VisualEncoder(3, 0, attention_module=ScaledDotProductAttention)
    model = Transformer_visualgpt(encoder)


    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    gen_ce_loss_fn = nn.NLLLoss(ignore_index=model.tokenizer.pad_token_id)
    loss_fn = BatchWeightedCE(ignore_index = model.tokenizer.pad_token_id, args=args)

    train_split_by_val_scored_comment_id_pair_df = pd.read_csv(os.path.join(args.comments_root, 'train_split_by_val_scored_comment_id_pair.csv'))
    val_df = pd.read_csv(os.path.join(args.test_and_val_root, 'val.csv'))
    
    train_dataset = DPC2022(train_split_by_val_scored_comment_id_pair_df, model.tokenizer, args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    batch_trainDataLoader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=4, collate_fn=train_dataset.collate_fn, drop_last=True, sampler=train_sampler)

    val_dataset = DPC2022_for_generate(val_df, model.tokenizer , args)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    batch_valDataLoader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=4, collate_fn=val_dataset.collate_fn, drop_last=False, sampler=val_sampler)


    start_epoch = 0
    eval_result_dict_list = []

    if args.train_from_scratch == True:
        print('training from scratch')

    else:
        print('training from last')
        model_fname = '%s/%s/%s_image_caption_last_model.pth' % (args.models_dir, args.batch_weighted_ce ,args.use_model)
        # model_fname = '%s/%s/%s_image_caption_epoch_14_model.pth' % (args.models_dir, args.batch_weighted_ce ,args.use_model)

        if os.path.exists(model_fname):
            # model_checkpoint = torch.load(model_fname, map_location='cuda:0')
            model_checkpoint = torch.load(model_fname, map_location={'cuda:0':f'cuda:{rank}'})
            torch.set_rng_state(model_checkpoint['torch_rng_state'].cpu())
            torch.cuda.set_rng_state(model_checkpoint['cuda_rng_state'].cpu())
            np.random.set_state(model_checkpoint['numpy_rng_state'])
            random.setstate(model_checkpoint['random_rng_state'])
            model.load_state_dict(model_checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(model_checkpoint['optimizer'])
            # 要把optimizer参数放到gpu, 不然报错
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

            start_epoch = model_checkpoint['epoch']
            print(f"start_epoch:{start_epoch}, eval_result_dict_list:{model_checkpoint['eval_result_dict_list']}")
                
            # print('generator loadingg %s' %(model_fname))
        
        else:
            
            print('no exits %s, training from scratch' % (model_fname))

    # 等待加载完成
    # dist.barrier()

    # 多gpu
    model.cuda()
    model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    training_loss = 0
    world_size = dist.get_world_size()
    for epoch in range(start_epoch, start_epoch+args.max_epoch):
        
        eval_result_dict_list = []

        training_loss = train_for_epoch(model, batch_trainDataLoader, loss_fn, optimizer, epoch, args)
        print('epoch %d, model training_loss : %.4f'%(epoch, training_loss))

        if epoch >= start_epoch+0:
            
            beam_eval_result_dict = eval_for_epoch(model, batch_valDataLoader, epoch, 5, args)
            eval_result_dict_list.append({'num_beams':5, 'eval_result_dict':beam_eval_result_dict})
            
            dist.barrier()

            # beam_eval_result_dict = eval_for_epoch(model, batch_valDataLoader, epoch, 128, args)
            # eval_result_dict_list.append({'num_beams':128, 'eval_result_dict':beam_eval_result_dict})
            
            # dist.barrier()


            rank = dist.get_rank()
            if rank == 0:   
                # All processes should see same parameters as they all start from same
                # random parameters and gradients are synchronized in backward passes.
                # Therefore, saving it in one process is sufficient.

                logging.info('epoch %d --- caption evaluation ---'%(epoch))
                logging.info('epoch %d, training_loss : %.4f'%(epoch, training_loss))
                print('epoch %d, training_loss : %.4f'%(epoch, training_loss))

                for item in eval_result_dict_list:
                    logging.info(f'epoch {epoch}, num_beams {item["num_beams"]}')
                    eval_result_dict = item['eval_result_dict']
                    for metric_dict in eval_result_dict.values():
                        for metric, score in metric_dict.items():
                            print(f'{metric}: {metric_dict[metric]:.4f}')
                            logging.info('epoch %d, %s : %.4f'%(epoch, metric, metric_dict[metric]))
              
                torch.save({
                    'torch_rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                    'random_rng_state': random.getstate(),
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'eval_result_dict_list': eval_result_dict_list,
                }, '%s/%s/%s_image_caption_last_model.pth' % (args.models_dir, args.batch_weighted_ce ,args.use_model))


                torch.save({
                    'torch_rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                    'random_rng_state': random.getstate(),
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'eval_result_dict_list': eval_result_dict_list,
                }, '%s/%s/%s_image_caption_epoch_%d_model.pth' % (args.models_dir, args.batch_weighted_ce ,args.use_model, epoch))

            # 等待模型保存完毕
            dist.barrier()
                


