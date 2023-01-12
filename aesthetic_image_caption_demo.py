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

from models.rollout import rollout_for_epoch


if __name__ == '__main__':

    args = get_args()
    print(args)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(rank)

    logging.basicConfig(filename=os.path.join(args.log_dir, args.batch_weighted_ce, args.use_model+'_'+args.exp_name+'.txt'), level=logging.INFO)
    logging.info(args)



    # Model and dataloaders
    encoder =VisualEncoder(3, 0, attention_module=ScaledDotProductAttention)
    model = Transformer_visualgpt(encoder)


    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    gen_ce_loss_fn = nn.NLLLoss(ignore_index=model.tokenizer.pad_token_id)
    loss_fn = BatchWeightedCE(ignore_index = model.tokenizer.pad_token_id, args=args)

    train_split_by_val_scored_comment_id_pair_df = pd.read_csv(os.path.join(args.comments_root, 'train_split_by_val_scored_comment_id_pair.csv'))
    train_df = pd.read_csv(os.path.join(args.train_root, 'train.csv'))
    val_df = pd.read_csv(os.path.join(args.test_and_val_root, 'val.csv'))
    
    
    train_dataset = DPC2022(train_split_by_val_scored_comment_id_pair_df, model.tokenizer, args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    batch_trainDataLoader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=4, collate_fn=train_dataset.collate_fn, drop_last=True, sampler=train_sampler)


    train_dataset = DPC2022(train_split_by_val_scored_comment_id_pair_df, model.tokenizer, args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    batch_trainDataLoader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=4, collate_fn=train_dataset.collate_fn, drop_last=True, sampler=train_sampler)


    train_for_gen_dataset = DPC2022_for_generate(train_df, model.tokenizer, args)
    train_for_gen_sampler = torch.utils.data.distributed.DistributedSampler(train_for_gen_dataset)
    batch_train_for_genDataLoader = DataLoader(train_for_gen_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=4, collate_fn=train_for_gen_dataset.collate_fn, drop_last=True, sampler=train_for_gen_sampler)


    val_for_gen_dataset = DPC2022_for_generate(val_df, model.tokenizer, args)
    val_for_gen_sampler = torch.utils.data.distributed.DistributedSampler(val_for_gen_dataset)
    batch_val_for_gen_DataLoader = DataLoader(val_for_gen_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=4, collate_fn=val_for_gen_dataset.collate_fn, drop_last=True, sampler=val_for_gen_sampler)


    start_epoch = 0
    eval_result_dict_list = []

    if args.train_from_scratch == True:
        print('training from scratch')

    else:
        print('training from last')
        model_fname = '%s/%s/%s_image_caption_last_model.pth' % (args.models_dir, args.batch_weighted_ce ,args.use_model)

        if os.path.exists(model_fname):
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
            eval_result_dict_list = model_checkpoint['eval_result_dict_list']
            
            print(f"start_epoch:{start_epoch}, eval_result_dict_list:{eval_result_dict_list}")
                
            print('generator loadingg %s' %(model_fname))
        

        else:
            
            print('no exits %s, training from scratch' % (model_fname))

    # 等待加载完成
    dist.barrier()

    # 多gpu
    model = model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    training_loss = 0
    world_size = dist.get_world_size()
    for epoch in range(start_epoch, start_epoch+args.max_epoch):

        
        rollout_for_epoch(model, batch_train_for_genDataLoader, epoch, args)
        print('end')
                


