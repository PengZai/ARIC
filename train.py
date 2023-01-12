from tqdm import tqdm
import torch.distributed as dist


def train_for_epoch(model, dataloader, loss_fn, optimizer, epoch, args):
    
    training_loss = 0
    dataset = dataloader.dataset
    text_tokenizer = dataset.text_tokenizer

    model.train()
    
    dataloader.sampler.set_epoch(epoch)
    desc_template = 'Epoch %d - train ' %(epoch)

    rank = dist.get_rank()
    if rank == 0:
        pbar = tqdm(desc=desc_template, unit='it', total=len(dataloader))

    for i, batch_data_dict in enumerate(dataloader):
        
  
        imgfeats = batch_data_dict['imgfeats'].cuda()
        input_ids = batch_data_dict['text_encoding']['input_ids'].cuda()
        attention_mask = batch_data_dict['text_encoding']['attention_mask'].cuda()
        cscores = batch_data_dict['cscores'].cuda()
        IDs = batch_data_dict['IDs']
        
        log_prob, past_key_values = model(imgfeats, input_ids)

        gt_ids = input_ids[:, 1:].contiguous()
        log_prob = log_prob[:, :-1].contiguous()

        loss = loss_fn(log_prob.view(-1, text_tokenizer.vocab_size), gt_ids.view(-1), cscores.view(-1))
        

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)
        loss = loss/ dist.get_world_size()
        training_loss+=loss.item()

        rank = dist.get_rank()    
        if rank == 0:
            pbar.set_description(desc_template + 'local_rank=%d, rank=%d, loss: %.4f' %(args.local_rank, rank, loss.item()))
            pbar.update()
        
        if i > 2:
            break

    return training_loss/len(dataloader)