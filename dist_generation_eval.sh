# this is implement experiment Image AQA based on Generated Captions
# we will save result every single epoch in workspace folder.
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 generation_eval.py