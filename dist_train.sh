# training visualgpt and generate caption each epoch using Diverse Aesthetic Caption Selector (DACS)
# the result we saved during evaltion will be used at experiment Image AQA based on Generated Captions
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 image_caption_demo.py