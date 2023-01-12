import argparse
import os
import torch
import numpy as np

def get_args():

    parser = argparse.ArgumentParser(description="captionGPT")
    parser.add_argument('--exp_name', type=str, default='experiment')
    # parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument("--gpu_ids",type=str, default='0', help='0')
    parser.add_argument("--master_gpu_ids", type=int, default=0, help="master gpu using logging info and save model")
    parser.add_argument("--local_rank",type=int, default=0, help='local device id on current node')
    parser.add_argument('--log_dir',type = str, default="logs")
    parser.add_argument('--models_dir',type = str, default="saved_models")
    parser.add_argument('--use_model',type = str, default="visualgpt", help="visualgpt, lstm_cnn")
    parser.add_argument('--comment_assessment_model', type = str, default='bert', help='bert, roberta')
    parser.add_argument("--max_sentence_length", type= int, default = 64)
    parser.add_argument("--max_multi_sentence_length", type= int, default = 512)
    parser.add_argument("--rollout_num", type= int, default = 1)
    parser.add_argument("--max_epoch", type= int, default = 100)
    parser.add_argument("--gen_pretrain_epoch", type= int, default = 100)
    parser.add_argument("--train_from_scratch", action='store_true', default=True)
    parser.add_argument('--random_seed', type = int, default="42")
    parser.add_argument('--lr',type = float, default=1e-5)
    parser.add_argument('--imgSize', default=224, type=int)
    parser.add_argument('--num_hidden_layers', default=12, type=int)
    parser.add_argument('--similarity_threshold',type = float, default=0.7)
    parser.add_argument('--visual_model', type=str, default='vit', help = 'none, vgg16, resnet18, densenet121, resnext50, vit')
    parser.add_argument('--text_model', type=str, default='bert', help = 'none, cnn, rnn, bert, roberta')
    parser.add_argument('--multimodal_model', type=str, default='mlp', help = 'mlp')

    parser.add_argument('--batch_weighted_ce', default='cs_cscore', type=str, help='constant, length, cs_cscore, clip_cscore')
    parser.add_argument('--generation_train_comment_root',type = str, default="./generation_comment_for_train")
    parser.add_argument('--generation_eval_comment_root',type = str, default="./generation_comment_for_val")


    parser.add_argument('--huggingface_model_root', default='/raid/pengzai/model', type=str)
    parser.add_argument('--image_root', default='/raid/pengzai/database/DPC2022/img_256x256', type=str)
    parser.add_argument('--image_bottom_up_attention_feature_root', default='/raid/pengzai/database/DPC2022/img_bottom_up_feature_256x256', type=str)
    parser.add_argument('--data_root', default='/raid/pengzai/database/DPC2022/anotation/clean_and_all_score', type=str)
    parser.add_argument('--train_root', default='/raid/pengzai/database/DPC2022/anotation/train', type=str)
    parser.add_argument('--comments_root', default='/raid/pengzai/database/DPC2022/anotation/comments', type=str)
    parser.add_argument('--test_and_val_root', default='/raid/pengzai/database/DPC2022/anotation/test_and_val', type=str)

    # parser.add_argument('--huggingface_model_root', default='/media/pengzai/389215239214E762/huggingface_model', type=str)
    # parser.add_argument('--image_root', default='/media/pengzai/389215239214E762/database/DPC2022/img_256x256', type=str)
    # parser.add_argument('--image_bottom_up_attention_feature_root', default='/media/pengzai/389215239214E762/database/DPC2022/img_bottom_up_feature_256x256', type=str)
    # parser.add_argument('--data_root', default='/media/pengzai/389215239214E762/database/DPC2022/anotation/clean_and_all_score', type=str)
    # parser.add_argument('--train_root', default='/media/pengzai/389215239214E762/database/DPC2022/anotation/train', type=str)
    # parser.add_argument('--comments_root', default='/media/pengzai/389215239214E762/database/DPC2022/anotation/comments', type=str)
    # parser.add_argument('--test_and_val_root', default='/media/pengzai/389215239214E762/database/DPC2022/anotation/test_and_val', type=str)


    args = parser.parse_args()



    if not os.path.exists('%s' % (args.models_dir)):
        os.mkdir('%s' % (args.models_dir))

    if not os.path.exists(os.path.join(args.models_dir, args.batch_weighted_ce)):
        os.mkdir(os.path.join(args.models_dir, args.batch_weighted_ce))

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    if not os.path.exists(os.path.join(args.log_dir, args.batch_weighted_ce)):
        os.mkdir(os.path.join(args.log_dir, args.batch_weighted_ce))


    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["TOKENIZERS_PARALLELISM"] = "True"

    

    return args