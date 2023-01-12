import torch 
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
from .visual_model import VGG16, ResNet18, DenseNet121, ResNext50, Vit
from .text_model import Bert, RoBERTa, TextCNN, TextRNN
from transformers import BertTokenizer, RobertaTokenizer




class MLP_Head(nn.Module):
    def __init__(self, args):
        super(MLP_Head, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(1536, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.ReLU(),
        )
    
    def forward(self, feat):

        preds = self.head(feat)

        return preds




class MAANet(nn.Module):
    def __init__(self, args):
        super(MAANet, self).__init__()

        self.args = args

        if args.visual_model == 'vgg16':
            self.visual_base_model = VGG16(args)
        elif args.visual_model == 'resnet18':
            self.visual_base_model = ResNet18(args)
        elif args.visual_model == 'densenet121':
            self.visual_base_model = DenseNet121(args)
        elif args.visual_model == 'resnext50':
            self.visual_base_model = ResNext50(args)
        elif args.visual_model == 'vit':
            self.visual_base_model = Vit(args)

        if args.text_model == 'cnn':
            self.text_base_model = TextCNN(args)

        elif args.text_model == 'rnn':
            self.text_base_model = TextRNN(args)

        elif args.text_model == 'bert':
            self.text_base_model = Bert(args)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        elif args.text_model == 'roberta':
            self.text_base_model = RoBERTa(args)
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        if args.multimodal_model == 'mlp':
            self.head = MLP_Head(args)

    
    def forward(self, imgs, input_ids, attention_mask):

        
        visual_feat = self.visual_base_model(imgs)
        text_feat = self.text_base_model(input_ids, attention_mask)

        if self.args.multimodal_model == 'mlp':
            fusion_feat = torch.cat((visual_feat, text_feat), dim=-1)

        preds = self.head(fusion_feat)

        return preds


