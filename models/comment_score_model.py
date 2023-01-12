import torch 
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
from transformers import BertModel, RobertaModel
import os
from transformers import BertTokenizer, RobertaTokenizer


class MLP_Head(nn.Module):
    def __init__(self, ):
        super(MLP_Head, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 2),
            nn.ReLU(),
        )
    
    def forward(self, feat):

        preds = self.head(feat)
        as_preds = preds[:, 0]
        cs_preds = preds[:, 1]

        return as_preds, cs_preds


class CommentScoreNet(nn.Module):
    def __init__(self, args):
        super(CommentScoreNet, self).__init__()

        self.args = args

        if args.comment_assessment_model == 'bert':
            self.text_base_model = Bert(args)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif args.comment_assessment_model == 'roberta':
            self.text_base_model = RoBERTa(args)
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        self.head = MLP_Head()

    def forward(self, input_ids, attention_mask):
    
        text_feat = self.text_base_model(input_ids, attention_mask)

        preds = self.head(text_feat)

        return preds




class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args

        
        self.backbone = BertModel.from_pretrained(os.path.join(args.huggingface_model_root, "bert-base-uncased"))
        self.activate = nn.LeakyReLU(0.2)

    def forward(self, input_ids, attention_mask):
    
        bert_output = self.backbone(input_ids, attention_mask)
        feat = bert_output.last_hidden_state[:, 0]

        feat = self.activate(feat)

        
        
        return feat


class RoBERTa(nn.Module):
    def __init__(self, args):
        super(RoBERTa, self).__init__()
        self.args = args
        self.backbone = RobertaModel.from_pretrained(os.path.join(args.huggingface_model_root, "roberta-base"))
        self.activate = nn.LeakyReLU(0.2)

    def forward(self, input_ids, attention_mask):
    
        roberta_output = self.backbone(input_ids, attention_mask)
        feat = roberta_output.last_hidden_state[:, 0]

        feat = self.activate(feat)

        
        
        return feat