import torch 
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
from transformers import BertModel, RobertaModel
import os

class MLP_Head(nn.Module):
    def __init__(self, ):
        super(MLP_Head, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.ReLU(),
        )
    
    def forward(self, feat):

        preds = self.head(feat)

        return preds




class TAANet(nn.Module):
    def __init__(self, args):
        super(TAANet, self).__init__()

        self.args = args

        if args.text_model == 'cnn':
            self.text_base_model = TextCNN(args)

        elif args.text_model == 'rnn':
            self.text_base_model = TextRNN(args)

        elif args.text_model == 'bert':
            self.text_base_model = Bert(args)

        elif args.text_model == 'roberta':
            self.text_base_model = RoBERTa(args)


        self.head = MLP_Head()

    def forward(self, input_ids, attention_mask):
    
        text_feat = self.text_base_model(input_ids, attention_mask)

        preds = self.head(text_feat)

        return preds


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        
        self.args = args
        Ci = 1
        Co = 100
        
        token_groups = [3,4,5,6,7]

        self.embed = nn.Embedding(
            num_embeddings = args.vocab_size,
            embedding_dim = args.embed_dim,
            padding_idx=args.pad_idx,
            )


        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (group, args.embed_dim)) for group in token_groups])
        self.fc = nn.Sequential(
            nn.Linear(len(token_groups) * Co, args.embed_dim),
            nn.Tanh(),
            # nn.LeakyReLU(0.2),
        )
        

        
    def forward(self, input_ids, attention_mask):
    
        x = self.embed(input_ids,)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        feat = self.fc(torch.cat(x, 1))

        
        return feat



class TextRNN(nn.Module):
    def __init__(self, args):
        super(TextRNN, self).__init__()
        
        self.args = args

        self.embed = nn.Embedding(
            num_embeddings = args.vocab_size,
            embedding_dim = args.embed_dim,
            padding_idx=args.pad_idx,
            )

        self.gru = nn.GRU(
            input_size=args.embed_dim, 
            hidden_size=args.embed_dim, 
            num_layers=2, 
            bidirectional=True,
            batch_first=True,
            )

        self.fc = nn.Sequential(
            nn.Linear(args.embed_dim*2 * 2, args.embed_dim),
            nn.Tanh(),
        )



        
    def forward(self, input_ids, attention_mask):
    
        x = self.embed(input_ids,)
        
        feat, hn = self.gru(x)
        
        last_output = feat[:,-1]
        first_output = feat[:,0]

        feat = self.fc(torch.cat([first_output , last_output], dim=-1))

        return feat


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

        if self.args.exp_attribute == 'maa' and self.args.multimodal_model == 'transformer':
            return roberta_output.last_hidden_state

        return feat