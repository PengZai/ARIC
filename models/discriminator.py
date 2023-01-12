import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self, vocab_size, emb_dim):
        super(Discriminator, self).__init__()

        num_classes = 2

        self.emb = nn.Embedding(vocab_size, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=2, dim_feedforward=768, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.fc = nn.Linear(emb_dim, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.init_parameters()

    def forward(self, x):
        """
        Args:
            x: (batch_size * seq_len)
        """
        emb = self.emb(x)  # batch_size * 1 * seq_len * emb_dim
        output_feature = self.transformer_encoder(emb)
        pooler_feature = output_feature[:, 0]

        fc_feature = self.fc(pooler_feature)
        pred = fc_feature

        log_prob = self.log_softmax(pred)

        return log_prob


    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)