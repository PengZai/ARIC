import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):

    def __init__(self,):
        super(GANLoss, self).__init__()


    def forward(self, log_prob, targets, rewards):
        

        batch_size, seq_len, vocab_size = log_prob.size()
        target_onehot = F.one_hot(targets, vocab_size).float()  # batch_size * seq_len * vocab_size
        pred = torch.sum(log_prob * target_onehot, dim=-1)  # batch_size * seq_len
        loss = -torch.sum(pred * rewards)

        return loss