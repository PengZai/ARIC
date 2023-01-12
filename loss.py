import torch
import torch.nn.functional as F
import torch.nn as nn


class BatchWeightedCE(nn.Module):
    def __init__(self, ignore_index, args):
        super(BatchWeightedCE, self).__init__()

        self.args = args
        # self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.ce = nn.NLLLoss(ignore_index=ignore_index,  reduction='none')

    def forward(self, preds, labels, cscores):

        
        batch_loss = self.ce(preds, labels)
        weighted_loss = cscores*batch_loss
        
        
        loss = torch.mean(weighted_loss)

        return loss
