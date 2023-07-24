from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal loss implementation.
    Args；
        alpha:(optional) Weighting factor in range (0,1) to balance
                positive vs negative examples
        num_classes: num of all classes
        gamma: focusing parameter, default = 2
    """
    def __init__(self, alpha: List, num_classes: int, gamma: int =2):
        super(FocalLoss,self).__init__()
        assert len(alpha) == num_classes
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma

    def forward(self, preds_logits, labels):
        self.alpha = self.alpha.to(preds_logits.device)
        p = torch.sigmoid(preds_logits)
        ce_loss = F.binary_cross_entropy_with_logits(preds_logits, labels, reduction="none")
        p_t = p * labels + (1 - p) * (1 - labels)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
        loss = alpha_t * loss
        loss = loss.mean()
        return loss

