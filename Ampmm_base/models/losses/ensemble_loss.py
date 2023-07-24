from typing import List,Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Ensemble_BCELoss(nn.Module):
    """
    Cross entropy loss for ensemble model
    """
    def __init__(self, loss_weight:Dict, pos_weight: List[float]):
        super(Ensemble_BCELoss,self).__init__()
        self.loss_weight=loss_weight
        self.pos_weight=torch.tensor(pos_weight).cuda()

    def forward(self, pred_results, labels):
        losses = {}
        for name,weight in self.loss_weight.items():
            weighted_loss = weight*F.binary_cross_entropy_with_logits(pred_results[name], labels, pos_weight=self.pos_weight)
            losses[name] = weighted_loss
        sum_loss = sum([_value for _key,_value in losses.items()])
        return sum_loss

class Ensemble_MSELoss(nn.Module):
    """
    Cross entropy loss for ensemble model
    """
    def __init__(self, loss_weight:Dict):
        super(Ensemble_MSELoss,self).__init__()
        self.loss_weight=loss_weight
    
    def forward(self, pred_results, labels):
        losses = {}
        for name,weight in self.loss_weight.items():
            weighted_loss = weight*F.mse_loss(pred_results[name], labels)
            losses[name] = weighted_loss
        sum_loss = sum([_value for _key,_value in losses.items()])
        return sum_loss