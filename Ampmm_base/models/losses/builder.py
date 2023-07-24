import imp
from unicodedata import name
import torch
from .lambdaloss import LambdaLoss
from .focalloss import FocalLoss
from .ensemble_loss import Ensemble_BCELoss,Ensemble_MSELoss

def build_loss_fn(loss_name, loss_cfg):
    if loss_name == 'LambdaLoss':
        return LambdaLoss()
    elif loss_name == 'FocalLoss':
        return FocalLoss(**loss_cfg.kwargs)
    elif loss_name == 'Ensemble_BCELoss':
        return Ensemble_BCELoss(**loss_cfg.kwargs)
    elif loss_name == 'Ensemble_MSELoss':
        return Ensemble_MSELoss(**loss_cfg.kwargs)
    else:
        loss_fn = getattr(torch.nn,loss_name)
        return loss_fn(**loss_cfg.kwargs)

class CombinedLossEvaluator(object):
    """
    Combined multiple loss evaluator
    """
    def __init__(self, loss_evaluators, loss_weights):

        self.loss_evaluators = loss_evaluators
        self.loss_weights = loss_weights
        
    def __call__(self, pred_results, gt, **kwargs):
        comb_loss_dict = {}
        for loss_name, loss_evaluator in self.loss_evaluators.items():
            loss = loss_evaluator(pred_results,gt)
            weight = self.loss_weights[loss_name]
            if isinstance(loss,dict):
                loss = {k:v*weight for k,v in loss.items()}
            else:
                comb_loss_dict[loss_name] = loss*weight
        return comb_loss_dict

def build_loss_evaluator(cfg):
    loss_evaluators = dict()
    loss_weights = dict()
    loss_dict = cfg.model.losses.copy()
    for loss_name,loss_cfg in loss_dict.items():
        loss_evaluator = build_loss_fn(loss_name,loss_cfg)
        loss_evaluators[loss_name] = loss_evaluator
        loss_weights[loss_name] = loss_cfg.weight
    return CombinedLossEvaluator(loss_evaluators,loss_weights)
