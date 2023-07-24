import esm
import torch
import torch.nn as nn
from .losses import build_loss_evaluator

class BaseStcClsModel(nn.Module):
    """
    Base cls model using stc_info
    """
    def __init__(self, cfg, stc_size=676, emb_size=1280, n_classes=1):
        super(BaseStcClsModel,self).__init__()
        self.emb_layer = nn.Sequential(
            nn.Linear(stc_size, emb_size),
            nn.ReLU())

        self.cls_head = nn.Sequential(
            
            nn.Linear(emb_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,n_classes)
        )

        if cfg.benchmark_name == 'amp_cls':
            self.mode = 'cls'
        elif cfg.benchmark_name == 'amp_reg':
            self.mode = 'reg'

        self.loss_evaluator = build_loss_evaluator(cfg)

    def forward(self, input_data):
        data = input_data['stc']
        emb = self.emb_layer(data)
        pred_results = self.cls_head(emb)
        pred_results = pred_results.squeeze(dim=-1)
        gt = input_data['label'] if self.mode == 'cls' else input_data['mic']
        if self.training:
            loss_dict = self.loss_evaluator(pred_results,gt)
            return loss_dict
        else:
            
            results = dict(
                model_outputs = pred_results,
                labels = gt
            )
            return results

META_MODELS = {'BaseStcClsModel':BaseStcClsModel}