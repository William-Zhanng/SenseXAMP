import esm
import torch
import torch.nn as nn
from .losses import build_loss_evaluator

class EsmBaseClsModel(nn.Module):
    """
    Baseline Amp classfication model base on esm finetuning
    """
    def __init__(self, cfg, n_embedding=1280, n_classes=1):
        super(EsmBaseClsModel,self).__init__()
        self.ProteinBert, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.classifier = nn.Sequential(
            nn.Linear(n_embedding, 768),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64,n_classes)
        )
        self.loss_evaluator = build_loss_evaluator(cfg)

    def forward(self, input_data):
        data = input_data['batch_tokens']
        results = self.ProteinBert(data, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        cls_embedding = token_representations[:,0,:]  # cls token
        output_logits = self.classifier(cls_embedding)
        output_logits = output_logits.squeeze(dim=-1)
        if self.training:
            loss_dict = self.loss_evaluator(output_logits,input_data['label'])
            return loss_dict
        else:
            results = dict(
                model_outputs = output_logits,
                labels = input_data['label']
            )
            return results

class EsmBaseMultiLabelClsModel(nn.Module):
    """
    Baseline Amp classfication model base on esm finetuning
    """
    def __init__(self, cfg, n_embedding=1280, n_classes=7):
        super(EsmBaseMultiLabelClsModel,self).__init__()
        self.ProteinBert, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.classifier = nn.Sequential(
            nn.Linear(n_embedding, 768),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64,n_classes)
        )
        self.loss_evaluator = build_loss_evaluator(cfg)

    def forward(self, input_data):
        data = input_data['batch_tokens']
        results = self.ProteinBert(data, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        cls_embedding = token_representations[:,0,:]  # cls token
        output_logits = self.classifier(cls_embedding)
        output_logits = output_logits.squeeze(dim=-1)
        if self.training:
            loss_dict = self.loss_evaluator(output_logits,input_data['label'])
            return loss_dict
        else:
            results = dict(
                model_outputs = output_logits,
                labels = input_data['label']
            )
            return results

class EsmBaseRegModel(nn.Module):
    """
    Baseline Amp regression model base on esm finetuning
    """
    def __init__(self,cfg, n_embedding=1280):
        super(EsmBaseRegModel,self).__init__()
        self.ProteinBert, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.Predictor = nn.Sequential(
            nn.Linear(n_embedding, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
        self.loss_evaluator = build_loss_evaluator(cfg)

    def forward(self, input_data):
        data = input_data['batch_tokens']
        results = self.ProteinBert(data, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        cls_embedding = token_representations[:,0,:]  # cls token
        pred_result = self.Predictor(cls_embedding)
        pred_result = pred_result.squeeze(dim=-1)
        if self.training:
            loss_dict = self.loss_evaluator(pred_result,input_data['label'])
            return loss_dict
        else:
            result = dict(
                model_outputs = pred_result,
                labels = input_data['label']
            )
            return result

class EsmBaseRankingModel(nn.Module):
    """
    Baseline Amp ranking model base on lambdarank + esm finetuneing
    """
    def __init__(self, cfg, n_embedding=1280):
        super(EsmBaseRankingModel,self).__init__()
        self.ProteinBert, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.Predictor = nn.Sequential(
            nn.Linear(n_embedding, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
        self.loss_evaluator = build_loss_evaluator(cfg)

    def forward(self, input_data):
        data = input_data['batch_tokens']
        results = self.ProteinBert(data, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        cls_embedding = token_representations[:,0,:]  # cls token
        pred_result = self.Predictor(cls_embedding)
        pred_result = pred_result.squeeze(dim=-1)
        if self.training:
            outputs = self.loss_evaluator(pred_result,input_data['mic'])
            outputs['pred_scores'] = pred_result
            return outputs
        else:
            pass
            result = dict(
                model_outputs = pred_result,
                mic = input_data['mic']
            )
            return result

META_MODELS = {'EsmBaseClsModel':EsmBaseClsModel,'EsmBaseMultiLabelClsModel':EsmBaseMultiLabelClsModel,
               'EsmBaseRegModel':EsmBaseRegModel,'EsmBaseRankingModel':EsmBaseRankingModel}