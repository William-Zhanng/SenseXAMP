import os
import numpy as np
import torch
from .base import Hook
from Ampmm_base.utils.metrics import *

class AmpClsEvalHook(Hook):
    def __init__(self, cls_threshold=0.5) -> None:
        self.samples_num = 0
        self.cls_threshold = cls_threshold
        self.y_true = []
        self.y_pred = []

    def reset(self):
        self.samples_num = 0
        self.y_true = []
        self.y_pred = []

    def compare_results(self, cur_results, best_results):
        if cur_results['F1_score'] > best_results['F1_score']:
            return True
        return False

    def before_run(self, runner):
        self.reset()
        runner.best_val_results = {'F1_score':0}

    def before_val_epoch(self, runner):
        self.reset()
    
    def after_val_iter(self, runner):
        logits = runner.outputs['model_outputs']
        labels = runner.outputs['labels']
        preds = torch.sigmoid(logits)
        preds = preds >= self.cls_threshold
        self.samples_num += len(preds)
        self.y_true.extend(list(labels.cpu().numpy()))
        self.y_pred.extend(list(preds.cpu().numpy()))

    def after_val_epoch(self, runner):
        results = cal_confusion_matrix(self.y_true,self.y_pred)
        runner.cur_val_results = results 
        if self.compare_results(results, runner.best_val_results):
            runner.best_val_results = results
            if runner.local_rank == 0 and runner.mode == 'val':
                runner.save_checkpoint()
        
class AmpMultiLabelClsEvalHook(Hook):
    def __init__(self) -> None:
        self.samples_num = 0
        self.y_true = []
        self.y_pred = []

    def reset(self):
        self.samples_num = 0
        self.y_true = []
        self.y_pred = []

    def compare_results(self, cur_results, best_results):
        if cur_results['Acc'] > best_results['Acc']:
            return True
        return False  
    
    def before_run(self, runner):
        
        self.reset()
        runner.best_val_results = {'Acc':0}

    def before_val_epoch(self, runner):
        self.reset()
    
    def after_val_iter(self, runner):
        logits = runner.outputs['model_outputs']
        labels = runner.outputs['labels']
        preds = torch.sigmoid(logits)
        preds = preds >= 0.5
        self.samples_num += len(preds)
        self.y_true.extend(list(labels.cpu().numpy()))
        self.y_pred.extend(list(preds.cpu().numpy()))
    
    def after_val_epoch(self, runner):
        self.y_true = np.array(self.y_true)
        self.y_pred = np.array(self.y_pred)
        results = cal_multilabel_metrics(self.y_true,self.y_pred)
        runner.cur_val_results = results 
        if self.compare_results(results, runner.best_val_results):
            runner.best_val_results = results
            if runner.local_rank == 0 and runner.mode == 'val':
                runner.save_checkpoint()

class AmpRankingEvalHook(Hook):
    def __init__(self) -> None:
        self.sequences = []
        self.mic_list = []
        self.pred_scores = []
        self.samples_num = 0
    
    def reset(self):
        self.sequences = []
        self.mic_list = []
        self.pred_scores = []
        self.samples_num = 0
    
    def compare_results(self, cur_results, best_results):
        metrics_keys = ["top10_precision","top20_precision","top50_precision","top100_precision","top150_precision"]
        idx = len(metrics_keys) - 1
        while(idx >= 0):
            if cur_results[metrics_keys[idx]] == best_results[metrics_keys[idx]]:
                idx -= 1
                continue
            else:
                return cur_results[metrics_keys[idx]] > best_results[metrics_keys[idx]] 

        return False

    def before_run(self, runner):
        self.reset()
        runner.best_val_results = {"top10_precision":0,"top20_precision":0,"top50_precision":0,"top100_precision":0,"top150_precision":0}

    def before_val_epoch(self, runner):
        self.reset()
    
    def after_val_iter(self, runner):
        preds_score = runner.outputs['model_outputs']
        mic_gt = runner.outputs['mic']
        sequence = runner.outputs['seq']
        self.samples_num += len(preds_score)
        self.pred_scores.extend(list(preds_score.cpu().numpy()))
        self.mic_list.extend(list(mic_gt.cpu().numpy()))
        self.sequences.extend(sequence)

    def after_val_epoch(self, runner):
        results = cal_ranking_metrics(self.pred_scores,self.mic_list,self.sequences)
        runner.cur_val_results = results 
        if self.compare_results(results, runner.best_val_results):
            runner.best_val_results = results
            if runner.local_rank == 0 and runner.mode == 'val':
                runner.save_checkpoint()

class AmpRegEvalHook(Hook):
    def __init__(self) -> None:
        self.samples_num = 0
        self.y_true = []
        self.y_pred = []

    def reset(self):
        self.samples_num = 0
        self.y_true = []
        self.y_pred = []
    
    def calculate_total_score(self, result_dict):
        """
        Lower is better
        """
        score_dict = {"top10_mse":0.00,"top30_mse":0., \
                      "top50_mse":0.00,"top100_mse":0.0,'mse':0.3,"pos_mse":0.7}
        score = 0
        for k,v in score_dict.items():
            score += v*result_dict[k]
        return score

    def compare_results(self, cur_results, best_results):
        if self.calculate_total_score(cur_results) < self.calculate_total_score(best_results):
            return True
        return False

    def before_run(self, runner):
        self.reset()
        runner.best_val_results = {"top10_mse":1e7,"top30_mse":1e7, "top50_mse":1e7, \
                                   "top100_mse":1e7,'mse':1e7,"pos_mse":1e7, \
                                    "ndcg@10":0,"ndcg@30":0,"ndcg@50":0,"ndcg@100":0,"ndcg@150":0,"ndcg@all":0}

    def before_val_epoch(self, runner):
        self.reset()
    
    def after_val_iter(self, runner):
        preds = runner.outputs['model_outputs']
        labels = runner.outputs['labels']
        self.samples_num += len(preds)
        self.y_true.extend(list(labels.cpu().numpy()))
        self.y_pred.extend(list(preds.cpu().numpy()))

    def after_val_epoch(self, runner):
        reg_results = cal_reg_metrics(self.y_pred,self.y_true)
        ndcg_results = cal_ndcg_metrics(self.y_pred,self.y_true)
        results = {**reg_results,**ndcg_results}
        runner.cur_val_results = results 
        if self.compare_results(results, runner.best_val_results):
            runner.best_val_results = results
            if runner.local_rank == 0 and runner.mode == 'val':
                runner.save_checkpoint()