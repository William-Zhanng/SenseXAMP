import os
import numpy as np
import torch
import pandas as pd

from .base import Hook
from Ampmm_base.utils.metrics import *

class AmpClsAnalyseHook(Hook):
    def __init__(self, work_dir, cls_threshold=0.5) -> None:
        self.work_dir = work_dir
        self.samples_num = 0
        self.cls_threshold = cls_threshold
        self.sequence = []
        self.y_true = []
        self.y_pred = []
        self.pred_logits = []

    def before_val_epoch(self, runner):
        # Load checkpoint
        ckpt_path = runner.cfg.ckpt_path
        if ckpt_path is not None and runner.local_rank == 0:
            runner.load_checkpoint(ckpt_path)
            runner.logger.info("Start test, load checkpoint from {}".format(ckpt_path))

    def after_val_iter(self, runner):
        sequence = runner.outputs['seq']
        logits = runner.outputs['model_outputs']
        labels = runner.outputs['labels']
        preds_logits = torch.sigmoid(logits)
        preds = preds_logits >= self.cls_threshold
        self.samples_num += len(preds)
        self.y_true.extend(list(labels.cpu().numpy()))
        self.y_pred.extend(list(preds.cpu().numpy()))
        self.sequence.extend(list(sequence))
        self.pred_logits.extend(list(preds_logits.cpu().numpy()))

    def save_results(self):
        result_df = pd.DataFrame({'Sequence':self.sequence,'Labels':self.y_true,'pred_label':self.y_pred,\
                                  'AMP_probability':self.pred_logits})
        result_df.to_csv(os.path.join(self.work_dir,'prediction.csv'),index=False)

    def after_val_epoch(self, runner):
        results = cal_confusion_matrix(self.y_true,self.y_pred)
        # for logger output logs
        runner.cur_val_results = results        
        runner.best_val_results = results
        # output test results
        self.save_results()

class AmpMultiLabelClsAnalyseHook(Hook):
    def __init__(self, work_dir) -> None:
        self.work_dir = work_dir
        self.samples_num = 0
        self.sequence = []
        self.y_true = []
        self.y_pred = []
    
    def before_val_epoch(self, runner):
        # Load checkpoint
        ckpt_path = runner.cfg.ckpt_path
        if ckpt_path is not None and runner.local_rank == 0:
            runner.load_checkpoint(ckpt_path)
            runner.logger.info("Start test, load checkpoint from {}".format(ckpt_path))

    def after_val_iter(self, runner):
        sequence = runner.outputs['seq']
        logits = runner.outputs['model_outputs']
        labels = runner.outputs['labels']
        preds = torch.sigmoid(logits)
        preds = preds >= 0.5
        self.samples_num += len(preds)
        self.y_true.extend(list(labels.cpu().numpy()))
        self.y_pred.extend(list(preds.cpu().numpy()))

    def save_results(self):
        result_df = pd.DataFrame({'Sequence':self.sequence,'label':self.y_true,'Prediction':self.y_pred})
        result_df.to_csv(os.path.join(self.work_dir,'prediction.csv'),index=False)

    def after_val_epoch(self, runner):
        self.y_true = np.array(self.y_true)
        self.y_pred = np.array(self.y_pred)
        results = cal_multilabel_metrics(self.y_pred,self.y_true)
        # for logger output logs
        runner.cur_val_results = results        
        runner.best_val_results = results
        # output test results
        # self.save_results()

class AmpRankingAnalyseHook(Hook):
    def __init__(self, work_dir) -> None:
        self.work_dir = work_dir
        self.sequences = []
        self.mic_list = []
        self.pred_scores = []
        self.samples_num = 0

    def before_val_epoch(self, runner):
        # Load checkpoint
        ckpt_path = runner.cfg.ckpt_path
        if ckpt_path is not None and runner.local_rank == 0:
            runner.load_checkpoint(ckpt_path)
            runner.logger.info("Start test, load checkpoint from {}".format(ckpt_path))
        
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
        # for logger output logs
        runner.cur_val_results = results        
        runner.best_val_results = results
        # output test results
        self.save_results()

    def save_results(self):
        result_df = pd.DataFrame({'sequence':self.sequences,'MIC':self.mic_list,'pred_scores':self.pred_scores})
        result_df.sort_values("pred_scores", ascending=False, inplace=True)
        result_df.to_csv(os.path.join(self.work_dir,'prediction.csv'),index=False)

class AmpRegAnalyseHook(Hook):
    def __init__(self, work_dir) -> None:
        self.work_dir = work_dir
        self.samples_num = 0
        self.sequence = []
        self.y_true = []
        self.y_pred = []
        
    def before_val_epoch(self, runner):
        # Load checkpoint
        ckpt_path = runner.cfg.ckpt_path
        if ckpt_path is not None and runner.local_rank == 0:
            runner.load_checkpoint(ckpt_path)
            runner.logger.info("Start test, load checkpoint from {}".format(ckpt_path))

    def after_val_iter(self, runner):
        sequence = runner.outputs['seq']
        preds = runner.outputs['model_outputs']
        labels = runner.outputs['labels']
        self.samples_num += len(preds)
        self.y_true.extend(list(labels.cpu().numpy()))
        self.y_pred.extend(list(preds.cpu().numpy()))
        self.sequence.extend(list(sequence))

    def save_results(self):
        result_df = pd.DataFrame({'Sequence':self.sequence,'MIC_gt':self.y_true,'MIC_pred':self.y_pred})
        result_df.sort_values("MIC_gt", inplace=True)
        result_df.to_csv(os.path.join(self.work_dir,'prediction.csv'),index=False)

    def after_val_epoch(self, runner):
        reg_results = cal_reg_metrics(self.y_pred,self.y_true)
        ndcg_results = cal_ndcg_metrics(self.y_pred,self.y_true)
        results = {**reg_results,**ndcg_results}
        # for logger output logs
        runner.cur_val_results = results        
        runner.best_val_results = results
        # output test results
        self.save_results()