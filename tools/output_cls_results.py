from math import sqrt
import numpy as np
from sklearn.metrics import matthews_corrcoef
import pandas as pd
import os
"""
A script to calculate classification of other methods
"""

def cal_confusion_matrix(y_true,y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            FN += 1

    TP,FP,TN,FN = TP/len(y_true),FP/len(y_true),TN/len(y_true),FN/len(y_true)
    Acc = TP + TN
    Precision = TP / (TP+FP) if (TP+FP)!= 0 else 0
    Recall = TP / (TP + FN)
    Specificity = TN / (TN+FP) if (TN+FP) != 0 else 0
    MCC = matthews_corrcoef(y_true,y_pred)
    F1_score = 2*Precision*Recall / (Precision + Recall) if (Precision + Recall) != 0 else 0
    res = {'TP':TP,'FP':FP,'TN':TN,'FN':FN,'Acc':Acc,'Precision':Precision,'Recall':Recall,'Specificity':Specificity,'F1_score':F1_score,'MCC':MCC}
    return res

def print_res(gt_file,pred_file,gt_columns,labels_name):
    """
    Calculate results for files from other methods predict results
    labels_name:[]
    """
    gt_df = pd.read_csv(gt_file)
    if '.tsv' in pred_file:
        pred_df = pd.read_csv(pred_file,sep='\t')
        
    else:
        pred_df = pd.read_csv(pred_file)
    pred = pred_df[gt_columns].tolist()
    pred = np.array([1 if (i == labels_name[0]) else 0 for i in pred])
    gt = gt_df['Labels'].tolist()
    print(len(pred),len(gt))
    res = cal_confusion_matrix(gt, pred)
    print(res)

if __name__ == '__main__':
    gt_file = './datasets/ori_datasets/generalization_exp/test.csv'
    pred_file = './other_results/AMPlify_results/generalization_exp_model/results20230718205622.tsv'

    print_res(gt_file, pred_file, gt_columns='Prediction',labels_name=['AMP','non-AMP'])
    