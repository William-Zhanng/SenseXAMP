from math import sqrt
import numpy as np
from sklearn.metrics import matthews_corrcoef
from .ranking_utils import scores_mapping,ndcg

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

def cal_multilabel_metrics(pred,gt):
    """
    Return evaluation metrics of multilable classification
    """
    EMR = np.all(pred == gt, axis=1).mean() # Exact Match Ratio
    ACC = MultiLabelAccuracy(pred,gt)       # Accuracy
    PRE = MultiLabelPrecision(pred,gt)      # Precision
    REC = MultiLabelRecall(pred,gt)         # Recall
    HAM = Hamming_Loss(pred,gt)             # Hamming Loss  
    return {'EMR': EMR, 'ACC':ACC, 'PRE':PRE, 'REC':REC, 'HAM':HAM}

def cal_reg_metrics(pred,gt):
    """
    Return TopK MSE
    """
    # Sort by gt
    MAX_MIC = np.log10(8196)
    results =  [gt,pred]
    results = sorted(list(map(list, zip(*results))))
    results = list(map(list, zip(*results)))
    gt,pred = results[0],results[1]
    top10_mse = np.mean([(actual - predicted) ** 2 for actual, predicted in zip(gt[0:10], pred[0:10])])
    top30_mse = np.mean([(actual - predicted) ** 2 for actual, predicted in zip(gt[0:30], pred[0:30])])
    top50_mse = np.mean([(actual - predicted) ** 2 for actual, predicted in zip(gt[0:50], pred[0:50])])
    top100_mse = np.mean([(actual - predicted) ** 2 for actual, predicted in zip(gt[0:100], pred[0:100])])
    mse = np.mean([(actual - predicted) ** 2 for actual, predicted in zip(gt, pred)])
    pos_mse = np.mean([(actual - predicted) ** 2 
                                for actual, predicted in zip(gt, pred) 
                                if actual < MAX_MIC - 0.01])
    res_dict = {'top10_mse':top10_mse,'top30_mse':top30_mse,'top50_mse':top50_mse,'top100_mse':top100_mse, \
                'mse':mse,'pos_mse':pos_mse}
    return res_dict

def cal_ndcg_metrics(pred,gt):
    """
    Calculate ndcg metrics for all AMPs(exclude Non-AMPs)
    """
    MAX_MIC = 3.913
    # sorted by pred_mic
    results =  [pred,gt]
    results = sorted(list(map(list, zip(*results))))
    results = list(map(list, zip(*results)))
    pred, gt = np.array(results[0]),np.array(results[1])
    # exclude Non-AMPs
    amp_index = gt<MAX_MIC
    gt = gt[amp_index]
    pred = pred[amp_index]
    rel_scores = cal_rel_scores(gt)

    ndcg_at_10 = ndcg_at_k(rel_scores,10)
    ndcg_at_30 = ndcg_at_k(rel_scores,30)
    ndcg_at_50 = ndcg_at_k(rel_scores,50)
    ndcg_at_100 = ndcg_at_k(rel_scores,100)
    ndcg_at_150 = ndcg_at_k(rel_scores,150)
    ndcg_at_all = ndcg_at_k(rel_scores,len(rel_scores))
    res_dict = {'ndcg@10':ndcg_at_10,'ndcg@30':ndcg_at_30,'ndcg@50':ndcg_at_50,'ndcg@100':ndcg_at_100, \
            'ndcg@150':ndcg_at_150,'ndcg@all':ndcg_at_all}
    return res_dict

def cal_ranking_metrics(pred_scores,mic_list,sequences_list):
    """
    Args:
        pred: prediction scores of peptides, higher is better
        mic_list: MIC of peptides, lower is better
        sequence_list: sequence of peptides
    """
    pred_sort_index = np.argsort(pred_scores)[::-1] # scores higher to lower，mic lower to higher
    gt_sort_index = np.argsort(mic_list)            # mic lower to higher
    pred_peptides = list(np.array(sequences_list)[pred_sort_index]) # prediction topk sequences
    gt_peptides = list(np.array(sequences_list)[gt_sort_index])     # gt topk sequences
    top10_precision = topK_precision(pred_peptides,gt_peptides,10)
    top20_precision = topK_precision(pred_peptides,gt_peptides,20)
    top50_precision = topK_precision(pred_peptides,gt_peptides,50)
    top100_precision = topK_precision(pred_peptides,gt_peptides,100)
    top150_precision = topK_precision(pred_peptides,gt_peptides,150)
    metrics = {"top10_precision":top10_precision,"top20_precision":top20_precision,"top50_precision":top50_precision,"top100_precision":top100_precision,
               "top150_precision":top150_precision}

    return metrics

def MultiLabelAccuracy(y_pred,y_true):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]

def MultiLabelPrecision(y_pred,y_true):
    temp = 0
    for i in range(y_true.shape[0]):
        if sum(y_true[i]) == 0:
            continue
        temp+= sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_true[i])
    return temp/ y_true.shape[0]

def MultiLabelRecall(y_pred,y_true):
    temp = 0
    for i in range(y_true.shape[0]):
        if sum(y_pred[i]) == 0:
            continue
        temp+= sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_pred[i])
    return temp/ y_true.shape[0]

def Hamming_Loss(y_pred, y_true):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += (np.size(y_true[i])-np.count_nonzero(y_true[i] == y_pred[i]))
    return temp/(y_true.shape[0] * y_true.shape[1])

def topK_precision(pred,gt,k):
    pred_k = pred[:k]
    gt_k = gt[:k]
    correct = 0
    for i in pred_k:
        for j in gt_k:
            if i == j:
                correct += 1
                continue
    return correct / k

def cal_rel_scores(mic_gt):
    """
    Calculate relevance_scores for each AMP, according to their groudtruth MIC label
    """
    mic_gt = np.array(mic_gt)
    scale = 2
    max_mic = 3.913
    return np.exp((-scale)*(mic_gt - max_mic))

def ndcg_at_k(relevance_scores,k):
    """
    Calculate ndcg@k.
    Args:
        relevance_scores: relevance socres of model output. Calculate by gt of MIC label.
        k: k of ndcg@k
    """
    relevance_scores = np.asarray(relevance_scores)
    k = min(k, len(relevance_scores))
    
    # DCG@k
    dcg = np.sum(relevance_scores[:k] / np.log2(np.arange(2, k + 2)))
    
    # IDCG@k
    sorted_scores = np.sort(relevance_scores)[::-1]
    idcg = np.sum(sorted_scores[:k] / np.log2(np.arange(2, k + 2)))
    
    # NDCG@k
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return ndcg
