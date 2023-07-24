import torch
from Ampmm_base.utils.ranking_utils import compute_lambda,get_pairs,scores_mapping

class LambdaLoss(object):

    def __call__(self, pred_mic, gt_mic):
        device = torch.device("cuda:0")
        pred_mic_numpy = pred_mic.data.cpu().numpy()
        gt_mic = list(gt_mic.data.cpu().numpy())
        # Map mic to labels
        true_labels = scores_mapping(gt_mic)
        # Compute lambda
        order_pairs = get_pairs(true_labels)
        lambdas = compute_lambda(true_labels, pred_mic_numpy, order_pairs)
        lambdas_torch = torch.Tensor(lambdas).to(device)
        return lambdas_torch