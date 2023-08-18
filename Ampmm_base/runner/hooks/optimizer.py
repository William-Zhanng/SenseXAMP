import torch
from .base import Hook

class OptimizerHook(Hook):
    """
    A hook contains operations for the optimizer.
    Args:
        1. grad_clip (dict, optional): A config dict to control the clip_grad.
    Default funcitons:
        1. loss.backward()
        2. grad_clip()
        3. optimizer.step()
        4. scheduler.step()
    """
    def __init__(self, cfg):
        if hasattr(cfg, 'grad_clip'):
            self.grad_clip = cfg.grad_clip
        else:
            self.grad_clip = None

    def after_train_iter(self, runner):
        if runner.benchmark_name == 'amp_ranking':
            backward_tensor = runner.outputs['pred_scores']
            backward_input = runner.outputs['LambdaLoss']
            backward_tensor.backward(backward_input,retain_graph=True)
        else:
            runner.optimizer.zero_grad()
            runner.outputs['loss'].backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(runner.model.parameters(), **self.grad_clip)
        runner.optimizer.step()
        runner.scheduler.step()