import torch

def build_optimizer(model, cfg):
    """
    Build a simple optimizer.
    """
    optimizer_cls = getattr(torch.optim,cfg['type'])
    return optimizer_cls(filter(lambda p: p.requires_grad, model.parameters()), \
                                lr=cfg['lr'],eps=cfg['eps'])
    