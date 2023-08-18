import torch
import esm
from collections import OrderedDict

def parse_losses(losses):
    """
    Parse output losses from model.
    Args:
        losses: Dict(loss_name,loss_valve(torch.tensor, List[torch.tensor]))
    Return: 
        loss: sum of losses
        log_var: Dict(name, loss)
    """
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'Loss' in _key)
    for key,values in log_vars.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[key] = values.detach().cpu().numpy()
    log_vars['loss'] = loss.detach().cpu().numpy()
    return loss, log_vars

def default_batch_processor(model, data_batch, local_rank, train_mode):
    """
    Args:
        model: model
        data_batch: data batch from dataloader
        train_model: the model is training or eval
    Return: Dict{loss, log_vars, seq}
    """
    alphabet = esm.Alphabet.from_architecture("roberta_large")
    batch_converter = alphabet.get_batch_converter()
    prots = data_batch['seq']
    data = [(prots[i],prots[i]) for i in range(len(prots))]
    _, _, batch_tokens = batch_converter(data)
    if train_mode:
        model.train()
    else:
        model.eval()
    batch_tokens = batch_tokens.to(local_rank)
    label = data_batch['label'].to(local_rank)
    input_data = {'batch_tokens':batch_tokens, 'label':label}
    if 'mic' in data_batch:
        mic = data_batch['mic'].to(local_rank)
        input_data['mic'] = mic
    outputs = model(input_data)
    if train_mode:
        loss,log_vars = parse_losses(outputs)
        outputs = dict(loss=loss,log_vars=log_vars)
    outputs['seq'] = data_batch['seq']
    return outputs

def ranking_batch_processor(model, data_batch, local_rank, train_mode):
    """
    Args:
        model: model
        data_batch: data batch from dataloader
        train_model: the model is training or eval
    Return
    """
    alphabet = esm.Alphabet.from_architecture("roberta_large")
    batch_converter = alphabet.get_batch_converter()
    prots = data_batch['seq']
    data = [(prots[i],prots[i]) for i in range(len(prots))]
    _, _, batch_tokens = batch_converter(data)
    if train_mode:
        model.train()
    else:
        model.eval()
    batch_tokens = batch_tokens.to(local_rank)
    mic = data_batch['mic'].to(local_rank)
    label = data_batch['label']
    input_data = {'batch_tokens':batch_tokens, 'label':label, 'mic':mic}
    outputs = model(input_data)
    outputs['seq'] = data_batch['seq']
    return outputs

def base_batch_processor(model, data_batch, local_rank, train_mode):
    """
    Args:
        model: model
        data_batch: data batch from dataloader
        train_model: the model is training or eval
    Return: Dict{loss, log_vars, seq}
    """
    if train_mode:
        model.train()
    else:
        model.eval()
    batch_tokens = data_batch['emb'] # avg pooling of each token
    batch_tokens = batch_tokens.to(local_rank)
    label = data_batch['label'].to(local_rank)
    input_data = {'batch_tokens':batch_tokens, 'label':label}
    if 'stc' in data_batch:
        stc_info = data_batch['stc'].to(local_rank)
        input_data['stc'] = stc_info
    if 'mic' in data_batch:
        mic = data_batch['mic'].to(local_rank)
        input_data['mic'] = mic
    outputs = model(input_data)
    if train_mode:
        loss,log_vars = parse_losses(outputs)
        outputs = dict(loss=loss,log_vars=log_vars)
    outputs['seq'] = data_batch['seq']
    return outputs