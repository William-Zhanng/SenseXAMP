import torch
from .samplers import DistributedWeightedSampler

def build_data_loader(dataset,batch_per_gpu,mode,weighted=False):
    if mode == 'train':
        if weighted:
            # sample_nums = int(len(dataset)*(batch_per_gpu/8))
            sample_nums = int(len(dataset))
            sampler = DistributedWeightedSampler(dataset,samples_pproc=sample_nums)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_per_gpu, \
                                                    num_workers=0, sampler=sampler)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_per_gpu, \
                                                    num_workers=0,shuffle=False)

    return dataloader