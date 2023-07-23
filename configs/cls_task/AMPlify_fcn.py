import os
import torch
# Task of the model 
# Including 'amp_cls', 'amp_multilabel_cls', 'amp_ranking', and 'amp_regression',default 'amp_cls'
benchmark_name = 'amp_cls'
dataset_name = 'AMPlify'
work_dir = 'experiments'
# Training hyper-params settings
epochs = 80
lr = 1e-3
eps = 1e-8
batch_interval = 20 # every n batch output loss info
optimizer = dict(type='Adam', lr=lr, eps=eps)
optimizer_config = dict(
    grad_clip=dict(max_norm=1.0)
    )
model = dict(
    name = 'BaseStcClsModel',
    kwargs = dict(
        stc_size=676, 
        emb_size=1280, 
        n_classes=1
    ),
    losses = dict(
        BCEWithLogitsLoss=dict(
            weight=1.0, # weight of losses
            kwargs=dict(
                pos_weight=torch.tensor([1.0]).cuda()
            )
        )
    )
)
# Dataset settings
data_root = './datasets/ori_datasets/AMPlify'  # path to your original dataset
data = dict(
    train=dict(
        datafile=os.path.join(data_root,'train.csv'),
        embeddings_fpath='./datasets/esm_embeddings/all/AMPlify.h5',  # path to your esm-1b embeddings
        stc_fpath='./datasets/stc_info/AMPlify.h5',   # path to your protein desriptors
        batch_per_gpu=64
    ),
    val=dict(
        datafile=os.path.join(data_root,'val.csv'),
        embeddings_fpath='./datasets/esm_embeddings/all/AMPlify.h5',
        stc_fpath='./datasets/stc_info/AMPlify.h5',
        batch_per_gpu=1
    ),
    test=dict(
        datafile=os.path.join(data_root,'test.csv'),
        embeddings_fpath='./datasets/esm_embeddings/all/AMPlify.h5',
        stc_fpath='./datasets/stc_info/AMPlify.h5',
        batch_per_gpu=1
    ),
)
# Resume & Checkpoint setting
Resume = None # Resume from which ckpt to train
ckpt_path = None # Checkpoint for test

