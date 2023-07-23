import os
import torch
# Task of the model 
# Including 'amp_cls', 'amp_multilabel_cls', 'amp_ranking', and 'amp_regression',default 'amp_cls'
benchmark_name = 'amp_cls'
dataset_name = 'DeepAmPEP'
work_dir = 'experiments'
# Training hyper-params settings
epochs = 120
lr = 1e-6
eps = 1e-8
batch_interval = 10 # every n batch output loss info
optimizer = dict(type='Adam', lr=lr, eps=eps)
optimizer_config = dict(
    grad_clip=dict(max_norm=1.0)
    )
# Model settings
model = dict(
    name = 'BaseSlfAttnModel',
    kwargs = dict(
        emb_size=1280, 
        d_inner=2048, 
        n_slf_layers=2, 
        n_head=4, 
        d_k=1280, 
        d_v=1280, 
        dropout=0.2
    ),
    losses = dict(
        BCEWithLogitsLoss=dict(
            weight=1.0, # weight of losses
            kwargs=dict()
        )
    )
)
# Dataset settings
data_root = './datasets/ori_datasets/DeepAmPEP' # path to your original dataset
data = dict(
    train=dict(
        datafile=os.path.join(data_root,'train.csv'),
        embeddings_fpath='./datasets/esm_embeddings/all/DeepAmPEP.h5',  # path to your esm-1b embeddings
        stc_fpath='./datasets/stc_info/DeepAmPEP.h5',   # path to your protein desriptors
        batch_per_gpu=16
    ),
    val=dict(
        datafile=os.path.join(data_root,'val.csv'),
        embeddings_fpath='./datasets/esm_embeddings/all/DeepAmPEP.h5',
        stc_fpath='./datasets/stc_info/DeepAmPEP.h5',
        batch_per_gpu=1
    ),
    test=dict(
        datafile=os.path.join(data_root,'test.csv'),
        embeddings_fpath='./datasets/esm_embeddings/all/DeepAmPEP.h5',
        stc_fpath='./datasets/stc_info/DeepAmPEP.h5',
        batch_per_gpu=1
    ),
)
# Resume & Checkpoint setting
Resume = None # Resume from which ckpt to train
ckpt_path = None # Checkpoint for test

