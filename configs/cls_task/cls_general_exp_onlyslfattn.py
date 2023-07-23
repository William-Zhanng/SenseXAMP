import os
import torch
# Task of the model 
# Including 'amp_cls', 'amp_multilabel_cls', 'amp_ranking', and 'amp_regression',default 'amp_cls'
benchmark_name = 'amp_cls'
dataset_name = 'generalization_exp'
work_dir = 'experiments'
# Training hyper-params settings
epochs = 80
lr = 1e-5
eps = 1e-8
batch_interval = 100 # every n batch output loss info
optimizer = dict(type='Adam', lr=lr, eps=eps)
optimizer_config = dict(
    grad_clip=dict(max_norm=1.0)
    )
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
            kwargs=dict(
            )
        )
    )
)
cls_threshold = 0.50  # modify this parameter if using threshold moving
use_weighted_sampler=False # modify if using resampling to rebalance datasets
# Dataset settings
data_root = './datasets/ori_datasets/generalization_exp'
data = dict(
    train=dict(
        datafile=os.path.join(data_root,'train.csv'),
        embeddings_fpath='./datasets/esm_embeddings/all/cls_benchmark.h5',
        stc_fpath='./datasets/stc_info/cls_benchmark.h5',
        batch_per_gpu=64
    ),
    val=dict(
        datafile=os.path.join(data_root,'val.csv'),
        embeddings_fpath='./datasets/esm_embeddings/all/cls_benchmark.h5',
        stc_fpath='./datasets/stc_info/cls_benchmark.h5',
        batch_per_gpu=64
    ),
    test=dict(
        datafile=os.path.join(data_root,'test.csv'),
        embeddings_fpath='./datasets/esm_embeddings/all/cls_benchmark.h5',
        stc_fpath='./datasets/stc_info/cls_benchmark.h5',
        batch_per_gpu=64
    ),
)
# Resume & Checkpoint setting
# Resume from which ckpt to train
Resume = None
# Checkpoint for test
ckpt_path = None



