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
    name = 'MultiModalFusionModel',
    kwargs = dict(
        stc_size=676, 
        emb_size=1280, 
        d_inner=2048, 
        n_slf_layers=2, 
        n_cross_layers=1, 
        n_head=4, 
        d_k=1280, 
        d_v=1280, 
        dropout=0.2
    ),
    losses = dict(
        Ensemble_BCELoss=dict(
            weight=1.0,
            kwargs=dict(
                pos_weight=[1.0],
                loss_weight=dict(
                    stc_pred=0.25,
                    slf_pred=0.25,
                    final_pred=0.5,
                )
            )
        )
    )
)
# Dataset settings
data_root = './datasets/ori_datasets/generalization_exp'
data = dict(
    train=dict(
        datafile=os.path.join(data_root,'train.csv'),
        embeddings_fpath='./datasets/esm_embeddings/all/cls_benchmark.h5',
        stc_fpath='./datasets/stc_info/cls_benchmark.h5',
        batch_per_gpu=32
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


