import os
import torch
# Task of the model 
# Including 'amp_cls', 'amp_multilabel_cls', 'amp_ranking', and 'amp_regression',default 'amp_cls'
benchmark_name = 'amp_cls'
dataset_name = 'cls_benchmark_balanced'
work_dir = 'experiments'
# Training hyper-params settings
epochs = 80
lr = 1e-3
eps = 1e-8
batch_interval = 100 # every n batch output loss info
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
data_root = './datasets/ori_datasets/cls_benchmark_balanced'
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
Resume = None # Resume from which ckpt to train
ckpt_path = None # Checkpoint for test

