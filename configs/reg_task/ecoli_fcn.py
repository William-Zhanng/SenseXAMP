import os
# Task of the model 
# Including 'amp_cls', 'amp_multilabel_cls', 'amp_ranking', and 'amp_regression',default 'amp_cls'
benchmark_name = 'amp_reg'
dataset_name = 'E.coli'
work_dir = 'experiments'
# Training hyper-params settings
epochs = 40
lr = 1e-6
eps = 1e-8
batch_interval = 50 # every n batch output loss info
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
        MSELoss=dict(
            weight=1.0, # weight of losses
            kwargs=dict(
                reduction='mean'
            )
        )
    )
)
data_root = './datasets/ori_datasets/regression_benchmark/E.coli' # path to your original dataset
data = dict(
    train=dict(
        datafile=os.path.join(data_root,'train.csv'),
        embeddings_fpath='./datasets/esm_embeddings/all/reg_E.coli_1:5.h5',   # path to your esm-1b embeddings
        stc_fpath='./datasets/stc_info/reg_E.coli_1:5.h5',     # path to your protein desriptors
        batch_per_gpu=128
    ),
    val=dict(
        datafile=os.path.join(data_root,'val.csv'),
        embeddings_fpath='./datasets/esm_embeddings/all/reg_E.coli_1:5.h5',
        stc_fpath='./datasets/stc_info/reg_E.coli_1:5.h5',
        batch_per_gpu=1
    ),
    test=dict(
        datafile=os.path.join(data_root,'test.csv'),
        embeddings_fpath='./datasets/esm_embeddings/all/reg_E.coli_1:5.h5',
        stc_fpath='./datasets/stc_info/reg_E.coli_1:5.h5',
        batch_per_gpu=64
    ),
)
# Resume & Checkpoint setting
Resume = None # Resume from which ckpt to train
ckpt_path = None # Checkpoint for test