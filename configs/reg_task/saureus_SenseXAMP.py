import os

# Task of the model 
# Including 'amp_cls', 'amp_multilabel_cls', 'amp_ranking', and 'amp_regression',default 'amp_cls'
benchmark_name = 'amp_reg'
dataset_name = 'S.aureus'
work_dir = 'experiments'
# Training hyper-params settings
epochs = 120
lr = 1e-4
eps = 1e-8
batch_interval = 50 # every n batch output loss info
optimizer = dict(type='Adam', lr=lr, eps=eps)
optimizer_config = dict(
    # grad_clip=dict(max_norm=1.0)
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
        Ensemble_MSELoss=dict(
            weight=1.0,
            kwargs=dict(
                loss_weight=dict(
                    stc_pred=0.25,
                    slf_pred=0.25,
                    final_pred=0.5,
                )
            )
        )
    )
)
data_root = './datasets/ori_datasets/regression_benchmark/S.aureus' # path to your original dataset
data = dict(
    train=dict(
        datafile=os.path.join(data_root,'train.csv'),
        embeddings_fpath='./datasets/esm_embeddings/all/reg_S.aureus_1:5.h5', # path to your esm-1b embeddings
        stc_fpath='./datasets/stc_info/reg_S.aureus_1:5.h5', # path to your protein desriptors
        batch_per_gpu=128
    ),
    val=dict(
        datafile=os.path.join(data_root,'val.csv'),
        embeddings_fpath='./datasets/esm_embeddings/all/reg_S.aureus_1:5.h5',
        stc_fpath='./datasets/stc_info/reg_S.aureus_1:5.h5',
        batch_per_gpu=1
    ),
    test=dict(
        datafile=os.path.join(data_root,'test.csv'),
        embeddings_fpath='./datasets/esm_embeddings/all/reg_S.aureus_1:5.h5',
        stc_fpath='./datasets/stc_info/reg_S.aureus_1:5.h5',
        batch_per_gpu=16
    ),
)
# Resume & Checkpoint setting
Resume = None # Resume from which ckpt to train
ckpt_path = None # Checkpoint for test