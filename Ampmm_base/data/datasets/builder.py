from .base import AMPCls_Dataset,AMPMIC_Dataset,AMPMultiLabel_Dataset

def build_dataset(cfg,type):
    """
    Build datasets according to cfg
    type: train, test, or val
    """
    data_file = cfg.data[type].datafile
    embeddings_fpath = cfg.data[type].embeddings_fpath
    stc_fpath = cfg.data[type].stc_fpath
    if cfg.benchmark_name == 'amp_cls':
        dataset = AMPCls_Dataset(data_file,embeddings_fpath,stc_fpath)

    elif cfg.benchmark_name == 'amp_multilabel_cls':
        dataset = AMPMultiLabel_Dataset(data_file,embeddings_fpath,stc_fpath)

    elif cfg.benchmark_name == 'amp_ranking':
        dataset = AMPMIC_Dataset(data_file,embeddings_fpath,stc_fpath)

    elif cfg.benchmark_name == 'amp_reg':
        dataset = AMPMIC_Dataset(data_file,embeddings_fpath,stc_fpath)
    else:
        raise ValueError("There is no datasets of task {}".format(cfg.benchmark_name))

    return dataset