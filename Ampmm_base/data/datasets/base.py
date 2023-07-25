import numpy as np
import pandas as pd
import h5py
import torch
import torch.utils.data as data
import os

class AMPCls_Dataset(data.Dataset):
    """
    Datasets for Amps binary
    """
    def __init__(self,data_file, embeddings_fpath=None, stc_fpath=None):
        self.data_df = pd.read_csv(data_file)
        self.all_embeddings = h5py.File(embeddings_fpath,'r') if embeddings_fpath else None
        self.stc_info = h5py.File(stc_fpath,'r') if stc_fpath else None
        self.labels = torch.tensor(self.data_df['Labels'].tolist()) # for Distributed weighted data sampler

    def __len__(self):
        return len(self.data_df)    
        
    def _load_embeddings(self, protein_name):
        emb = self.all_embeddings[protein_name][:]
        return emb

    def _get_structured_features(self, protein_name):
        features = self.stc_info[protein_name][:]
        return features

    def __getitem__(self, idx):
        """
        For each peptide,return a dict contains:
            seq: sequence
            label: 1 or 0
            emb: embeddings from pretrain models
            stc: structure features
        """
        data_item = self.data_df.iloc[idx]
        label = data_item['Labels']
        tensor_label = torch.tensor(label, dtype=torch.float)
        seq =  data_item['Sequence']
        emb = []
        stc = []
        input_data = {'seq':seq,'label':tensor_label}
        if self.all_embeddings:
            emb = self._load_embeddings(seq)
        if self.stc_info:
            stc = self._get_structured_features(seq)
            input_data['stc'] = stc
        input_data['emb'] = emb
        return input_data

class AMPMultiLabel_Dataset(data.Dataset):
    """
    Datasets for Amps MultiLabel classfication
    """
    def __init__(self,data_file, embeddings_fpath=None, stc_fpath=None):
        self.data_df = pd.read_csv(data_file)
        self.all_embeddings = h5py.File(embeddings_fpath,'r') if embeddings_fpath else None
        self.stc_info = h5py.File(stc_fpath,'r') if stc_fpath else None

    def __len__(self):
        return len(self.data_df)    
        
    def _load_embeddings(self, protein_name):
        emb = self.all_embeddings[protein_name][:]
        return emb

    def _get_structured_features(self, protein_name):
        features = self.stc_info[protein_name][:]
        return features
        
    def __getitem__(self, idx):
        """
        For each peptide,return a dict contains:
            seq: sequence
            label: one hot 
            emb: embeddings from pretrain models
            stc: structure features
        """
        data_item = self.data_df.iloc[idx]
        label = data_item.iloc[1:8]
        tensor_label = torch.tensor(label, dtype=torch.float)
        seq =  data_item['Sequence']
        emb = []
        stc = []
        input_data = {'seq':seq,'label':tensor_label}
        if self.all_embeddings:
            emb = self._load_embeddings(seq)
        if self.stc_info:
            stc = self._get_structured_features(seq)
            input_data['stc'] = stc
        input_data['emb'] = emb
        return input_data

class AMPMIC_Dataset(data.Dataset):
    """
    Datasets for Amps regression and ranking
    """
    def __init__(self,data_file, embeddings_fpath=None, stc_fpath=None):
        self.data_df = pd.read_csv(data_file)
        self.all_embeddings = h5py.File(embeddings_fpath,'r') if embeddings_fpath else None
        self.stc_info = h5py.File(stc_fpath,'r') if stc_fpath else None
        self.labels = torch.tensor(self.data_df['Labels'].tolist()) # for Distributed data sampler
        
    def __len__(self):
        return len(self.data_df)

    def _load_embeddings(self, protein_name):
        emb = self.all_embeddings[protein_name][:]
        return emb

    def _get_structured_features(self, protein_name):
        features = self.stc_info[protein_name][:]
        return features

    def __getitem__(self, idx):
        """
        For each peptide,return a dict contains:
            seq: sequence
            mic: mic value of a peptides
            emb: embeddings from pretrain models
            stc: structure features
        """
        data_item = self.data_df.iloc[idx]
        mic = data_item['MIC']
        mic = torch.tensor(mic, dtype=torch.float)
        seq =  data_item['Sequence']
        label = data_item['Labels']
        tensor_label = torch.tensor(label, dtype=torch.float)
        emb = []
        stc = []
        input_data = {'seq':seq,'label':tensor_label,'mic':mic}
        if self.all_embeddings:
            emb = self._load_embeddings(seq)
        if self.stc_info:
            stc = self._get_structured_features(seq)
            input_data['stc'] = stc
        input_data['emb'] = emb
        return input_data
    