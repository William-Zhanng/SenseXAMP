from ast import Dict, Str
import imp
from .BasicDes import cal_discriptors
from .AAComposition import CalculateAAComposition,CalculateDipeptideComposition
from .Autocorrelation import CalculateNormalizedMoreauBrotoAutoTotal
from .CTD import CalculateCTD
from .QuasiSequenceOrder import GetSequenceOrderCouplingNumberTotal
from .PseudoAAC import _GetPseudoAAC,GetAPseudoAAC
import pandas as pd
import numpy as np
import os
import math
import torch
import h5py
from tqdm import tqdm
from typing import List,Dict

"""
小批量结构化数据生成
"""
def isvalid(line):
    seq = line.Sequence
    flag = True
    invalid_str = ['B','X','Z','O','U','J']
    for i in invalid_str:
        if i in seq:
            return False
    if seq.encode('utf-8').isalpha():
        return True
    return False

def cal_pep(peptide:Str)-> Dict:
    """
    Calculate structure data for a peptide
    """
    peptide = str(peptide)
    peptides_descriptor={}
    peptides_descriptor['Sequence'] = peptide
    AAC = CalculateAAComposition(peptide)
    DIP = CalculateDipeptideComposition(peptide)
    MBA = CalculateNormalizedMoreauBrotoAutoTotal(peptide, lamba=5)
    CCTD = CalculateCTD(peptide)
    QSO = GetSequenceOrderCouplingNumberTotal(peptide, maxlag=5)
    PAAC = _GetPseudoAAC(peptide,lamda=5)
    APAAC = GetAPseudoAAC(peptide, lamda=5)
    Basic = cal_discriptors(peptide)
    peptides_descriptor.update(AAC)
    peptides_descriptor.update(DIP)
    peptides_descriptor.update(MBA)
    peptides_descriptor.update(CCTD)
    peptides_descriptor.update(QSO)
    peptides_descriptor.update(PAAC)
    peptides_descriptor.update(APAAC)
    peptides_descriptor.update(Basic)

    return peptides_descriptor

def cal_pep_fromlist(peptides_list: List[Str], output_path: Str, retain_columns: List[str] = None, 
                 mic_results: List[float] = None, labels: List = None):
    """
    Calculate structure data for a list of peptides
    Args:
        peptides_list: list of peptides
        output_path: path to output csv
        retain_columns: if columns is not None, the output csv only contains columns in retain_columns
        mic_results: mic results of each peptide, if is not none, the output csv will contain 'MIC' columns
        labels: labels of each peptide, if is not none, the output csv will contain 'Label' columns
    Return:
        output_csv
    """
    print("total {} peptides ".format(len(peptides_list)))
    peptides_descriptors = []
    for idx,peptide in tqdm(enumerate(peptides_list)):
        if len(peptide) < 6:
            continue
        peptides_descriptor = cal_pep(peptide)
        if not(mic_results is None):
            peptides_descriptor["MIC"] = mic_results[idx]
        if not (labels is None):
            peptides_descriptor["Labels"] = labels[idx]
        peptides_descriptors.append(peptides_descriptor)
    output_csv = pd.DataFrame(peptides_descriptors)
    if retain_columns is not None:
        print("total {} features to reserve".format(len(retain_columns)))
        print(len(output_csv.columns))
        output_csv = output_csv[retain_columns]
    output_csv.to_csv(output_path,index=False) 
    print("The output csv shape is : {}".format(output_csv.shape))
    return output_csv

def df2h5(df,outdir):
    """
    Convert dataframe to h5 file
    """
    os.makedirs(outdir,exist_ok=True)
    all_seqs = df['sequence'].tolist()
    for seq in tqdm(all_seqs):
        f_name = os.path.join(outdir,'{}.h5'.format(seq))
        idx = df.index[(df['sequence'] == seq)].tolist()[0]
        data = df.loc[idx][1:]
        data = np.array(data,dtype=np.float32)
        data = torch.tensor(data,dtype=torch.float)
        with h5py.File(f_name, 'w') as hf:
            hf.create_dataset(seq, data=data)           

def df_norm(df):
    """
    Z-score normalization for each columns of dataframe
    Retrun:
        new_dataframe: normed dataframe
        norm_args: [n,2] array, contains mean,std of each columns
    """
    # drop all-zero columns
    newDataFrame = pd.DataFrame(index=df.index)
    columns = df.columns.tolist()
    norm_args = []
    nan_columns = []

    for col in columns:
        if (col == 'sequence'):
            newDataFrame[col] = df[col].tolist()
        else:
            data = df[col]
            mean = data.mean()
            std = data.std()
            if std != 0:
                data = ((data - mean) / std )
            contain_nan = (True in np.isnan(data.values))
            if contain_nan:
                nan_columns.append(col)
                print("nan col, std is {} mean is: {}".format(std,mean))

            newDataFrame[col] = data
            norm_args.append([mean,std])
    norm_args = np.array(norm_args)
    return newDataFrame,norm_args

if __name__ == "__main__":
    pass



