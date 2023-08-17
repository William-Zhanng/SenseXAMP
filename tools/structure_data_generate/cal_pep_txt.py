import BasicDes, Autocorrelation, CTD, PseudoAAC, AAComposition, QuasiSequenceOrder
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
# from openpyxl import Workbook
import multiprocessing
import sys
import gc

"""
多进程生成结构化数据的实现
"""

def doit(ind,sequence_list):

    sequences = []
    for sequence in sequence_list:
        sequences.append(sequence[:-1])
    peptides_descriptors=[]
    count = 0
    peptides = sequences
    print(sequences[:10])
    print(len(sequences))

    for peptide_list in peptides:
        peptides_descriptor={}
        peptide = str(peptide_list)
        # if peptide!="SSQRMW" and peptide!="WMRQSS":
        AAC = AAComposition.CalculateAAComposition(peptide)
        DIP = AAComposition.CalculateDipeptideComposition(peptide)
        MBA = Autocorrelation.CalculateNormalizedMoreauBrotoAutoTotal(peptide, lamba=5)
        CCTD = CTD.CalculateCTD(peptide)
        QSO = QuasiSequenceOrder.GetSequenceOrderCouplingNumberTotal(peptide, maxlag=5)
        PAAC = PseudoAAC._GetPseudoAAC(peptide,lamda=5)
        APAAC = PseudoAAC.GetAPseudoAAC(peptide, lamda=5)
        Basic = BasicDes.cal_discriptors(peptide)
        peptides_descriptor.update(AAC)
        peptides_descriptor.update(DIP)
        peptides_descriptor.update(MBA)
        peptides_descriptor.update(CCTD)
        peptides_descriptor.update(QSO)
        peptides_descriptor.update(PAAC)
        peptides_descriptor.update(APAAC)
        peptides_descriptor.update(Basic)
        peptides_descriptors.append(peptides_descriptor)
        if count%1000==0:
            print(peptide, count)
        count+=1
    gc.collect()
    # 保存生成的结构化数
    writeDataToExcleFile(sequences, peptides_descriptors, ('/home/xyc/peptide/sequence_generate/structured_data/7_peptide_7_' + str(ind) + '.csv'))

def writeDataToExcleFile(sequence,inputData,outPutFile):
    print(inputData[0:2])
    df = pd.DataFrame(inputData)
    tmp = {"sequence":sequence}
    sequence = DataFrame(tmp)
    result = pd.concat([sequence,df], axis=1)
    result.to_csv(outPutFile)

if __name__ == "__main__":
    file_path = "/home/xyc/peptide/sequence_generate/7_peptide_result/7_peptide_rule_7.txt" # 2657205/166075
    print(file_path)
    sequence_list = open(file_path,"r")
    contents = sequence_list.readlines()
    seq_count = len(contents) # 总数
    print(seq_count)
    chunk = int(seq_count/16) # 分片
    count = 0
    pros = []
    for i in range(8,16):
        if i<15:
            sequence_list = contents[(i*chunk):((i+1)*chunk)]
            # sequence_list = contents[0:100]
        else:
            sequence_list = contents[15*chunk:]

        process = multiprocessing.Process(target=doit, args=(i,sequence_list))
        pros.append(process)
        process.start()
    
    for process in pros:
        process.join()