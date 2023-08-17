import os
import argparse
import pandas as pd
import sys
from structure_data_generate.cal_pep_des import cal_pep_fromlist

if __name__ == '__main__':
    # generate structured data
    data_dir = './datasets/ori_datasets/AMPlify'
    files = os.listdir(data_dir)
    out_dir = './datasets/stc_datasets/AMPlify'
    os.makedirs(out_dir,exist_ok=True)

    for file in files:
        data_file = os.path.join(data_dir,file)
        data = pd.read_csv(data_file, encoding="utf-8")  
        sequence = data['Sequence']
        labels = data['Labels']
        # labels = data['MIC']
        peptides_list = sequence.values.copy().tolist()
        out_path = os.path.join(out_dir,file)
        print("output path: {}".format(out_path))
        cal_pep_fromlist(peptides_list,output_path = out_path, labels=labels)



