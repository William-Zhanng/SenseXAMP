import BasicDes, Autocorrelation, CTD, PseudoAAC, AAComposition, QuasiSequenceOrder
import pandas as pd
import numpy as np
# from openpyxl import Workbook
import multiprocessing
import sys
import gc

"""
多进程生成结构化数据的实现
"""
# PATH = '多肽最终参数2020.5.8.xlsx'
# data = pd.read_excel(PATH,sheet_name='Sheet1',usecols=[0])
# peptides = data.values.tolist()
# 7-19

def doit(ind):
	file = '/mnt/data1/xiaoziyang/result/result_'+str(ind)+'.csv' # 生成的序列信息
	print(file)
	data = pd.read_csv(file,encoding="utf-8")
	print("read %d done"%(ind))
	peptides_descriptors=[]
	count = 0
	
	sequence = data["sequence"]
	peptides = sequence.values.copy().tolist()
	for peptide_list in peptides:
		peptides_descriptor={}
		peptide = str(peptide_list)
		if peptide!="SSQRMW" and peptide!="WMRQSS":
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
	
	# 保存生成的结构化数据
	writeDataToExcleFile(sequence, peptides_descriptors, ('/home/xuyanchao/server_test/data/result_updated_test_v2_' + str(ind) + '.csv'))

def writeDataToExcleFile(sequence,inputData,outPutFile):
	print(inputData[0:2])
	df = pd.DataFrame(inputData)
	result = pd.concat([sequence,df], axis=1)
	result.to_csv(outPutFile)

if __name__ == "__main__":
	pros = []
	for i in range(14,20):
		process = multiprocessing.Process(target=doit, args=(i,))
		pros.append(process)
		process.start()

	for process in pros:
		process.join()


# ps aux|grep xxx|awk '{print $2}'|xargs kill -9 这里是批量杀死进程名包含xxx的进程，记录做个备忘