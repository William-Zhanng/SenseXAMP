import numpy as np

def cal_discriptors(secq):
	
	peptide_information = {
		'A':np.array([89.1, 0, -1, 0, 0.25, 1]),
		'R':np.array([174.2, 1, 1, 0, -1.8, 6.13]),
		'N':np.array([132.1, 0, 1, 0,-0.64, 2.95]),
		'D':np.array([133.1, -1, 1, 0, -0.72, 2.78]),
		'C':np.array([121.2, 0, 1, 0, 0.04, 2.43]),
		'Q':np.array([146.2, 0, 1, 0, -0.69, 3.95]),
		'E':np.array([147.1, -1, 1, 0, -0.62, 3.78]),
		'G':np.array([75.1, 0, -1, 0, 0.16, 0]),
		'H':np.array([155.2, 1, 1, 0, -0.4, 4.66]),
		'I':np.array([131.2, 0, -1, 0, 0.73, 4]),
		'L':np.array([131.2, 0, -1, 0, 0.53, 4]),
		'K':np.array([146.2, 1, 1, 0, -1.1, 4.77]),
		'M':np.array([149.2, 0, -1, 0, 0.26, 4.43]),
		'F':np.array([165.2, 0, -1, 1, 0.61, 5.89]),
		'P':np.array([115.1, 0, -1, 0, -0.07, 2.72]),
		'S':np.array([105.1, 0, 1, 0, -0.26, 1.6]),
		'T':np.array([119.1, 0, 1, 0, -0.18, 2.6]),
		'W':np.array([204.2, 0, -1, 1, 0.37, 8.08]),
		'Y':np.array([181.2, 0, 1, 1, 0.02, 6.47]),
		'V':np.array([117.2, 0, -1, 0, 0.54, 3])
	}

	sum_of_all = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	positive_charge = 0
	negative_charge = 0
	polar_number = 0
	unpolar_number = 0
	ph_number = 0
	hydrophobicity = 0
	van_der_Waals_volume = 0
	
	for amino_acid in secq:
		sum_of_all += peptide_information[amino_acid]
		if peptide_information[amino_acid][1] == 1:
			positive_charge += 1
		elif peptide_information[amino_acid][1] == -1:
			negative_charge += 1
		if peptide_information[amino_acid][2] == 1:
			polar_number += 1
		elif peptide_information[amino_acid][2] == -1:
			unpolar_number += 1
	ph_number = sum_of_all[3]
	weight = sum_of_all[0] - (len(secq)-1)*18
	charge_of_all = sum_of_all[1]
	hydrophobicity = sum_of_all[4]/len(secq)
	van_der_Waals_volume = sum_of_all[5]/len(secq)
	
	pep_discriptor = {
	'Mw':weight,
	'charge of all':charge_of_all,
	'positive_charge':positive_charge,
	'negative_charge':negative_charge,
	'polar_number':polar_number,
	'unpolar_number':unpolar_number,
	'ph_number':ph_number,
	'hydrophobicity':hydrophobicity,
	'vdW_volume':van_der_Waals_volume,
	}
	
	return pep_discriptor
