import pandas as pd

## GLOBALS
csv_path="../results2"
log_path="../result_logs"
plot_path="../result_plots"
obj_path=csv_path

lisa_path="../../mat_lev_lisa_2017.csv"
deso_path="../../mat_lev_deso_2017.csv"

"""
Index(['Unnamed: 0', 'LopNr', 'LopNr_FamId', 'LopNr_PeOrgNr', 'LopNr_CfarNr',
	   'LopNr_ArbstId', 'LopNr_KU3CfarNr', 'LopNr_KU2CfarNr',
	   'LopNr_KU1CfarNr', 'LopNr_KU3PeOrgNr', 'LopNr_KU2PeOrgNr',
	   'LopNr_KU1PeOrgNr', 'Alder', 'AstKommun', 'AstLan', 'AstNr_LISA',
	   'Barn0_3', 'Barn11_15', 'Barn16_17', 'Barn18_19', 'Barn20plus',
	   'Barn4_6', 'Barn7_10', 'Distriktskod', 'ExamAr', 'ExamKommun',
	   'LopNr_Fastlopnr_fastbet', 'FodelseAr', 'Kommun', 'Kon', 'KU1AstKommun',
	   'KU1AstLan', 'KU1AstNr', 'KU1Ink', 'KU1YrkStalln', 'KU2AstKommun',
	   'KU2AstLan', 'KU2AstNr', 'KU2Ink', 'KU2YrkStalln', 'KU3AstKommun',
	   'KU3AstLan', 'KU3AstNr', 'KU3Ink', 'KU3YrkStalln', 'Lan',
	   'Raks_EtablGrad', 'Raks_EtablGrans', 'Raks_Forvink',
	   'Raks_Huvudanknytning', 'Raks_SummaInk', 'Sun2000Grp', 'Sun2000Inr',
	   'Sun2000niva', 'Sun2000niva_old', 'SyssStat11', 'YrkStalln',
	   'YrkStallnKomb'],
"""

#read in data 
#data = pd.read_csv(lisa_path, usecols=["LopNr_CfarNr","LopNr_KU1CfarNr","AstNr_LISA"])
data = pd.read_csv(lisa_path)
print(f"Lisa length: {len(data.index)}")
print(data.columns)

# data_deso = pd.read_csv(deso_path)
# print(f"Deso length: {len(data_deso.index)}")



filter_data=data
#filter_data=data[~data["LopNr_KU1CfarNr"].isin(["0000","9999"])]
#print(f"Filtered Lisa length: {len(filter_data.index)}")


val_counts=filter_data["LopNr_CfarNr"].value_counts()

print(val_counts)

group_a=data.groupby("LopNr_CfarNr").filter(lambda x: len(x)>1)

print(group_a)


val_counts2=val_counts[val_counts<=1]

# with open(f'{log_path}/filtered_counts.txt', 'w') as f:
# 	f.write(val_counts2.to_string())

print(val_counts2)

print(val_counts2.sum())

# get value counts for companies
#com_vals=data["LopNr_CfarNr"].value_counts()
#print(data.groupby("LopNr_CfarNr").filter(lambda x: len(x)>1) )
#print(com2_numw.value_counts())
#print(com2_numw.value_counts().sum())

#com2_sumw=com2_numw.value_counts()
#print(com2_sumw)
#print(com2_sumw.sum())




# get value counts for workplaces?


