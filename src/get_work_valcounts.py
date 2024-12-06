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

def convert_to_remote(value):
    try:
        int_value = int(value)
        if 99980 <= int_value <= 99999:
            if int_value in [99995, 99997]:
                return str(value)
            else:
                return "Remote"
        else:
            return str(value)
    except ValueError:
        # Value is not convertible to an integer
        return str(value)

#read in data 
data = pd.read_csv(lisa_path,index_col="LopNr",usecols=["LopNr","LopNr_CfarNr","LopNr_KU1CfarNr","AstNr_LISA","AstKommun"])
#data = pd.read_csv(lisa_path)
print(f"Lisa length: {len(data.index)}")
print(data.columns)


# Read work attrbs per user
data_attb=pd.read_csv(f"{log_path}/node_final_2017.csv",index_col="PersonNr",usecols=["PersonNr","deg_work"],header=0)
# Merge with user
data=data.merge(data_attb,left_on="LopNr",right_on="PersonNr")



#filter_data=data

# Remove filtered kommuns
filter_data=data[~data["AstKommun"].astype(int).isin([0,9999])]
filter_data=filter_data[filter_data["LopNr_CfarNr"]!="-"]
print(f"Filtered Lisa length: {len(filter_data.index)}")
# Convert to remote locations
filter_data["AstNr_LISA"]=filter_data["AstNr_LISA"].astype(str).apply(convert_to_remote)

# Also remove outlier workplace
filter_data=filter_data[filter_data["LopNr_CfarNr"]!="946097.0"]


# Print users with deg=0, see what they look like
# Try further filtering for deg=0
filter_data=filter_data[filter_data["deg_work"]==0]
print(f"Filtered deg=0 Lisa length: {len(filter_data.index)}")
print(filter_data)


# val_counts=filter_data["LopNr_KU1CfarNr"].value_counts()

# print(val_counts)

group_a=filter_data.groupby(["LopNr_CfarNr","AstNr_LISA"]).agg({"AstKommun":"lambda x: value_counts(x,sort=True)"})

#.filter(lambda x: len(x)>1)

print(group_a)

# Filter
f_ga=group_a[group_a>1]

print(f_ga)
print(f_ga.sum())









