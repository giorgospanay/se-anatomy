import pandas as pd

## GLOBALS
csv_path="../results2"
log_path="../result_logs"
plot_path="../result_plots"
obj_path=csv_path

lisa_path="../../mat_lev_lisa_2017.csv"
deso_path="../../mat_lev_deso_2017.csv"


#read in data 
data = pd.read_csv(lisa_path, usecols=["LopNr", "Alder", "Sun2000niva", "Raks_SummaInk"])
print(f"Lisa length: {len(data.index)}")
data_deso = pd.read_csv(deso_path)

dict_columns = {"LopNr": "PersonNr", "Alder": "age"}
data.rename(columns=dict_columns, inplace=True)
data_deso = data_deso.rename(columns={"LopNr": "PersonNr"})


data_deso['DeSO'] = data_deso['DeSO'].astype(str)
data_deso['DeSO'] = data_deso['DeSO'].apply(lambda x: x[4] if len(x) > 4 else '')


data = data.merge(data_deso, how='inner', on='PersonNr')

#conversion of SUN_niva 
data['Sun2000niva'] = data['Sun2000niva'].astype(str)
data['education_level'] = data['Sun2000niva'].apply(lambda x: x[0] if len(x) > 0 else '')

di = {"1":"Primary","2":"Primary","3":"Secondary","4":"Tertiary","5":"Tertiary","6":"Tertiary","9":None}
data.replace({"education_level": di},inplace=True)

deso_i = {"A":"Not urban","B":"Urban","C":"Strongly urban","R":None}
data.replace({"DeSO":deso_i},inplace=True)

# income into deciles --> need to find that variable again
#what means what again 
data['income_group'] = pd.qcut(data['Raks_SummaInk'], q=10, labels=False)


#select variables 
data = data[["PersonNr","age", "DeSO", "income_group", "education_level"]]
data.to_csv(f"{log_path}/node_attributes_2017.csv", index=False)

