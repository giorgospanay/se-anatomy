import pandas as pd

## GLOBALS
csv_path="../results2"
log_path="../result_logs"
plot_path="../result_plots"
obj_path=csv_path

lisa_path="../../mat_lev_lisa_2017.csv"
deso_path="../../mat_lev_deso_2017.csv"


#read in data 
#data = pd.read_csv(lisa_path, usecols=["CfarNr", "Alder", "AstNr_LISA"])
data = pd.read_csv(lisa_path)
print(f"Lisa length: {len(data.index)}")
print(data.columns)

# data_deso = pd.read_csv(deso_path)
# print(f"Deso length: {len(data_deso.index)}")


# get value counts for companies
print(data["CfarNr"].value_counts())


# get value counts for workplaces?


