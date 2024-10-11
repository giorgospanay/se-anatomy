import os, glob, parse, pickle, sys, gc
import networkx as nx
import numpy as np
import pandas as pd
import random
import scipy.sparse
import scipy.sparse.csgraph

# Import approximate diameter function
import networkx.algorithms.approximation
from statistics import mean

# Local imports
from simplify_family import read_in_network, simplify_family_layer, make_entire_edge_list


## GLOBALS
csv_path="../results2"
log_path="../result_logs"
plot_path="../result_plots"
obj_path=csv_path


# Read cmd args
args=sys.argv[1:]
mode=""
if len(args)>=1:
	mode=args[0]


# Flatten layers using pandas
def pd_flatten_layers(l1,l2):
	return pd.concat([l1,l2],copy=False).drop_duplicates().reset_index(drop=True)

# Concatenate layers (keeping layer id)
def pd_concat_layers(l1,l2,l1_id,l2_id):
	return





# Read node info df here
node_df=pd.read_csv(f"{log_path}/node_b_2017.csv",index_col="PersonNr",header=0)


# Read network here
df=None
G=None
print("Reading in Family 2017")
fam_df=read_in_network(pd.read_csv(f"{csv_path}/final_network2017.csv"),"PersonNr")
df = make_entire_edge_list(fam_df)[["PersonNr","PersonNr2"]]
fam_df=None
gc.collect()

# Also calculate node triangles here
print("Get triangles.")
G=nx.from_pandas_edgelist(df,source="PersonNr",target="PersonNr2")
node_df["tri_fam"]=pd.Series(nx.triangles(G))


print("Reading in Neighbourhood 2017")
		
# Read alone for triangles
n_df=pd.read_csv(f"{csv_path}/neighbourhood2017.csv")
G=nx.from_pandas_edgelist(n_df,source="PersonNr",target="PersonNr2")
print("Get triangles.")
node_df["tri_nbr"]=pd.Series(nx.triangles(G))

print("Flatten.")
df=pd_flatten_layers(df,n_df)
n_df=None
G=nx.from_pandas_edgelist(df,source="PersonNr",target="PersonNr2")


print("Reading in Education 2017")

# Read alone for triangles
e_df=pd.read_csv(f"{csv_path}/education2017.csv")
G=nx.from_pandas_edgelist(e_df,source="PersonNr",target="PersonNr2")
print("Get triangles.")
node_df["tri_edu"]=pd.Series(nx.triangles(G))

print("Flatten.")
df=pd_flatten_layers(df,e_df)
e_df=None
G=nx.from_pandas_edgelist(df,source="PersonNr",target="PersonNr2")


print("Reading in Work 2017")

# Read work alone for triangles
w_df=pd.read_csv(f"{csv_path}/work2017.csv")
print("Get triangles (work).")
G=nx.from_pandas_edgelist(w_df,source="PersonNr",target="PersonNr2")
node_df["tri_work"]=pd.Series(nx.triangles(G))

print("Flatten.")
df=pd_flatten_layers(df,w_df)
w_df=None
G=nx.from_pandas_edgelist(df,source="PersonNr",target="PersonNr2")


