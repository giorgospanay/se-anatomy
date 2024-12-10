import os, glob, parse, pickle, sys, gc
import numpy as np
import pandas as pd
import random
import scipy.sparse
import scipy.sparse.csgraph
import math
import networkit as nk

# Import pyteexgraph
import pyteexgraph as teex

## GLOBALS
csv_path="../results2"
log_path="../result_logs"
plot_path="../result_plots"
obj_path=csv_path


# Read cmd args
args=sys.argv[1:]
mode=""
top=""
if len(args)>=1:
	mode=args[0]
	if len(args)>=2:
		top=args[1]


# Generate degrees for work layer using networkit. Compare to node attbs available.

layer_name="work"

print(f"Reading {layer_name}.")
G=nk.readGraph(f"{csv_path}/edgelist_{layer_name}2017.csv",nk.Format.EdgeListSpaceOne)


# Get degrees for each node
print(f"Calculating degrees.")

# Do manually?
degs=[]

# degs=nk.centrality.DegreeCentrality.run(G).scores()
# print(degs)

# Sort into dictionary
deg_dict={}
for u in G.iterNodes():
	print(f"Node {u} tested:")
	d_val=G.degree(u)

	# Add degree into list and dict
	degs.append(d_val)
	deg_dict[u+1]=d_val

# Now compare to deg_work from node attribute
print(f"Reading node_attb dataframe. Only use work col")
node_df=pd.read_csv(f"{log_path}/node_final_2017.csv",usecols=["PersonNr","deg_work"],
	index_col="PersonNr",header=0)

print(f"Making new degrees a column:")
work_df = pd.DataFrame(deg_dict,columns=["PersonNr","deg_work_2"])
work_df.set_index("PersonNr",inplace=True)

print(f"Merging dataframes.")
node_df=node_df.merge(work_df,left_on="PersonNr",right_on="PersonNr")

print(node_df)

print(f"Calculate diff sum.")
node_df["work_diff"]=node_df.apply(lambda x: x[0]-x[1],axis=1)
node_df=node_df[node_df["work_diff"]<0]

print(node_df)



# # If we need sorted deg. dist later:
# deg_dist=sorted(degs, reverse=True)

