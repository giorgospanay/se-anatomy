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
lisa_path="../../mat_lev_lisa_2017.csv"


# Read cmd args
args=sys.argv[1:]
mode=""
top=""
if len(args)>=1:
	mode=args[0]
	if len(args)>=2:
		top=args[1]


# Set top as single layer
layer_names=["close_family","extended_family","household","neighbourhood","education","work"]
if top!="":
	layer_names=[top]

node_df=None
lisa_df=None

if mode=="make-edge":
	# Read from node attribute to compare
	print(f"Reading LISA dataframe")
	lisa_df=pd.read_csv(lisa_path,index_col="LopNr",usecols=["LopNr","LopNr_CfarNr"])
	lisa_members=list(lisa_df.index)

for layer_name in layer_names:

	# Mode create edgelist
	if mode=="make-edge":
		# Reading layer
		print(f"Reading {layer_name}.")
		df=pd.read_csv(f"{csv_path}/{layer_name}2017.csv")[["PersonNr","PersonNr2"]]

		# Mask out everything not in lisa_members
		print(f"Dropping non-LISA members.")
		df=df.drop(df[~df["PersonNr"].isin(lisa_members) & ~df["PersonNr2"].isin(lisa_members)].index)

		# Save to csv
		print(f"Saving to csv & edgelist.")
		df.to_csv(f"{csv_path}/filtered_{layer_name}_2017.csv")
		# Save to edgelist
		df.to_csv(f"{csv_path}/filtered_edgelist_{layer_name}_2017.csv",sep=" ",index=False,header=False)

	# Mode calc degree
	elif mode=="calc-degs":

		print(f"Reading {layer_name} from edgelist.")
		G=nk.readGraph(f"{csv_path}/filtered_edgelist_{layer_name}_2017.csv",nk.Format.EdgeListSpaceOne)

		# Get degrees for each node
		print(f"Calculating degrees, making dict.")

		# Do manually?
		degs=[]

		# degs=nk.centrality.DegreeCentrality.run(G).scores()
		# print(degs)

		# Sort into dictionary
		deg_dict={}
		for u in G.iterNodes():
			# Sanity check
			if G.hasNode(u):
				#print(f"Node {u} tested:")
				d_val=G.degree(u)

				# Add degree into list and dict
				degs.append(d_val)
				deg_dict[u+1]=d_val

		
		# Set short layer name
		layer_short=""
		if layer_name=="close_family":
			layer_short="close"
		elif layer_name=="extended_family":
			layer_short="ext"
		elif layer_name=="household":
			layer_short="house"
		elif layer_name=="education":
			layer_short="edu"
		elif layer_name=="neighbourhood":
			layer_short="nbr"
		elif layer_name=="work":
			layer_short="work"

		# If first layer, set as node_df.
		if layer_name=="close_family":
			node_df=pd.DataFrame.from_dict(deg_dict,orient="index",columns=[f"deg_{layer_short}"])

		# Otherwise add as column
		else:
			layer_df=pd.DataFrame.from_dict(deg_dict,orient="index",columns=[f"deg_{layer_short}"])
			node_df=node_df.merge(layer_df,left_index=True,right_index=True)


		# Also make degree distribution:
		deg_dist=sorted(degs,reverse=True)
		deg_d,freq_d=np.unique(deg_dict,return_counts=True)
		# ...and save to file as with old log_degs
		with open(f"{log_path}/filtered_histogram_{layer_name}_2017.txt","w") as h_wf:
			h_wf.write(f"{freq_d}\n")



# If degrees calculated: save node_df
if mode=="calc-degs":
	print("Saving node_df.")
	node_df.to_csv(f"{log_path}/filtered_node_a_2017.csv")


