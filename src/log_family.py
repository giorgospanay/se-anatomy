import os, glob, parse, pickle, sys, gc
import networkx as nx
import numpy as np
import pandas as pd

# Local imports
from simplify_family import read_in_network, simplify_family_layer, make_entire_edge_list

## GLOBALS
csv_path="../results2"
log_path="../result_logs"
plot_path="../result_plots"
obj_path=csv_path


args=sys.argv[1:]
mode=""
half=""

if len(args)>=1:
	mode=args[0]
	if len(args)>=2:
		half=args[1]


net_names=["close_family","extended_family","household"]

for name in net_names:

	print(f"Reading {name}:")
	df=pd.read_csv(f"{csv_path}/{name}2017.csv")

	print("Creating network")
	net_year=nx.from_pandas_edgelist(df,source="PersonNr",target="PersonNr2")

	print("Calculating degrees")
	# Calculate degrees & deg. histogram and save to a file
	degs=net_year.degree()
	with open(f"{log_path}/degrees_{name}2017.txt","w") as d_wf:
		for line in degs:
			d_wf.write(f"{line}\n")
	degs=None

	print("Calculating degree histogram")
	deg_hist=nx.degree_histogram(net_year)
	with open(f"{log_path}/histogram_{name}2017.txt","w") as h_wf:
		h_wf.write(f"{deg_hist}")
	deg_hist=None

