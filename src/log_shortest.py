import os, glob, parse, pickle, sys, gc
import networkx as nx
import numpy as np
import pandas as pd
import random

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


# Find approximate avg shortest path length by sampling
def find_avg_shortest_path(G,n_samples=10000):
	nodes=list(G.nodes())
	lengths=[]
	for i in range(n_samples):
		if i%1000==0: print(f"Progress: {i//1000}/{n_samples}")
		u,v=random.choices(nodes,k=2)
		lengths.append(nx.shortest_path_length(G,source=u,target=v))

	return mean(lengths)


# Modes: prepare flat and log (reading from flat)


# Read node info df here
node_df=pd.read_csv(f"{log_path}/node_2017.csv",index_col="PersonNr",header=0)

table_2=pd.DataFrame(columns=["n","m","comp","gc","diam","avg_sp"])

for net_name in ["family","flat_fn","flat_fne","flat_all"]:

	# Read network here
	df=None
	G=None
	if net_name=="family":
		print("Reading in Family 2017")
		fam_df=read_in_network(pd.read_csv(f"{csv_path}/final_network2017.csv"),"PersonNr")
		df = make_entire_edge_list(fam_df)[["PersonNr","PersonNr2"]]
		fam_df=None
		gc.collect()

		# Also calculate node triangles here
		print("Get triangles.")
		G=nx.from_pandas_edgelist(df,source="PersonNr",target="PersonNr2")
		node_df["tri_fam"]=pd.Series(nx.triangles(G))

	elif net_name=="flat_fn":
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

	elif net_name=="flat_fne":
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

	elif net_name=="flat_all":
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

		# Calculate flat: closeness centrality
		print("Get closeness centrality (flat).")
		node_df["closeness_centrality"]=pd.Series(nx.closeness_centrality(G))
		# flat: clustering coefficient
		print("Get local clustering coefficient (flat).")
		node_df["lcc"]=pd.Series(nx.clustering(G))



	# N -- number of nodes:
	n=G.number_of_nodes()

	# M -- number of edges:
	m=G.number_of_edges()

	print("Finding components.")
	# n_comps -- number of components:
	components=sorted(nx.connected_components(G), key=len, reverse=True)
	n_comps=len(components)

	print("Finding GC.")
	# GC -- relative size of giant component:
	gc_pct=len(components[0])/n
	GC=G.subgraph(components[0])

	print("Finding approximate GC diameter.")
	# D -- approx diameter of GC:
	diam_len=nx.approximation.diameter(GC)

	print("Finding approximate GC shortest path")
	# d -- (estimated) average shortest path of GC:
	d_len=find_avg_shortest_path(GC,n_samples=20000)

	# Collect garbage
	G=None
	GC=None
	gc.collect()

	# Add to table 2
	f_df=pd.DataFrame({"n":[n],"m":[m],"comp":[n_comps],"gc":[gc_pct],"diam":[diam_len],"avg_sp":[d_len]})
	table_2=pd.concat([table_2,f_df],axis=0)

# Print out Table 2
table_2.to_csv(f"{plot_path}/table_2.csv",index=False)

# Also print out new node csv
node_df.to_csv(f"{log_path}/node_2017.csv")

