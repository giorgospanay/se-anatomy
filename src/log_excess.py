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


def read_nk_from_pandas(df):
	# Initialize an empty graph in NetworKit
	# Set 'directed=False' for an undirected graph, 'directed=True' for a directed graph
	G = nk.graph.Graph(n=max(df["PersonNr"].max(),df["PersonNr2"].max())+1, directed=False)

	# Add edges from the DataFrame to the NetworKit graph
	for index, row in df.iterrows():
	    G.addEdge(row["PersonNr"],row["PersonNr2"])

	return G

# Returns triangles per node, and scores (list of edges)
def get_node_triangles(G, out_scores=False):
	G.indexEdges()
    e_triangles=nk.sparsification.TriangleEdgeScore(G)
    e_triangles.run()

    # Run through all edges, calculate triangles per node
    print("Calculating triangle edge score")
	node_tri={}
	for u,v in G.iterEdges():
		if u not in node_tri: node_tri[u]=0
		if v not in node_tri: node_tri[v]=0
		score=e_triangles.score(u,v)

		print(f"Score({u},{v}):{score}")
		# Add score divided by 3
		node_tri[u]+=score//3
		node_tri[v]+=score//3

	if out_scores: 
		return node_tri, e_triangles.scores()
	else: 
		return node_tri

# Checks for embeddedness and tie range (second shortest path)
def get_embeddedness(G,e_scores):
	# For all edges in G: if embeddedness==0 (score=0) then find second SP

	# Otherwise continue plotting

	return



def get_tie_pairs(G):

	return


def get_approx_closeness(G,n_samples=100):
	# Best pun I can leave in my code. Shoutout to Central Cee
	central_c=nk.centrality.ApproxCloseness(G,n_samples,normalized=True)




# For all sets of networks: get triangles. More care for flat.

net_names=["flat_all"]

if mode=="calc-tri":
	net_names=["close_family","extended_family","household","neighborhood","education","work"]

for layer_name in net_names:
	# Make NetworKit graph from dataframe
	G=read_nk_from_pandas(
		pd.read_csv(f"{csv_path}/{layer_name}.csv").astype({"PersonNr":"int","PersonNr2":"int"})
	)

	# Calculate triangles for individual layers
	if mode=="calc-tri":
		# Decide column string based on layer name
		df_str=""
		if layer_name==""


		# Read node dataframe

		# Calculate triangles and save result to node df
		node_df[]=get_node_triangles(G)
		# Save result to node dataframe


	# Calculate excess closure and clustering coefficient (assuming triangle data exists)
	if mode=="calc-excess":
		pass
	# Calculate embeddedness from triangles
	if mode=="calc-embed":
		pass





