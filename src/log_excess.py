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


def read_nk_from_pandas(df,multi_weight=False):
	# Initialize an empty graph in NetworKit
	# Set 'directed=False' for an undirected graph, 'directed=True' for a directed graph
	G = nk.graph.Graph(n=max(df["PersonNr"].max(),df["PersonNr2"].max())+1, weighted=multi_weight, directed=False)

	# Add edges from the DataFrame to the NetworKit graph
	for index, row in df.iterrows():
		if multi_weight:
			# First check if edge exists

			G.addEdge(row["PersonNr"],row["PersonNr2"])

		else: G.addEdge(row["PersonNr"],row["PersonNr2"])

	return G

# Returns triangles per node, and scores (list of edges)
def get_node_triangles(G,multi_weight=False,out_scores=False):
	e_triangles=nk.sparsification.TriangleEdgeScore(G)
	e_triangles.run()
	e_scores=e_triangles.scores()

	# Run through all edges, calculate triangles per node
	print("Calculating triangle edge score")
	node_tri={}
	ctr=0
	# No weight on edges:: normal triangle calculation
	if not multi_weight:
		for u,v in G.iterEdges():
			score=e_scores[G.edgeId(u,v)]
			# Progress print
			if ctr%1000000==0: print(f"#{ctr//1000000}({u},{v}):score={score}")
			# Add to dictionary if missing
			if u not in node_tri: node_tri[u]=0
			if v not in node_tri: node_tri[v]=0
			# Add score
			node_tri[u]+=score
			node_tri[v]+=score
			# Counter++
			ctr+=1
	# Weight on (multi-)edges:: calculate triangles x weight (#layers connected)
	else:
		for u,v,w in G.iterEdgesWeights():
			score=e_scores[G.edgeId(u,v)]
			# Progress print
			if ctr%1000000==0: print(f"#{ctr//1000000}({u},{v}[w={w}]):score={score}")
			# Add to dictionary if missing
			if u not in node_tri: node_tri[u]=0
			if v not in node_tri: node_tri[v]=0
			# Add score*w
			node_tri[u]+=(score*w)
			node_tri[v]+=(score*w)
			# Counter++
			ctr+=1
	if out_scores: 
		return node_tri, e_scores
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
	# Shoutout
	central_c = nk.centrality.ApproxCloseness(G,n_samples,normalized=True)




# For all sets of networks: get triangles. More care for flat.

net_names=["flat_all"]

if mode=="calc-tri":
	net_names=["close_family","extended_family","household","education","neighbourhood","work"]


if mode=="fix-tri":
	net_names=[]
	# Read node dataframe
	print("Read node_b")
	node_df=pd.read_csv(f"{log_path}/node_b_2017.csv",index_col="PersonNr",header=0)
	node_df.fillna(0.0,inplace=True)

	# Calculate pure triangles (sum of tri on each layer separately)
	node_df["pure_tri"]=node_df["tri_close"]+node_df["tri_ext"]+node_df["tri_house"]+node_df["tri_nbr"]+node_df["tri_edu"]+node_df["tri_work"]

	# Save result to node dataframe
	node_df.to_csv(f"{log_path}/node_b_2017.csv")


# For normal networks:
for layer_name in net_names:
	print(f"Reading in {layer_name}:")

	# Make NetworKit graph from dataframe
	# G=read_nk_from_pandas(
	# 	pd.read_csv(f"{csv_path}/{layer_name}.csv").astype({"PersonNr":"int","PersonNr2":"int"})
	# )

	# Make Networkit graph from edgelist. Format EdgeListSpaceOne (sep=" ",firstNode=1)
	G=nk.readGraph(f"{csv_path}/edgelist_{layer_name}2017.csv",nk.Format.EdgeListSpaceOne)
	# Index all edges
	G.indexEdges()

	# Calculate triangles for individual layers
	if mode=="calc-tri":
		# Decide column string based on layer name
		df_str=""
		if layer_name=="close_family": df_str="tri_close"
		elif layer_name=="extended_family": df_str="tri_ext"
		elif layer_name=="household": df_str="tri_house"
		elif layer_name=="neighbourhood": df_str="tri_nbr"
		elif layer_name=="education": df_str="tri_edu"
		elif layer_name=="work": df_str="tri_work"

		# Read node dataframe
		print("Read node_b")
		node_df=pd.read_csv(f"{log_path}/node_b_2017.csv",index_col="PersonNr",header=0)
		node_df.fillna(0.0,inplace=True)

		# Calculate triangles and save result to node df
		print("Get triangles")
		node_df[df_str]=pd.Series(get_node_triangles(G))

		# Save result to node dataframe
		node_df.fillna(0.0,inplace=True)
		node_df.to_csv(f"{log_path}/node_b_2017.csv")

	# Calculate local clustering coefficient??
	if mode=="calc-lcc":
		pass

	# Calculate excess closure and clustering coefficient (assuming triangle data exists)
	if mode=="calc-excess":
		# Need to only read flat (with ids?)
		pass

	# Calculate embeddedness from triangles
	if mode=="calc-embed":
		# Need to only read flat (potentially with ids)
		pass





