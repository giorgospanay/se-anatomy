import os, glob, parse, pickle, sys, gc
import networkx as nx
import igraph as ig
import numpy as np
import pandas as pd
import random
import scipy.sparse
import scipy.sparse.csgraph
import math

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
def pd_concat_layers(l1,l2,l1_id=None,l2_id=None):
	if l1_id is not None:
		l1["layer_id"]=l1_id
	if l2_id is not None:	
		l2["layer_id"]=l2_id
	return pd.concat([l1,l2],copy=False)

# Find tie pairs for each node
def find_tie_pairs(G_id):
	tie_pairs={}
	# For all nodes:
	for u in G_id:
		k_u=G.degree(u)

		# Get neighbours
		nbr=G.neighbors(u)
		sum_nbr_e=0
		for v in nbr(u):
			sum_nbr_e+=math.comb(G.number_of_edges(u,v),2)

	tie_pairs[u]=math.comb(k_u,2)-sum_nbr_e

	return tie_pairs


# Find approximate avg shortest path length by sampling
def find_avg_shortest_path(G,n_samples=10000):
	nodes=list(G.vs)
	lengths=[]
	for i in range(n_samples):
		# Progress update
		if i%1000==0: print(f"Progress: {i//1000}/{n_samples/1000}")
		# Sample two nodes to calculate shortest path length between them
		u,v=random.choices(nodes,k=2)
		## Uncomment to return to NetworkX
		#lengths.append(nx.shortest_path_length(G,source=u,target=v))
		## igraph code
		lengths.append(G_id.shortest_paths_dijkstra(source=u,target=v))

	return mean(lengths)

# Find approximate diameter by BFS traversal on psuedo-peripheral vertex
def find_pseudo_diameter(G):
    # Start from a random node
    start_vertex = random.randint(0, G.vcount()-1)
    
    # Perform the first BFS/DFS to find the farthest node from start_vertex
    distances_from_start = G.shortest_paths_dijkstra(source=start_vertex)[0]
    farthest_node_1 = distances_from_start.index(max(distances_from_start))

    # Perform the second BFS/DFS from the farthest node found
    distances_from_farthest = G.shortest_paths_dijkstra(source=farthest_node_1)[0]
    farthest_node_2 = distances_from_farthest.index(max(distances_from_farthest))
    
    # The maximum distance found in the second BFS/DFS is the approximate diameter
    approximate_diameter = max(distances_from_farthest)
    
    return approximate_diameter

# Find approximate closeness centrality by sampling
def find_closeness_centrality(G,n_samples=10000):
	# Randomly sample nodes to compute shortest paths
	sample_nodes=random.sample(G.nodes(),n_samples)

	# Compute shortest path lengths from each node in the sample
	closeness_centrality={}
	for i,node in enumerate(G.nodes()):

		# Progress update
		if i%1000==0: print(f"Progress: {i//1000}/{n_samples/1000}")

		# Calculate shortest paths from each node
		shortest_paths=nx.single_source_shortest_path_length(G,node)
		
		# Only sum up the distances to the sampled nodes
		reachable_nodes=set(shortest_paths.keys()).intersection(sample_nodes)
		if len(reachable_nodes)>1:
			total_distance=sum(shortest_paths[target] for target in reachable_nodes)
			if total_distance>0:
				# Approximate closeness centrality formula
				closeness=(len(reachable_nodes)-1)/total_distance
				# Normalize by the fraction of reachable nodes
				closeness*=(len(G)-1)/(len(reachable_nodes)-1)
			else:
				closeness=0.0
		else:
			closeness=0.0
		closeness_centrality[node]=closeness

	return closeness_centrality


# Find approximate closeness centrality by sampling
def find_closeness_centrality_target(G,n_samples=10000):
	# Randomly sample nodes to compute shortest paths
	sample_nodes=random.sample(G.nodes(),n_samples)

	shortest_paths=[]
	# For all nodes in sample: calculate shortest distance to target
	for j,node_v in enumerate(sample_nodes):
		print(f"Target #{j}: n_id:{node_v}")
		shortest_paths.append(nx.single_target_shortest_path_length(G,node))

	# Compute shortest path lengths from each node in the sample
	closeness_centrality={}
	for i,node in enumerate(G.nodes()):
		# Progress update
		if i%1000==0: print(f"Progress: {i//1000}/{n_samples/1000}")

		count_reachable=0
		sum_distances=0
		# Check if node is not disconnected (should not be the case)
		if G.degree[node]>=1:
			for j,_node_v in enumerate(sample_nodes):
				# Check if node has key in appended dict
				if node in shortest_paths[j]:
					length_reachable=shortest_paths[j][node]
					# Check again if 0
					if length_reachable!=0:
						count_reachable+=1
					sum_distances+=length_reachable
			# Approximate closeness centrality
			if sum_distances>0:
				closeness=1.0*count_reachable/sum_distances
				closeness*=(G.number_of_nodes()-1)/(len(sample_nodes)-1)
			else: closeness=0.0
		else:
			closeness=0.0
		closeness_centrality[node]=closeness

		return closeness_centrality




node_df=None
table_2=pd.DataFrame(columns=["n","m","comp","gc","diam","avg_sp"])

if mode!="calc-node":
	# Read node info df here
	if mode=="flatten":
		print("Read node_a")
		node_df=pd.read_csv(f"{log_path}/node_a_2017.csv",index_col="PersonNr",header=0)
		node_df.fillna(0.0,inplace=True)

	df=None
	df_id=None

	# Iterate over all layers (order: F-N-E-W)
	for net_name in ["family","flat_fn","flat_fne","flat_all"]:

		# Read network here
		G=None
		G_id=None
		if net_name=="family":
			print("Reading in Family 2017")
			## Removed code. Uncomment if necessary to recreate all family types of edges
			#
			# fam_df=read_in_network(pd.read_csv(f"{csv_path}/final_network2017.csv"),"PersonNr")
			# df = make_entire_edge_list(fam_df)[["PersonNr","PersonNr2"]]
			# # Save us from future calculations!!
			# df.to_csv(f"{csv_path}/family2017.csv")
			# Collect garbage
			# fam_df=None
			# gc.collect()
			df=pd.read_csv(f"{csv_path}/family2017.csv").astype({"PersonNr":"int","PersonNr2":"int"})

			
			#
			## Switch to igraph on other modes 
			#
			G=ig.Graph.DataFrame(df, directed=False)

			if mode=="flatten":
				# Do networkX here for triangle calc.
				G=nx.from_pandas_edgelist(df,source="PersonNr",target="PersonNr2")
				# Also calculate node triangles here
				print("Get triangles.")
				node_df["tri_fam"]=pd.Series(nx.triangles(G))

			G_id=G

		elif net_name=="flat_fn":
			print("Reading in Neighbourhood 2017")
			
			if "flatten" in mode:
				# Read alone for triangles
				n_df=pd.read_csv(f"{csv_path}/neighbourhood2017.csv")
				if mode=="flatten":
					G=nx.from_pandas_edgelist(n_df,source="PersonNr",target="PersonNr2")
					print("Get triangles.")
					node_df["tri_nbr"]=pd.Series(nx.triangles(G))

					print("Flatten.")
					df=pd_flatten_layers(df,n_df)
					# Save us from future calculations!!
					df.to_csv(f"{csv_path}/flat_fn2017.csv")

				if mode=="flatten-id":
					print("Flatten with ids.")
					df_id=pd_concat_layers(df,n_df,l1_id="family",l2_id="neighbourhood")
					# Save us from future calculations!!
					df_id.to_csv(f"{csv_path}/flat_fn_id2017.csv")
			else:
				df_id=pd.read_csv(f"{csv_path}/flat_fn_id2017.csv").astype({"PersonNr":"int","PersonNr2":"int"})
				#G_id=nx.from_pandas_edgelist(df_id,source="PersonNr",target="PersonNr2", edge_attr=["layer_id"], create_using=nx.MultiGraph())
				G_id=ig.Graph.DictList(df_id, directed=False)

		elif net_name=="flat_fne":
			print("Reading in Education 2017")

			if "flatten" in mode:
				# Read alone for triangles
				e_df=pd.read_csv(f"{csv_path}/education2017.csv")
				if mode=="flatten":
					G=nx.from_pandas_edgelist(e_df,source="PersonNr",target="PersonNr2")
					print("Get triangles.")
					node_df["tri_edu"]=pd.Series(nx.triangles(G))

					print("Flatten.")
					df=pd_flatten_layers(df,e_df)
					# Save us from future calculations!!
					df.to_csv(f"{csv_path}/flat_fne2017.csv")

				if mode=="flatten-id":
					print("Flatten with ids.")
					df_id=pd_concat_layers(df_id,e_df,l2_id="education")
					# Save us from future calculations!!
					df_id.to_csv(f"{csv_path}/flat_fne_id2017.csv")
			else:
				df_id=pd.read_csv(f"{csv_path}/flat_fne_id2017.csv").astype({"PersonNr":"int","PersonNr2":"int"})
				#G_id=nx.from_pandas_edgelist(df_id,source="PersonNr",target="PersonNr2", edge_attr=["layer_id"], create_using=nx.MultiGraph())
				G_id=ig.Graph.DictList(df_id, directed=False)

		elif net_name=="flat_all":
			print("Reading in Work 2017")

			if "flatten" in mode:
				# Read work alone for triangles
				w_df=pd.read_csv(f"{csv_path}/work2017.csv")
				if mode=="flatten":
					G=nx.from_pandas_edgelist(w_df,source="PersonNr",target="PersonNr2")
					print("Get triangles.")
					node_df["tri_work"]=pd.Series(nx.triangles(G))

					print("Flatten.")
					df=pd_flatten_layers(df,w_df)
					# Save us from future calculations!!
					df.to_csv(f"{csv_path}/flat_all2017.csv")

				if mode=="flatten-id":
					print("Flatten with ids.")
					df_id=pd_concat_layers(df_id,w_df,l2_id="work")
					# Save us from future calculations!!
					df_id.to_csv(f"{csv_path}/flat_all_id2017.csv")
			else:
				df_id=pd.read_csv(f"{csv_path}/flat_all_id2017.csv").astype({"PersonNr":"int","PersonNr2":"int"})
				print("Create igraph")
				#G_id=nx.from_pandas_edgelist(df_id,source="PersonNr",target="PersonNr2", edge_attr=["layer_id"], create_using=nx.MultiGraph())
				G_id=ig.Graph.DictList(df_id, directed=False)

		if mode=="calc-table":

			#### Uncomment below to switch back to NetworkX code.
			#
			#
			# # N -- number of nodes:
			# n=G_id.number_of_nodes()
			# 
			# # M -- number of edges:
			# m=G_id.number_of_edges()
			# 
			# # Debug
			# print(f"N={n} M={m}")
			# 
			# print("Finding components.")
			# # n_comps -- number of components:
			# components=sorted(nx.connected_components(G_id),key=len,reverse=True)
			# n_comps=len(components)
			# 
			# print("Finding GC.")
			# # GC -- relative size of giant component:
			# gc_pct=len(components[0])/n
			# GC=G.subgraph(components[0])
			# 
			# print("Finding approximate GC diameter.")
			# # D -- approx diameter of GC:
			# diam_len=nx.approximation.diameter(GC)
			#
			# print("Finding approximate GC shortest path")
			# # d -- (estimated) average shortest path of GC:
			# d_len=find_avg_shortest_path(GC,n_samples=5000)

			#### Switch to igraph code:
			# N -- number of nodes:
			n=G_id.vcount()
			
			# M -- number of edges:
			m=G_id.ecount()
			
			# Debug
			print(f"N={n} M={m}")
			
			print("Finding components.")
			# n_comps -- number of components:
			components=sorted(G_id.decompose(mode="weak"),key=lambda s: s.vcount(),reverse=True)
			n_comps=len(components)
			
			print("Finding GC.")
			# GC -- relative size of giant component:
			GC=components[0]
			gc_pct=GC.vcount()/n

			print(f"#Comps={n_comps} GC={gc_pct}")
			
			print("Finding approximate GC diameter.")
			# D -- (approx) diameter of GC:
			# @TODO: try now and see if it works
			#diam_len=GC.diameter(directed=False)
			diam_len=find_pseudo_diameter(GC)
			
			print(f"Diameter:{diam_len}")

			print("Finding approximate GC shortest path")
			# d -- (estimated) average shortest path of GC:
			d_len=find_avg_shortest_path(GC,n_samples=5000)

			print(f"SP:{d_len}")

			# Add to table 2
			f_df=pd.DataFrame({"n":[n],"m":[m],"comp":[n_comps],"gc":[gc_pct],"diam":[diam_len],"avg_sp":[d_len]})
			table_2=pd.concat([table_2,f_df],axis=0)

	if mode=="flatten":
		# Print out new node csv
		node_df.to_csv(f"{log_path}/node_b_2017.csv")

	if mode=="calc-table":
		# Print out Table 2
		table_2.to_csv(f"{plot_path}/table_2.csv",index=False)


# For flat:
if mode=="calc-node":
	# Read node_b
	node_df=pd.read_csv(f"{log_path}/node_b_2017.csv",index_col="PersonNr",header=0)

	# Read flat_all (no id)
	print("create nx all")
	G=nx.from_pandas_edgelist(pd.read_csv(f"{csv_path}/flat_all2017.csv").astype({"PersonNr":"int","PersonNr2":"int"}),source="PersonNr",target="PersonNr2")

	# Calculate approx closeness centrality (sample size: 0.03% of GC)
	print("Get approx closeness centrality (flat).")
	node_df["closeness"]=pd.Series(find_closeness_centrality_target(G,n_samples=int(len(components[0])*0.0003)))
	# flat: clustering coefficient
	print("Get local clustering coefficient (flat).")
	node_df["lcc"]=pd.Series(nx.clustering(G))

	# Collect garbage
	df=None
	G=None

	# Read flat_all with ids
	print("Read flat id")
	df_id=pd.read_csv(f"{csv_path}/flat_all_id2017.csv")
	G_id=nx.from_pandas_edgelist(df_id,source="PersonNr",target="PersonNr2", edge_attr=["layer_id"], create_using=nx.MultiGraph())

	# flat_id: excess closure
	print("Get excess closure")
	node_df["sum_tri"]=node_df["tri_fam"]+node_df["tri_nbr"]+node_df["tri_edu"]+node_df["tri_work"]
	print("Get tie pairs")
	node_df["tie_pairs"]=pd.Series(find_tie_pairs(G_id))

	def _cpure(row):
		tpure=row[0]
		tpairs=row[1]
		if tpure==0 or tpairs==0: return 0
		else: return tpure/tpairs

	def _excess(row):
		c_unique=row[0]
		c_pure=row[1]
		if c_pure==1: return 0
		else: return (c_unique-c_pure)/(1-c_pure)

	print("Get excess closure")
	node_df["c_pure"]=node_df[["sum_tri","tie_pairs"]].apply(_cpure,axis=1)
	node_df["excess"]=node_df[["lcc","c_pure"]].apply(_excess,axis=1)

	
	# Also print out new node csv
	node_df.to_csv(f"{log_path}/node_c_2017.csv")


