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


# Manually removes duplicate edges from graph
def remove_dups(G,n_weighted=False):
	# Create empty graph, weighted if necessary.
	G_n=nk.Graph(G.numberOfNodes(),directed=False,weighted=n_weighted)
	# Iterate over all edges and add them into new empty graph. Ignore multiedges
	G.forEdges(lambda u,v,w,id: G_n.addEdge(u,v,w=w,checkMultiEdge=True))
	return G_n

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
			if u+1 not in node_tri: node_tri[u+1]=0
			if v+1 not in node_tri: node_tri[v+1]=0
			# Add score
			node_tri[u+1]+=score
			node_tri[v+1]+=score
			# Counter++
			ctr+=1

	# Weight on (multi-)edges:: calculate triangles x weight (#layers connected)
	else:
		for u,v,w in G.iterEdgesWeights():
			score=e_scores[G.edgeId(u,v)]
			# Progress print
			if ctr%1000000==0: print(f"#{ctr//1000000}({u},{v}[w={w}]):score={score}")
			# Add to dictionary if missing
			if u+1 not in node_tri: node_tri[u+1]=0
			if v+1 not in node_tri: node_tri[v+1]=0
			# Add score*w
			node_tri[u+1]+=(score*w)
			node_tri[v+1]+=(score*w)
			# Counter++
			ctr+=1

	# Now div 2 for all adjacent edges
	for u in node_tri:
		temp=node_tri[u]
		node_tri[u]=temp//2

	if out_scores: 
		return node_tri, e_scores
	else: 
		return node_tri


# Returns tie pairs per node for a graph G
def get_tie_pairs(G,node_df):
	nbr_sum={}
	tie_pairs={}
	ctr=0
	# Iterate first over all edges
	for u,v,w in G.iterEdgesWeights():
		# Progress print
		if ctr%1000000==0: print(f"#{ctr//1000000}({u},{v}[w={w}])")
		# Add to dictionary if missing
		if u+1 not in nbr_sum: nbr_sum[u+1]=0
		if v+1 not in nbr_sum: nbr_sum[v+1]=0
		# Add edge sum to u,v
		nbr_sum[u+1]+=math.comb(int(w),2)
		nbr_sum[v+1]+=math.comb(int(w),2)
		# Counter++
		ctr+=1

	# Then iterate over all nodes to calculate final number of tie pairs
	ctr=0
	for u in nbr_sum:
		k_u=node_df.loc[u]["deg_total"]
		# P(u)=comb(deg_total(u),2)-sum_neighbours(comb(n_layers,2))
		tie_pairs[u]=math.comb(int(k_u),2)-nbr_sum[u]
		# Progress print
		if ctr%1000000==0: print(f"Node #{ctr//1000000}({u}): P(u)={tie_pairs[u]}")
		# Counter++
		ctr+=1

	return tie_pairs

# Calculates excess closure based on df rows.
def _excess(row):
	tri_actual=float(row[0])
	tri_pure=float(row[1])
	tie_pairs=float(row[2])

	if tie_pairs==0: return 0.0

	c_pure=tri_pure/tie_pairs
	c_actual=tri_actual/tie_pairs

	if c_pure==1: return 0.0
	else: 
		res=(c_actual-c_pure)/(1-c_pure)
		if res<0: return 0.0
		elif res>1: return 1.0
		return res

# Checks for embeddedness and tie range
def get_embeddedness(G):
	# Get triangles
	print("Getting triangles")
	e_triangles=nk.sparsification.TriangleEdgeScore(G)
	e_triangles.run()
	e_scores=e_triangles.scores()

	emb_dict={}
	e_check=[]

	# For all edges in G: if embeddedness==0 (score=0) then find second SP
	print("Checking embeddedness score")
	ctr=0
	for u,v,w in G.iterEdgesWeights():
		score=e_scores[G.edgeId(u,v)]
		# Calculate final score by multiplying by weight (common #edges)
		f_score=score*w
		# Progress print
		if ctr%1000000==0: print(f"#{ctr//1000000}({u},{v}[w={w}]):f_score={f_score}")
		# Add f_score count to dictionary for plotting
		if f_score not in emb_dict:
			emb_dict[f_score]=0
		emb_dict[f_score]+=1
		# If embeddedness=0: add edge to e_check
		if f_score==0:
			e_check.append((u,v))
		# Counter
		ctr+=1

	return emb_dict, e_check

# Finds second shortest path for nodes with embeddedness=0
def get_tie_range(G,e_check):
	tr_dict={}
	# For all edges in e_check:
	ctr=0
	ctr_skip=0
	for u,v in e_check:
		# Progress print plus skipped
		if ctr%10000==0: print(f"#{ctr//10000}({u},{v})")
		if ctr_skip%10000==0: print(f"Skipped#{ctr_skip//10000}.")
		# First check degree of u,v. If =1, skip.
		if G.degree(u)==1 or G.degree(v)==1: 
			G.addEdge(u,v)
			ctr+=1
			ctr_skip+=1
			continue
		# Remove edge first
		G.removeEdge(u,v)
		
		# Run SP (should be the shortest path now)
		bi_bfs=nk.distance.BidirectionalBFS(G,u,v)
		bi_bfs.run()
		dist=bi_bfs.getDistance()
		
		# If >0, calculate. Also check for inf (modeled as MAX_VALUE)
		if dist>1 and dist<G.numberOfEdges():
			tie_range=dist
			# Add to dictionary for plotting
			if tie_range not in tr_dict:
				tr_dict[tie_range]=0
			tr_dict[tie_range]+=1
		# Restore edge
		G.addEdge(u,v)
		# Counter
		ctr+=1

	return tr_dict



# Read node_b
print("Reading node_b")
node_df=pd.read_csv(f"{log_path}/filtered_node_b_2017.csv",index_col="PersonNr",header=0)
node_df.fillna(0.0,inplace=True)

# Set net-names
net_names=["flat_all"]

# Special lists for calc-tri: init node_df, iterate over all layers
if mode=="calc-tri":
	net_names=["close_family","extended_family","household","education","neighbourhood","work"]
	node_df=node_df[["deg_close","deg_ext","deg_house","deg_edu","deg_nbr","deg_work","deg_total","closeness"]]

# Special lists for fix-tri: sum node_df, no need to iterate layers
if mode=="fix-tri":
	net_names=[]
	# Calculate pure triangles (sum of tri on each layer separately)
	node_df["pure_tri"]=node_df["tri_close"]+node_df["tri_ext"]+node_df["tri_house"]+node_df["tri_nbr"]+node_df["tri_edu"]+node_df["tri_work"]

# Special lists for fix-exc: work on node_df, no need to iterate layers
if mode=="fix-excess":
	net_names=[]
	# Fix excess closure exceptions: if negative, set to 0. If over 1, set to 1.
	node_df["excess_closure"]=node_df[["actual_tri","pure_tri","tie_pairs"]].apply(_excess,axis=1)

# Special lists for fix-node: add node_df attributes. No need to iterate layers
if mode=="fix-node":
	net_names=[]
	# Read node attributes from b
	print("Reading node_attb")
	node_attb=pd.read_csv(f"{log_path}/node_attributes_2017.csv",index_col="PersonNr",header=0)

	print("Merging:")
	#node_full=pd.concat([node_df,node_attb],axis=1)
	node_full = pd.merge(node_df, node_attb, left_index=True, right_index=True)

	print("Saving:")
	node_full.fillna(0.0,inplace=True)
	node_full.to_csv(f"{log_path}/filtered_node_final_2017.csv")

# For normal modes:
for layer_name in net_names:
	print(f"Reading in {layer_name}:")
	flag_weighted_saved=True

	G=None
	# Read weighted graph if saved already
	if (mode=="calc-excess" or mode=="calc-embed") and flag_weighted_saved:
		# Make Networkit graph from weighted edgelist.
		G=nk.readGraph(f"{csv_path}/filtered_flat_all_id_w2017.csv",nk.Format.METIS)

	else:
		# Make Networkit graph from edgelist. Format EdgeListSpaceOne (sep=" ",firstNode=1)
		G=nk.readGraph(f"{csv_path}/filtered_edgelist_{layer_name}_2017.csv",nk.Format.EdgeListSpaceOne)
	
	# Manually remove duplicate edges
	print("Removing duplicate edges.")
	if mode=="calc-excess":
		G=remove_dups(G,n_weighted=True)
	else:	
		G=remove_dups(G)

	# Index all edges
	print("Indexing edges.")
	G.indexEdges()

	# For special triangle modes (excess, embeddedness): 
	#  calculate # neighbors over all edges and add as weight
	if mode=="calc-excess" or mode=="calc-embed":
		if not flag_weighted_saved:
			# Make weighted graph
			print("Making weighted graph.")
			G=nk.graphtools.toWeighted(G)

			# Set flag for grouping done here
			grouping_flag=True

			if not grouping_flag:
				# Read flat_all with ids
				print("Reading flat_all_id:")
				df=pd.read_csv(f"{csv_path}/filtered_flat_all_id_2017.csv").astype({"PersonNr":"int","PersonNr2":"int"})[["PersonNr","PersonNr2"]]

				# Find number of layers where edge exists
				print("Find n_layers")
				df["n_layers"] = df.groupby(["PersonNr","PersonNr2"])["PersonNr"].transform('size')
				df=df[["PersonNr","PersonNr2","n_layers"]]
				df=df.set_index(["PersonNr","PersonNr2"])
				df.to_csv(f"{csv_path}/filtered_flat_all_id_nl_2017.csv")
			else:
				print("Reading, setting index, sorting.")
				df=pd.read_csv(f"{csv_path}/filtered_flat_all_id_nl_2017.csv",header=0)
				# df=df.set_index(["PersonNr","PersonNr2"])
				# df=df.sort_index()

				print(df)

			# Filter: Only check rows where n_layers>=2
			print("Filter df")
			df=df[df["n_layers"]>=2][["PersonNr","PersonNr2","n_layers"]]
			print(f"Length df:{len(df)}")
			print(df)

			print("Set weights on G from filtered df")
			for index, row in df.iterrows():
				if G.hasEdge(row[0]-1,row[1]-1):
					G.setWeight(row[0]-1,row[1]-1,row[2])
				elif G.hasEdge(row[1]-1,row[0]-1):
					G.setWeight(row[1]-1,row[0]-1,row[2])
				else:
					print(f"Skipped index ({row[0]},{row[1]}).")

			# # Set weights on G accordingly
			# print("Set weights on G")
			# for u,v in G.iterEdges():
			# 	if df.index.isin([(u+1,v+1)]).any():
			# 		row=df.loc[pd.IndexSlice[(u+1,v+1)]]
			# 		val=row[["n_layers"]].values[0]
			# 		G.setWeight(u,v,val)
			# 	elif df.index.isin([(v+1,u+1)]).any():
			# 		row=df.loc[pd.IndexSlice[(v+1,u+1)]]
			# 		val=row[["n_layers"]].values[0]
			# 		G.setWeight(u,v,val)
			# 	else:
			# 		print(f"Skipped index ({u+1},{v+1}).")
			# 		print(f"Index in lv0: {u+1 in df.index.levels[0]}")
			# 		print(f"Index in lv1: {v+1 in df.index.levels[1]}")

			# Save G for future usage
			print("Saving G (weighted):")
			nk.writeGraph(G,f"{csv_path}/filtered_flat_all_id_w_2017.csv",nk.Format.METIS)

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

		# Calculate triangles
		print("Get triangles")
		node_df[df_str]=pd.Series(get_node_triangles(G))


	# Calculate local clustering coefficient
	if mode=="calc-lcc":
		print("Calculating lcc")
		lcc_scores=nk.centrality.LocalClusteringCoefficient(G).run().scores()

		# Create dictionary to pass as series
		lcc_dict={}
		for u in G.iterNodes():
			lcc_dict[u+1]=lcc_scores[u]

		node_df["lcc"]=pd.Series(lcc_dict)

	# Calculate excess closure and clustering coefficient (assuming triangle data exists)
	if mode=="calc-excess":
		# Calculate multi triangles
		print("Get multi-triangles.")
		node_df["actual_tri"]=pd.Series(get_node_triangles(G,multi_weight=True))

		# Calculate tie pairs per node
		print("Get tie pairs.")
		node_df["tie_pairs"]=pd.Series(get_tie_pairs(G,node_df))

		# Calculate excess closure
		print("Get excess closure.")
		node_df["excess_closure"]=node_df[["actual_tri","pure_tri","tie_pairs"]].apply(_excess,axis=1)
		

	# Calculate embeddedness from triangles (flat id)
	if mode=="calc-embed":
		# Calculate embeddedness distribution
		print("Get embeddedness.")
		emb_dict,e_check=get_embeddedness(G)

		# Save embeddedness dict
		print("Saving emb_dict")
		with open(f"{log_path}/filtered_embeddedness_dist_2017.txt","w") as wf:
			for key, value in emb_dict.items():
				wf.write(f"{key}: {value}\n")

		# Save e_check:
		with open(f"{log_path}/filtered_edgezero_dist_2017.txt","w") as wf:
			for u,v in e_check:
				wf.write(f"{u},{v}\n")

	# Calculate tie range from edgelist echeck
	if mode=="calc-tr":
		# Read e_check:
		print("Get tie ranges from file")
		e_check=[]
		with open(f"{log_path}/filtered_edgezero_dist_2017.txt","r") as file:
			lines=[line.rstrip().split(",") for line in file]
			for ln in lines:
				e_check.append((int(ln[0]),int(ln[1])))

		# For list of edges with embeddedness=0: get second SP
		print("Get tie range.")
		tr_dict=get_tie_range(G,e_check)

		# Save tie range dict
		print("Saving tr_dict")
		with open(f"{log_path}/filtered_tie_range_dist_2017.txt","w") as wf:
			for key, value in tr_dict.items():
				wf.write(f"{key}: {value}\n")


# Save result to node dataframe
if mode!="fix-node" and mode!="calc-embed" and mode!="calc-tr":
	print("Saving node_b")
	node_df.fillna(0.0,inplace=True)
	node_df.to_csv(f"{log_path}/filtered_node_b_2017.csv")


