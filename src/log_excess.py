import os, glob, parse, pickle, sys, gc
import networkx as nx
import igraph as ig
import numpy as np
import pandas as pd
import random
import scipy.sparse
import scipy.sparse.csgraph
import math

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



def 


def get_excess_closure(G, selected_nodes=[], node_type="label", selected_layers=[], layer_type = "layer", batchsize=100000):
		
	# ====================== input handling ========================

		# handling layer input
		if layer_type not in ["label", "layer", "binary", "group"]:
			raise ValueError(f"Invalid layer_type '{layer_type}'. Please choose from 'label' or 'layer' or 'binary'.")
		
		for layer in selected_layers:
			if layer not in self.layers[layer_type].tolist():
				raise ValueError(f"Invalid layer '{layer}' for layer_type '{layer_type}'. Please choose from {self.layers[layer_type].unique().tolist()}.")
		
		if len(selected_layers)==0:
			selected_layers = self.layers[layer_type].unique().tolist()

		# get corresponding binary representation
		binary_repr = sum([self._layer_conversion_dict[f"{layer_type}_to_binary"][layer] for layer in selected_layers])
	
		# handling node input
		if len(selected_nodes)==0:
			selected_nodes = self.nodes["id"].tolist()
		else:
			if node_type == "label":
				selected_nodes = [self.to_id(n) for n in selected_nodes]
			elif node_type == "id":
				pass
			else:
				raise ValueError(f"Invalid node_type '{node_type}'. Please choose from 'label' or 'id'.")

		# ====================== actual calculation ========================
		# obtain a copy of the sparse adjacency matrix such that each element
		# A[i,j] contains the number of links between i and j
		A = deepcopy(self.A)
		A.data = A.data & binary_repr
		# this is necessary, otherwise the denominators will not be correct
		# (we're using indptr and indices to calculate neighbor degrees)
		A.eliminate_zeros()


		# create lookup table for mapping binary values to number of layers / multiplexity
		lookup = {}
		for num in np.unique(A.data):
			lookup[num] = bin(num).count("1")
		# apply lookup table
		A.data = np.array([lookup[x] for x in A.data])

		# find all triangles
		triangles = np.empty(0, dtype=np.int64)
		# B = A @ A @ A contains the number of paths of length 3 between B[i,j]
		# so B.diagonal() contains the number of triangles:
		#     paths of length 3 between a node and itself
		# // 2 as every path is found in both directions
		# B will not fit in memory, so we do this in steps
		for i in range(0, len(selected_nodes), batchsize):
			start = i
			end = i+batchsize
			# if end is larger than matrix size, let it be the matrixsize
			if end > len(selected_nodes):
				end = len(selected_nodes)
			A_ = A[selected_nodes[start:end],:]
			res = (A_ @ A @ A_.T).diagonal() // 2
			triangles = np.concatenate((triangles, res))

		# get number of triangles which only use 1 layer
		pure_triangles = np.zeros(len(selected_nodes))
		for layer in selected_layers:
			t = np.empty(0, dtype=np.int64)
			lA = self.get_layer_adjacency_matrix(layer=layer, layer_type=layer_type)
			for i in range(0,  len(selected_nodes), batchsize):
				start = i
				end = i+batchsize
				# if end is larger than matrix size, let it be the matrixsize
				if end > len(selected_nodes):
					end = len(selected_nodes)
				A_ = lA[selected_nodes[start:end],:]
				res = (A_ @ lA @ A_.T).diagonal() // 2
				t = np.concatenate((t, np.array(res)))
			pure_triangles += t

		# decrease matrix size to only selected nodes
		A = A[selected_nodes,:]

		# compute the denominator
		# sum of neighbor degrees, over 2
		l = comb(A.sum(axis=1), 2)
		# sum of: neighbor degrees over two
		r = csr_matrix((comb(A.data, 2), A.indices, A.indptr),(len(selected_nodes),self.N)).sum(axis=1)
		P = np.array(l - r).T[0]
		# we ensure that division by zero errors are correctly handled and nan/inf
		# values are avoided    

		# pure_triangles / P
		Cpure = np.divide(pure_triangles, P, out=np.zeros(len(selected_nodes)), where=P!=0)
		# triangles / P
		Cunique = np.divide(triangles, P, out=np.zeros(len(selected_nodes)), where=P!=0)

		# (Cunique - Cpure) / (1 - Cpure)
		excess_closure = np.divide(Cunique-Cpure, 1-Cpure, out=np.zeros(len(selected_nodes)), where=(1-Cpure)!=0)
		clustering_coefficient = Cunique

		return {
			"clustering_coefficient": dict(zip([self.to_label(n) for n in selected_nodes],clustering_coefficient)),
			"excess_closure": dict(zip([self.to_label(n) for n in selected_nodes],excess_closure))
		}

