import os, glob, parse, pickle
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# For (all) networks:
# 	(a) check degrees on each layer
# 	(b) check for each layer how many nodes have deg=0
# 	(c) flatten all layers and check overall degrees
# 	(d) see on how many of the layers deg<>0 for every node


degs_flat=None, hist_flat=None, degs_work=None, hist_work=None, degs_edu=None, hist_edu=None, degs_nbr=None, hist_nbr=None, degs_fam=None, hist_fam=None


with open(f"{log_path}/degrees_flat_all.txt","r") as d_wf:
	

with open(f"{log_path}/histogram_flat_all.txt","r") as h_wf:



# (A) Plot histogram per layer



# (B) Plot 