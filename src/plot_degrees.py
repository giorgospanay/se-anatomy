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


# Get hists
with open(f"{log_path}/histogram_fam_all.txt","r") as h_wf:
	hist_fam = [line.rstrip() for line in h_wf]
with open(f"{log_path}/histogram_edu_all.txt","r") as h_wf:
	hist_edu = [line.rstrip() for line in h_wf]
with open(f"{log_path}/histogram_nbr_all.txt","r") as h_wf:
	hist_nbr= [line.rstrip() for line in h_wf]
with open(f"{log_path}/histogram_work_all.txt","r") as h_wf:
	hist_work = [line.rstrip() for line in h_wf]

with open(f"{log_path}/histogram_flat_all.txt","r") as h_wf:
	hist_flat = [line.rstrip() for line in h_wf]





# Get degrees if necessary	

with open(f"{log_path}/degrees_flat_all.txt","r") as d_wf:
	degs_flat = [line.rstrip() for line in d_wf]




# (A) Plot histogram per layer



# (B) Plot 