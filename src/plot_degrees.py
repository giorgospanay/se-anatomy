import os, glob, parse, pickle
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## GLOBALS
csv_path="../results2"
log_path="../result_logs"
obj_path=csv_path



# For (all) networks:
# 	(a) check degrees on each layer
# 	(b) check for each layer how many nodes have deg=0
# 	(c) flatten all layers and check overall degrees
# 	(d) see on how many of the layers deg<>0 for every node


degs_flat=None
hist_flat=None
degs_work=None
hist_work=None
degs_edu=None
hist_edu=None
degs_nbr=None
hist_nbr=None
degs_fam=None
hist_fam=None

# Set subplot figures (1A, 1C)
fig1, (ax1a, ax1c) = plt.subplots(1,2)

# # Get hists. When all is available, uncomment:
# with open(f"{log_path}/histogram_fam_all.txt","r") as h_wf:
# 	hist_fam = [line.rstrip() for line in h_wf]
# with open(f"{log_path}/histogram_edu_all.txt","r") as h_wf:
# 	hist_edu = [line.rstrip() for line in h_wf]
# with open(f"{log_path}/histogram_nbr_all.txt","r") as h_wf:
# 	hist_nbr= [line.rstrip() for line in h_wf]
# with open(f"{log_path}/histogram_work_all.txt","r") as h_wf:
# 	hist_work = [line.rstrip() for line in h_wf]
# with open(f"{log_path}/histogram_flat_all.txt","r") as h_wf:
# 	hist_flat = [line.rstrip() for line in h_wf]


# Get hists. Now showing year=2017:
with open(f"{log_path}/histogram_family2017.txt","r") as h_wf:
	hist_fam = [line.rstrip() for line in h_wf]
with open(f"{log_path}/histogram_education2017.txt","r") as h_wf:
	hist_edu = [line.rstrip() for line in h_wf]
with open(f"{log_path}/histogram_neighbourhood2017.txt","r") as h_wf:
	hist_nbr= [line.rstrip() for line in h_wf]
with open(f"{log_path}/histogram_work2017.txt","r") as h_wf:
	hist_work = [line.rstrip() for line in h_wf]
with open(f"{log_path}/histogram_flat2017.txt","r") as h_wf:
	hist_flat = [line.rstrip() for line in h_wf]

# Fig. 1A: Plot each histogram (individual layers) as line
ax1a.legend(["Family","Education","Neighbourhood","Work"],loc="upper center")
ax1a.set_ylabel("Frequency")
ax1a.set_xlabel("Degree")
ax1a.set_yscale("log")
ax1a.set_xscale("log")

ax1a.plot(hist_fam,color="tab:blue",marker=".",linestyle="dashdot")
ax1a.plot(hist_edu,color="tab:orange",marker=".",linestyle="dashdot")
ax1a.plot(hist_nbr,color="tab:green",marker=".",linestyle="dashdot")
ax1a.plot(hist_work,color="tab:red",marker=".",linestyle="dashdot")

# Fig. 1C: Plot histogram (flattened opp. network) as line
ax1c.legend(["Total degree"],loc="upper center")
ax1c.set_ylabel("Frequency")
ax1c.set_xlabel("Degree")
ax1c.set_yscale("log")
ax1c.set_xscale("log")

ax1c.plot(hist_flat,color="black",marker=".",linestyle="dashdot")


plt.show()



# Get degrees if necessary	

# with open(f"{log_path}/degrees_flat_all.txt","r") as d_wf:
# 	degs_flat = [line.rstrip() for line in d_wf]




# (A) Plot histogram per layer



# (B) Plot 