import os, glob, parse, pickle
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## GLOBALS
csv_path="../results2"
log_path="../result_logs"
plot_path="../result_plots"
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

print(hist_fam)
print(hist_edu)
print(hist_nbr)
print(hist_work)
print(hist_flat)

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

# Save figures 1A & 1C
fig1.savefig(f"{plot_path}/fig1.png",bbox_inches='tight',dpi=300, transparent=True)


# ---------------------------------------------------------------------------

# # Load degree files

# with open(f"{log_path}/degrees_family2017.txt","r") as h_wf:
# 	degs_fam = [line.rstrip() for line in h_wf]
# with open(f"{log_path}/degrees_education2017.txt","r") as h_wf:
# 	degs_edu = [line.rstrip() for line in h_wf]
# with open(f"{log_path}/degrees_neighbourhood2017.txt","r") as h_wf:
# 	degs_nbr= [line.rstrip() for line in h_wf]
# with open(f"{log_path}/degrees_work2017.txt","r") as h_wf:
# 	degs_work = [line.rstrip() for line in h_wf]
# with open(f"{log_path}/degrees_flat2017.txt","r") as h_wf:
# 	degs_flat = [line.rstrip() for line in h_wf]



# # Fig. 1B: Plot disconnected nodes in each layer
# zero_fam=0
# zero_edu=0
# zero_nbr=0
# zero_work=0

# for ln in degs_fam:
# 	toks=ln.split(" ")
# 	if int(toks[len(toks)-1])!=0: continue
# 	zero_fam=len(toks)-1
# for ln in degs_edu:
# 	toks=ln.split(" ")
# 	if int(toks[len(toks)-1])!=0: continue
# 	zero_edu=len(toks)-1
# for ln in degs_nbr:
# 	toks=ln.split(" ")
# 	if int(toks[len(toks)-1])!=0: continue
# 	zero_nbr=len(toks)-1
# for ln in degs_work:
# 	toks=ln.split(" ")
# 	if int(toks[len(toks)-1])!=0: continue
# 	zero_work=len(toks)-1





# # Fig. 1D: Plot # layers in which a node has non-zero degree
# node_dict=dict()

# for ln in degs_fam:
# 	toks=ln.split(" ")
# 	if int(toks[len(toks)-1])!=0: continue
# 	zero_fam=len(toks)-1
# for ln in degs_edu:
# 	toks=ln.split(" ")
# 	if int(toks[len(toks)-1])!=0: continue
# 	zero_edu=len(toks)-1
# for ln in degs_nbr:
# 	toks=ln.split(" ")
# 	if int(toks[len(toks)-1])!=0: continue
# 	zero_nbr=len(toks)-1
# for ln in degs_work:
# 	toks=ln.split(" ")
# 	if int(toks[len(toks)-1])!=0: continue
# 	zero_work=len(toks)-1


