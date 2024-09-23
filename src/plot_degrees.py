import os, glob, parse, pickle
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast


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

# Fig 1A: Plot degree histogram per layer
print("Figure 1A")
fig1a, ax1a = plt.subplots()

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

hist_fam=ast.literal_eval(hist_fam[0])
hist_edu=ast.literal_eval(hist_edu[0])
hist_nbr=ast.literal_eval(hist_nbr[0])
hist_work=ast.literal_eval(hist_work[0])
hist_flat=ast.literal_eval(hist_flat[0])

# Fig. 1A: Plot each histogram (individual layers) as line
ax1a.set_ylabel("Frequency")
ax1a.set_xlabel("Degree")
ax1a.set_yscale("log")
ax1a.set_xscale("log")
ax1a.set_xticks([1,10,100,1000],labels=["1","10","100","1K"])
ax1a.set_yticks([1,10,100,1000,10000,100000,1000000],labels=["1","10","100","1K","10K","100K","1M"])

ax1a.plot(hist_fam,color="tab:blue",marker=",",linestyle="dashdot")
ax1a.plot(hist_edu,color="tab:orange",marker=",",linestyle="dashdot")
ax1a.plot(hist_nbr,color="tab:green",marker=",",linestyle="dashdot")
ax1a.plot(hist_work,color="tab:red",marker=",",linestyle="dashdot")

# Save
fig1a.legend(labels=["Family","Education","Neighbourhood","Work"],loc="upper center",alignment="center",ncols=2)
fig1a.savefig(f"{plot_path}/fig1a.png",bbox_inches='tight',dpi=300)

# ---------------------------------------------------------------------------

# Fig. 1C: Plot histogram (flattened opp. network) as line
print("Figure 1C")
fig1c, ax1c = plt.subplots()

#ax1c.set_ylabel("Frequency")
ax1c.set_xlabel("Degree")
ax1c.set_yscale("log")
ax1c.set_xscale("log")
ax1c.set_xticks([1,10,100,1000],labels=["1","10","100","1K"])
ax1c.set_yticks([1,10,100,1000,10000,100000,1000000],labels=["1","10","100","1K","10K","100K","1M"])

ax1c.plot(hist_flat,color="black",marker=",",linestyle="dashdot")

# Save
fig1c.legend(labels=["Total degree"],loc="upper center",alignment="center",ncols=2)
fig1c.savefig(f"{plot_path}/fig1c.png",bbox_inches='tight',dpi=300)

# ---------------------------------------------------------------------------

#### Uncomment for disconnected nodes. Curr no node is disconnected in the network

# print("Loading all degree files")
# # # Load degree files into dataframe
# node_df=None
# fam_df=None
# edu_df=None
# nbr_df=None
# work_df=None


# with open(f"{log_path}/degrees_family2017.txt","r") as h_wf:
# 	fam_df = pd.DataFrame(
# 		[ast.literal_eval(line.rstrip()) for line in h_wf],
# 		columns=["PersonNr","deg_fam"]
# 	)
# 	fam_df.set_index("PersonNr")
# with open(f"{log_path}/degrees_education2017.txt","r") as h_wf:
# 	edu_df = pd.DataFrame(
# 		[ast.literal_eval(line.rstrip()) for line in h_wf],
# 		columns=["PersonNr","deg_edu"]
# 	)
# 	edu_df.set_index("PersonNr")
# with open(f"{log_path}/degrees_neighbourhood2017.txt","r") as h_wf:
# 	nbr_df = pd.DataFrame(
# 		[ast.literal_eval(line.rstrip()) for line in h_wf],
# 		columns=["PersonNr","deg_nbr"]
# 	)
# 	nbr_df.set_index("PersonNr")
# with open(f"{log_path}/degrees_work2017.txt","r") as h_wf:
# 	work_df = pd.DataFrame(
# 		[ast.literal_eval(line.rstrip()) for line in h_wf],
# 		columns=["PersonNr","deg_work"]
# 	)
# 	work_df.set_index("PersonNr")
# # Concat all on node_df
# node_df=pd.concat([fam_df,edu_df,nbr_df,work_df],axis=1,join="outer",copy=False)
# node_df.fillna(0)


# # ---------------------------------------------------------------------------

# # Fig. 1B: Plot disconnected nodes in each layer
# print("Figure 1B")
# fig1b, ax1b = plt.subplots()
# zero_fam=(node_df["deg_fam"]==0).sum()
# zero_edu=(node_df["deg_edu"]==0).sum()
# zero_nbr=(node_df["deg_nbr"]==0).sum()
# zero_work=(node_df["deg_work"]==0).sum()

# N, bins, patches = ax1b.hist([zero_fam,zero_edu,zero_nbr,zero_work])

# patches[0].set_facecolor("tab:blue")
# patches[1].set_facecolor("tab:orange")
# patches[2].set_facecolor("tab:green")
# patches[3].set_facecolor("tab:red")


# ax1b.set_xlabel("No connections")
# ax1b.set_yticks([0,2000000,4000000,6000000],labels=["0","2M","4M","6M"])
# ax1b.tick_params(axis="x",labelbottom=False)

# # Save
# fig1b.savefig(f"{plot_path}/fig1b.png",bbox_inches='tight',dpi=300)


# # ---------------------------------------------------------------------------

# # Fig. 1D: Plot #layers for which a node is disconnected
# print("Figure 1D")
# fig1d, ax1d = plt.subplots()
# node_df["nz_layers"]=np.count_nonzero(node_df==0,axis=1)
# #node_df.sort_values("nz_layers",inplace=True)

# ax1d.hist(node_df["nz_layers"],color="black")

# ax1d.set_yticks([0,2000000,4000000,6000000],labels=["0","2M","4M","6M"])
# ax1d.set_xticks([0,2,4])
# #ax1d.tick_params(axis="x",labelbottom=False)

# # Save
# fig1d.savefig(f"{plot_path}/fig1d.png",bbox_inches='tight',dpi=300)

# # ---------------------------------------------------------------------------

# Fig. 2A: Cumulative inverse degree distribution. Plot as line histograms
print("Figure 2A")
fig2a, ax2a = plt.subplots()

cnt_fam=hist_fam.reverse()
deg_fam=reversed(range(len(cnt_fam)))
cs_fam=np.cumsum(cnt_fam)
cnt_edu=hist_edu.reverse()
deg_edu=reversed(range(len(cnt_edu)))
cs_edu=np.cumsum(cnt_edu)
cnt_nbr=hist_nbr.reverse()
deg_nbr=reversed(range(len(cnt_nbr)))
cs_nbr=np.cumsum(cnt_nbr)
cnt_work=hist_work.reverse()
deg_work=reversed(range(len(cnt_work)))
cs_work=np.cumsum(cnt_work)

ax2a.plot(deg_fam,cs_fam,color="tab:blue",marker=",",linestyle="dashdot")
ax2a.plot(deg_edu,cs_edu,color="tab:orange",marker=",",linestyle="dashdot")
ax2a.plot(deg_nbr,cs_nbr,color="tab:green",marker=",",linestyle="dashdot")
ax2a.plot(deg_work,cs_work,color="tab:red",marker=",",linestyle="dashdot")

ax2a.set_xlabel("Degree")
ax2a.set_yscale("log")
ax2a.set_xscale("log") 


# Save
fig2a.legend(labels=["Family","Education","Neighbourhood","Work"],loc="upper center",alignment="center",ncols=2)
fig2a.savefig(f"{plot_path}/fig2a.png",bbox_inches='tight',dpi=300)

# ---------------------------------------------------------------------------

# Fig. 2B: Inverse cumulative degree distribution on flat
print("Figure 2B")
fig2b, ax2b = plt.subplots()

cnt_flat=hist_flat.reverse()
deg_flat=reversed(range(len(cnt_flat)))
cs_flat=np.cumsum(cnt_flat)

ax2b.plot(deg_flat,cs_flat,color="black",marker=",",linestyle="dashdot")

ax2b.set_xlabel("Degree")
ax2b.set_yscale("log")
ax2b.set_xscale("log") 


fig2b.legend(labels=["Total degree"],loc="upper center",alignment="center",ncols=2)
fig2b.savefig(f"{plot_path}/fig2b.png",bbox_inches='tight',dpi=300)

