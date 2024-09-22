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

# Set subplot figures (1A, 1C)
fig1a, ax1a = plt.subplots()

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



# Fig. 1C: Plot histogram (flattened opp. network) as line
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

# # Load degree files into dataframe
node_df=None
fam_df=None
edu_df=None
nbr_df=None
work_df=None


with open(f"{log_path}/degrees_family2017.txt","r") as h_wf:
	fam_df = pd.DataFrame(
		[ast.literal_eval(line.rstrip()) for line in h_wf],
		columns=["PersonNr","deg_fam"]
	)
	fam_df.set_index("PersonNr")
with open(f"{log_path}/degrees_education2017.txt","r") as h_wf:
	edu_df = pd.DataFrame(
		[ast.literal_eval(line.rstrip()) for line in h_wf],
		columns=["PersonNr","deg_edu"]
	)
	edu_df.set_index("PersonNr")
	# node_df.join(pd.DataFrame(
	# 		[ast.literal_eval(line.rstrip()) for line in h_wf],
	# 		columns=["PersonNr","deg_edu"]
	# 	), on="PersonNr", how="outer"
	# )
with open(f"{log_path}/degrees_neighbourhood2017.txt","r") as h_wf:
	nbr_df = pd.DataFrame(
		[ast.literal_eval(line.rstrip()) for line in h_wf],
		columns=["PersonNr","deg_nbr"]
	)
	nbr_df.set_index("PersonNr")
	# node_df.join(pd.DataFrame(
	# 		[ast.literal_eval(line.rstrip()) for line in h_wf],
	# 		columns=["PersonNr","deg_nbr"]
	# 	), on="PersonNr", how="outer"
	# )
with open(f"{log_path}/degrees_work2017.txt","r") as h_wf:
	work_df = pd.DataFrame(
		[ast.literal_eval(line.rstrip()) for line in h_wf],
		columns=["PersonNr","deg_work"]
	)
	work_df.set_index("PersonNr")
	# node_df.join(pd.DataFrame(
	# 		[ast.literal_eval(line.rstrip()) for line in h_wf],
	# 		columns=["PersonNr","deg_work"]
	# 	), on="PersonNr", how="outer"
	# )
# Concat all on node_df
node_df=pd.concat([fam_df,edu_df,nbr_df,work_df],axis=1,join="outer",copy=False)
node_df.fillna(0)

# ---------------------------------------------------------------------------

# Fig. 1B: Plot disconnected nodes in each layer
fig1b, ax1b = plt.subplots()
zero_fam=node_df[node_df["deg_fam"]==0].count()
zero_edu=node_df[node_df["deg_edu"]==0].count()
zero_nbr=node_df[node_df["deg_nbr"]==0].count()
zero_work=node_df[node_df["deg_work"]==0].count()

ax1b.hist([zero_fam,zero_edu,zero_nbr,zero_work],color=["tab:blue","tab:orange","tab:green","tab:red"])

ax1b.set_xlabel("No connections")
ax1b.set_yticks([0,2000000,4000000,6000000],labels=["0","2M","4M","6M"])
ax1b.tick_params(axis="x",labelbottom=False)

# Save
fig1b.savefig(f"{plot_path}/fig1b.png",bbox_inches='tight',dpi=300)


# ---------------------------------------------------------------------------

# Fig. 1D: Plot #layers for which a node is disconnected
fig1d, ax1d = plt.subplots()
node_df["nz_layers"]=np.count_nonzero(node_df,axis=1)
#node_df.sort_values("nz_layers",inplace=True)

ax1d.hist(node_df["nz_layers"],color="black")

ax1d.set_yticks([0,2000000,4000000,6000000],labels=["0","2M","4M","6M"])
ax1d.set_xticks([0,2,4])
#ax1d.tick_params(axis="x",labelbottom=False)

# Save
fig1d.savefig(f"{plot_path}/fig1d.png",bbox_inches='tight',dpi=300)

# ---------------------------------------------------------------------------

# Fig. 2A: Cumulative inverse degree distribution. Plot as line histograms
fig2a, ax2a = plt.subplots()

ax2a.hist(node_df["deg_fam"],cumulative=True,color="tab:blue",log=True,histtype="step")
ax2a.hist(node_df["deg_edu"],cumulative=True,color="tab:orange",log=True,histtype="step")
ax2a.hist(node_df["deg_nbr"],cumulative=True,color="tab:green",log=True,histtype="step")
ax2a.hist(node_df["deg_work"],cumulative=True,color="tab:red",log=True,histtype="step")

# Save
fig2a.legend(labels=["Family","Education","Neighbourhood","Work"],loc="upper center",alignment="center",ncols=2)
fig2a.savefig(f"{plot_path}/fig2a.png",bbox_inches='tight',dpi=300)

# ---------------------------------------------------------------------------

# Fig. 2B: Inverse cumulative degree distribution on flat
fig2b, ax2b = plt.subplots()
flat_df=None
with open(f"{log_path}/degrees_flat2017.txt","r") as h_wf:
	flat_df=pd.DataFrame(
		[ast.literal_eval(line.rstrip()) for line in h_wf],
		columns=["PersonNr","deg_flat"]
	)

ax2b.hist(flat_df["deg_flat"],cumulative=True,color="black",log=True,histtype="step")

fig2b.legend(labels=["Total degree"],loc="upper center",alignment="center",ncols=2)
fig2b.savefig(f"{plot_path}/fig2b.png",bbox_inches='tight',dpi=300)

