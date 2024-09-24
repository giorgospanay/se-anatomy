import os, sys, glob, parse, pickle, gc
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

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

node_df=None

# If mode=calc: make df
if mode=="calc":
	print("Loading all degree files into pandas")
	# # Load degree files into dataframe
	fam_df=None
	edu_df=None
	nbr_df=None
	work_df=None

	with open(f"{log_path}/degrees_family2017.txt","r") as h_wf:
		fam_df = pd.DataFrame(
			[ast.literal_eval(line.rstrip()) for line in h_wf],
			columns=["PersonNr","deg_fam"]
		)
		fam_df.set_index("PersonNr",inplace=True)
	with open(f"{log_path}/degrees_education2017.txt","r") as h_wf:
		edu_df = pd.DataFrame(
			[ast.literal_eval(line.rstrip()) for line in h_wf],
			columns=["PersonNr","deg_edu"]
		)
		edu_df.set_index("PersonNr",inplace=True)
	with open(f"{log_path}/degrees_neighbourhood2017.txt","r") as h_wf:
		nbr_df = pd.DataFrame(
			[ast.literal_eval(line.rstrip()) for line in h_wf],
			columns=["PersonNr","deg_nbr"]
		)
		nbr_df.set_index("PersonNr",inplace=True)
	with open(f"{log_path}/degrees_work2017.txt","r") as h_wf:
		work_df = pd.DataFrame(
			[ast.literal_eval(line.rstrip()) for line in h_wf],
			columns=["PersonNr","deg_work"]
		)
		work_df.set_index("PersonNr",inplace=True)
	# Concat all on node_df
	node_df=pd.concat([fam_df,edu_df,nbr_df,work_df],axis=1,join="outer",copy=False)
	node_df.fillna(0.0,inplace=True)

	# Save to csv for comparison
	node_df.to_csv(f"{log_path}/node_2017.csv")

# If no mode set, read file from csv
else:
	node_df=pd.read_csv(f"{log_path}/node_2017.csv",index_col="PersonNr",header=0)


#### Uncomment for disconnected nodes. Curr no node is disconnected in the network
# ---------------------------------------------------------------------------

# Fig. 1B: Plot disconnected nodes in each layer
print("Figure 1B")
fig1b, ax1b = plt.subplots()
zero_fam=(node_df["deg_fam"]==0).sum()
zero_edu=(node_df["deg_edu"]==0).sum()
zero_nbr=(node_df["deg_nbr"]==0).sum()
zero_work=(node_df["deg_work"]==0).sum()

N, bins, patches = ax1b.hist([zero_fam,zero_edu,zero_nbr,zero_work])

patches[0].set_facecolor("tab:blue")
patches[1].set_facecolor("tab:orange")
patches[2].set_facecolor("tab:green")
patches[3].set_facecolor("tab:red")


ax1b.set_xlabel("No connections")
ax1b.set_yticks([0,2000000,4000000,6000000],labels=["0","2M","4M","6M"])
ax1b.tick_params(axis="x",labelbottom=False)

# Save
fig1b.savefig(f"{plot_path}/fig1b.png",bbox_inches='tight',dpi=300)


# ---------------------------------------------------------------------------

# Fig. 1D: Plot #layers for which a node is disconnected
print("Figure 1D")
fig1d, ax1d = plt.subplots()
node_df["nz_layers"]=np.count_nonzero(node_df==0,axis=1)
#node_df.sort_values("nz_layers",inplace=True)

ax1d.hist(node_df["nz_layers"],color="black")

ax1d.set_yticks([0,2000000,4000000,6000000],labels=["0","2M","4M","6M"])
ax1d.set_xticks([0,2,4])
#ax1d.tick_params(axis="x",labelbottom=False)

# Save
fig1d.savefig(f"{plot_path}/fig1d.png",bbox_inches='tight',dpi=300)

# ---------------------------------------------------------------------------

# Fig. 2A: Cumulative inverse degree distribution. Plot as line histograms
print("Figure 2A")
fig2a, ax2a = plt.subplots()

hist_fam.reverse()
deg_fam=list(reversed(range(len(hist_fam))))
cs_fam=np.cumsum(hist_fam)

hist_edu.reverse()
deg_edu=list(reversed(range(len(hist_edu))))
cs_edu=np.cumsum(hist_edu)

hist_nbr.reverse()
deg_nbr=list(reversed(range(len(hist_nbr))))
cs_nbr=np.cumsum(hist_nbr)

hist_work.reverse()
deg_work=list(reversed(range(len(hist_work))))
cs_work=np.cumsum(hist_work)

ax2a.plot(deg_fam,cs_fam,color="tab:blue",marker=",",linestyle="dashdot")
ax2a.plot(deg_edu,cs_edu,color="tab:orange",marker=",",linestyle="dashdot")
ax2a.plot(deg_nbr,cs_nbr,color="tab:green",marker=",",linestyle="dashdot")
ax2a.plot(deg_work,cs_work,color="tab:red",marker=",",linestyle="dashdot")

ax2a.set_xlabel("Degree")
ax2a.set_ylabel("Sample with k > Degree")
ax2a.set_yscale("log")
ax2a.set_xscale("log") 


# Save
fig2a.legend(labels=["Family","Education","Neighbourhood","Work"],loc="upper center",alignment="center",ncols=2)
fig2a.savefig(f"{plot_path}/fig2a.png",bbox_inches='tight',dpi=300)

# ---------------------------------------------------------------------------

# Fig. 2B: Inverse cumulative degree distribution on flat
print("Figure 2B")
fig2b, ax2b = plt.subplots()

hist_flat.reverse()
deg_flat=list(reversed(range(len(hist_flat))))
cs_flat=np.cumsum(hist_flat)

ax2b.plot(deg_flat,cs_flat,color="black",marker=",",linestyle="dashdot")

ax2b.set_xlabel("Degree")
ax2b.set_yscale("log")
ax2b.set_xscale("log") 


fig2b.legend(labels=["Total degree"],loc="upper center",alignment="center",ncols=2)
fig2b.savefig(f"{plot_path}/fig2b.png",bbox_inches='tight',dpi=300)


# --------------------------------------------------------------------------

# Table 1A: Pearson correlation between degree in layers
print("Table 1A")
table_1a=node_df.corr(method="pearson")
# Save correlation table to csv
table_1a.to_csv(f"{plot_path}/table_1a.csv")

# --------------------------------------------------------------------------

# Table 1B: Layer overlap percentage
print("Table 1B")
# Load each layer. Calculate intersection of each pair of distinct layers
#   as percentage:: L1.intersect(L2).edges / L1.edges.

inter_fam=[]
inter_edu=[]
inter_nbr=[]
inter_work=[]

############## @TODO: do intersection on dfs

# Family 2017:
df=read_in_network(pd.read_csv(f"{csv_path}/final_network2017.csv"),"PersonNr")
fam_df = make_entire_edge_list(df)[["PersonNr","PersonNr2"]]
df=None
gc.collect()
# Education 2017:
edu_df=pd.read_csv(f"{csv_path}/education2017.csv")
# Neighbourhood 2017:
nbr_df=pd.read_csv(f"{csv_path}/neighbourhood2017.csv")
# Work 2017:
work_df=pd.read_csv(f"{csv_path}/work2017.csv")


# Intersection Family / Family
inter_fam.append(1.0)
# Intersection Family / Education
inter_fe=pd.merge(fam_df,edu_df,how="inner",on=["PersonNr","PersonNr2"])
inter_fam.append(len(inter_fe.index)/len(fam_df.index))
inter_edu.append(len(inter_fe.index)/len(edu_df.index))
# Intersection Family / Neighbourhood
inter_fn=pd.merge(fam_df,nbr_df,how="inner",on=["PersonNr","PersonNr2"])
inter_fam.append(len(inter_fn.index)/len(fam_df.index))
inter_nbr.append(len(inter_fn.index)/len(nbr_df.index))
# Intersection Family / Work
inter_fw=pd.merge(fam_df,work_df,how="inner",on=["PersonNr","PersonNr2"])
inter_fam.append(len(inter_fw.index)/len(fam_df.index))
inter_work.append(len(inter_fw.index)/len(work_df.index))

fam_df=None
inter_fe=None
inter_fn=None
inter_fw=None
gc.collect()

# Intersection Education / Education
inter_edu.append(1.0)
# Intersection Education / Neighbourhood
inter_en=pd.merge(edu_df,nbr_df,how="inner",on=["PersonNr","PersonNr2"])
inter_edu.append(len(inter_en.index)/len(edu_df.index))
inter_nbr.append(len(inter_en.index)/len(nbr_df.index))
# Intersection Education / Work
inter_ew=pd.merge(edu_df,work_df,how="inner",on=["PersonNr","PersonNr2"])
inter_edu.append(len(inter_ew.index)/len(edu_df.index))
inter_work.append(len(inter_ew.index)/len(work_df.index))

edu_df=None
inter_en=None
inter_ew=None
gc.collect()

# Intersection Neighbourhood / Neighbourhood
inter_nbr.append(1.0)
# Intersection Neighbourhood / Work
inter_nw=pd.merge(nbr_df,work_df,how="inner",on=["PersonNr","PersonNr2"])
inter_nbr.append(len(inter_nw.index)/len(nbr_df.index))
inter_work.append(len(inter_nw.index)/len(work_df.index))

nbr_df=None
inter_nw=None
gc.collect()

# Intersection Work / Work
inter_work.append(1.0)

work_df=None
gc.collect()

# Create dataframe
table_1b=pd.DataFrame(columns=["F","E","N","W"])
f_df=pd.DataFrame(inter_fam)
table_1b=pd.concat([table_1b,f_df],axis=0,ignore_index=True)
e_df=pd.DataFrame(inter_edu)
table_1b=pd.concat([table_1b,e_df],axis=0,ignore_index=True)
n_df=pd.DataFrame(inter_nbr)
table_1b=pd.concat([table_1b,n_df],axis=0,ignore_index=True)
w_df=pd.DataFrame(inter_work)
table_1b=pd.concat([table_1b,w_df],axis=0,ignore_index=True)


# Save dataframe
table_1b.to_csv(f"{plot_path}/table_1b.csv")

