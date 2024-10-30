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
degs_house=None
hist_house=None
degs_ext=None
hist_ext=None
degs_close=None
hist_close=None


print("Figure 1")

fig1, (ax1a,ax1c) = plt.subplots(nrows=1,ncols=2)

# Fig 1A: Plot degree histogram per layer
# Get hists. Now showing year=2017:
with open(f"{log_path}/histogram_close_family2017.txt","r") as h_wf:
	hist_close = [line.rstrip() for line in h_wf]
with open(f"{log_path}/histogram_extended_family2017.txt","r") as h_wf:
	hist_ext = [line.rstrip() for line in h_wf]
with open(f"{log_path}/histogram_household2017.txt","r") as h_wf:
	hist_house = [line.rstrip() for line in h_wf]
with open(f"{log_path}/histogram_education2017.txt","r") as h_wf:
	hist_edu = [line.rstrip() for line in h_wf]
with open(f"{log_path}/histogram_neighbourhood2017.txt","r") as h_wf:
	hist_nbr= [line.rstrip() for line in h_wf]
with open(f"{log_path}/histogram_work2017.txt","r") as h_wf:
	hist_work = [line.rstrip() for line in h_wf]
with open(f"{log_path}/histogram_flat2017.txt","r") as h_wf:
	hist_flat = [line.rstrip() for line in h_wf]

hist_close=ast.literal_eval(hist_close[0])
hist_ext=ast.literal_eval(hist_ext[0])
hist_house=ast.literal_eval(hist_house[0])
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

ax1a.plot(hist_close,color="darkslategrey",marker=".",linestyle="dashdot")
ax1a.plot(hist_ext,color="steelblue",marker=".",linestyle="dashdot")
ax1a.plot(hist_house,color="crimson",marker=".",linestyle="dashdot")
ax1a.plot(hist_edu,color="teal",marker=".",linestyle="dashdot")
ax1a.plot(hist_nbr,color="gold",marker=".",linestyle="dashdot")
ax1a.plot(hist_work,color="grey",marker=".",linestyle="dashdot")

# Save
ax1a.legend(labels=["Close family","Extended family","Household","School","Neighbors","Work"],loc="upper center",alignment="center",ncols=3,bbox_to_anchor=(-0.25,1.05,1,0.2),mode="expand")
#fig1a.savefig(f"{plot_path}/fig1a.png",bbox_inches='tight',dpi=300)

# ---------------------------------------------------------------------------

node_df=None

# If mode=calc: make df
if mode=="calc":
	print("Loading all degree files into pandas")
	# # Load degree files into dataframe
	close_df=None
	ext_df=None
	house_df=None
	edu_df=None
	nbr_df=None
	work_df=None


	with open(f"{log_path}/degrees_close_family2017.txt","r") as h_wf:
		close_df = pd.DataFrame(
			[ast.literal_eval(line.rstrip()) for line in h_wf],
			columns=["PersonNr","deg_close"]
		)
		close_df.set_index("PersonNr",inplace=True)
	with open(f"{log_path}/degrees_extended_family2017.txt","r") as h_wf:
		ext_df = pd.DataFrame(
			[ast.literal_eval(line.rstrip()) for line in h_wf],
			columns=["PersonNr","deg_ext"]
		)
		ext_df.set_index("PersonNr",inplace=True)
	with open(f"{log_path}/degrees_household2017.txt","r") as h_wf:
		house_df = pd.DataFrame(
			[ast.literal_eval(line.rstrip()) for line in h_wf],
			columns=["PersonNr","deg_house"]
		)
		house_df.set_index("PersonNr",inplace=True)
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
	node_df=pd.concat([close_df,ext_df,house_df,edu_df,nbr_df,work_df],axis=1,join="outer",copy=False)
	node_df.fillna(0.0,inplace=True)

	# Add new line to calculate total degree for all nodes
	node_df["deg_total"]=node_df["deg_close"]+node_df["deg_ext"]+node_df["deg_house"]+node_df["deg_edu"]+node_df["deg_nbr"]+node_df["deg_work"]

	# Save to csv for comparison
	node_df.to_csv(f"{log_path}/node_a_2017.csv")

# If no mode set, read file from csv
else:
	node_df=pd.read_csv(f"{log_path}/node_a_2017.csv",index_col="PersonNr",header=0)
	if "deg_total" not in node_df:
		# Add new line to calculate total degree for all nodes
		node_df["deg_total"]=node_df["deg_close"]+node_df["deg_ext"]+node_df["deg_house"]+node_df["deg_edu"]+node_df["deg_nbr"]+node_df["deg_work"]
		node_df.to_csv(f"{log_path}/node_a_2017.csv")


# ---------------------------------------------------------------------------

# Fig. 1C: Plot histogram (flattened opp. network) as line
#fig1c, ax1c = plt.subplots()

# Calculate histogram from deg_total
hist_total=node_df["deg_total"].value_counts().sort_index()

# Plot deg_total histogram
ax1c.plot(hist_total,color="black",marker=".",linestyle="dashdot")
# # Also plot deg_flat histogram
# ax1c.plot(hist_flat,color="grey",marker=",",linestyle="dashdot")

#ax1c.set_ylabel("Frequency")
ax1c.set_xlabel("Degree")
ax1c.set_yscale("log")
ax1c.set_xscale("log")
ax1c.set_xticks([1,10,100,1000],labels=["1","10","100","1K"])
ax1c.set_yticks([1,10,100,1000,10000,100000,1000000],labels=["1","10","100","1K","10K","100K","1M"])

# Save
ax1c.legend(labels=["Total degree"],loc="upper center",alignment="center",ncols=1,bbox_to_anchor=(0,1.05,1,0.2),mode="expand")
#fig1c.legend(labels=["Total degree","Total degree (flat)"],loc="upper center",alignment="center",ncols=2)
#fig1c.savefig(f"{plot_path}/fig1c.png",bbox_inches='tight',dpi=300)


# ---------------------------------------------------------------------------

# Fig. 1B: Plot disconnected nodes in each layer
#fig1b, ax1b = plt.subplots()

# Set inset axes
l,b,w,h=[0.35,0.175,0.25,0.175]
ax1b=ax1a.inset_axes([l,b,w,h])

zero_close=(node_df["deg_close"]==0.0).sum()
zero_ext=(node_df["deg_ext"]==0.0).sum()
zero_house=(node_df["deg_house"]==0.0).sum()
zero_edu=(node_df["deg_edu"]==0.0).sum()
zero_nbr=(node_df["deg_nbr"]==0.0).sum()
zero_work=(node_df["deg_work"]==0.0).sum()

# # Debug
# print(f"f:{zero_fam}, e:{zero_edu}, n:{zero_nbr}, w:{zero_work}")

ax1b.bar(range(6),[zero_close,zero_ext,zero_house,zero_edu,zero_nbr,zero_work],color=["darkslategrey","steelblue","crimson","teal","gold","grey"])

ax1b.set_xlabel("No connections")
ax1b.set_yticks([0,2000000,4000000,6000000,8000000],labels=["0","2M","4M","6M","8M"])
ax1b.tick_params(axis="x",labelbottom=False)

# Save
#fig1b.savefig(f"{plot_path}/fig1b.png",bbox_inches='tight',dpi=300)


# ---------------------------------------------------------------------------

# Fig. 1D: Plot #layers for which a node is disconnected
#fig1d, ax1d = plt.subplots()

# Set inset axes
l,b,w,h=[0.35,0.175,0.25,0.175]
ax1d=ax1c.inset_axes([l,b,w,h])

node_df["nz_layers"]=6-np.count_nonzero(node_df[["deg_close","deg_ext","deg_house","deg_edu","deg_nbr","deg_work"]]==0.0,axis=1)

ax1d.hist(node_df["nz_layers"],color="black")

ax1d.set_yticks([0,2000000,4000000,6000000],labels=["0","2M","4M","6M"])
ax1d.set_xticks([0,3,6])
#ax1d.tick_params(axis="x",labelbottom=False)

# Save
#fig1d.savefig(f"{plot_path}/fig1d.png",bbox_inches='tight',dpi=300)

# Drop col here for rest
node_df.drop(labels=["nz_layers"],axis=1,inplace=True)


# Save entire figure here:
fig1.tight_layout()
fig1.savefig(f"{plot_path}/fig1.png",bbox_inches='tight',dpi=300)

# # ---------------------------------------------------------------------------

print("Figure 2")
fig2, (ax2a,ax2b) = plt.subplots(nrows=1,ncols=2)


# Fig. 2A: Cumulative inverse degree distribution. Plot as line histograms
hist_close.reverse()
deg_close=list(reversed(range(len(hist_close))))
cs_close=np.cumsum(hist_close)

hist_ext.reverse()
deg_ext=list(reversed(range(len(hist_ext))))
cs_ext=np.cumsum(hist_ext)

hist_house.reverse()
deg_house=list(reversed(range(len(hist_house))))
cs_house=np.cumsum(hist_house)

hist_edu.reverse()
deg_edu=list(reversed(range(len(hist_edu))))
cs_edu=np.cumsum(hist_edu)

hist_nbr.reverse()
deg_nbr=list(reversed(range(len(hist_nbr))))
cs_nbr=np.cumsum(hist_nbr)

hist_work.reverse()
deg_work=list(reversed(range(len(hist_work))))
cs_work=np.cumsum(hist_work)


ax2a.plot(deg_close,cs_close,color="darkslategrey",marker=".",linestyle="dashdot")
ax2a.plot(deg_ext,cs_ext,color="steelblue",marker=".",linestyle="dashdot")
ax2a.plot(deg_house,cs_house,color="crimson",marker=".",linestyle="dashdot")
ax2a.plot(deg_edu,cs_edu,color="teal",marker=".",linestyle="dashdot")
ax2a.plot(deg_nbr,cs_nbr,color="gold",marker=".",linestyle="dashdot")
ax2a.plot(deg_work,cs_work,color="grey",marker=".",linestyle="dashdot")

ax2a.set_xlabel("Degree")
ax2a.set_ylabel("Sample with k > Degree")
ax2a.set_yscale("log")
ax2a.set_xscale("log") 

ax2a.set_xticks([1,10,100,1000],labels=["1","10","100","1K"])
ax2a.set_yticks([1,10,100,1000,10000,100000,1000000,10000000],labels=["1","10","100","1K","10K","100K","1M","10M"])


ax2a.legend(labels=["Close family","Extended family","Household","School","Neighbors","Work"],loc="upper center",alignment="center",ncols=3,bbox_to_anchor=(-0.25,1.05,1,0.2),mode="expand")

#fig2a.savefig(f"{plot_path}/fig2a.png",bbox_inches='tight',dpi=300)

# ---------------------------------------------------------------------------

# Fig. 2B: Inverse cumulative degree distribution on flat

#fig2b, ax2b = plt.subplots()

hist_total.sort_index(ascending=False,inplace=True)
deg_total=list(reversed(range(len(hist_total))))
cs_total=np.cumsum(hist_total)

# hist_flat.reverse()
# deg_flat=list(reversed(range(len(hist_flat))))
# cs_flat=np.cumsum(hist_flat)

ax2b.plot(deg_total,cs_total,color="black",marker=".",linestyle="dashdot")
#ax2b.plot(deg_flat,cs_flat,color="gray",marker=",",linestyle="dashdot")

ax2b.set_xlabel("Degree")
ax2b.set_yscale("log")
ax2b.set_xscale("log")

ax2b.set_xticks([1,10,100,1000],labels=["1","10","100","1K"])
ax2b.set_yticks([1,10,100,1000,10000,100000,1000000,10000000],labels=["1","10","100","1K","10K","100K","1M","10M"])

ax2b.legend(labels=["Total degree"],loc="upper center",alignment="center",ncols=2,bbox_to_anchor=(0,1.05,1,0.2),mode="expand")
#fig2b.legend(labels=["Total degree","Total degree (flat)"],loc="upper center",alignment="center",ncols=2)

# Save
fig2.savefig(f"{plot_path}/fig2.png",bbox_inches='tight',dpi=300)


# # --------------------------------------------------------------------------

# # Table 1A: Pearson correlation between degree in layers
# print("Table 1A")
# table_1a=node_df[["deg_close","deg_ext","deg_house","deg_edu","deg_nbr","deg_work"]].corr(method="pearson")
# # Save correlation table to csv
# table_1a.to_csv(f"{plot_path}/table_1a.csv")

# # --------------------------------------------------------------------------

# # Table 1B: Layer overlap percentage
# print("Table 1B")
# # Load each layer. Calculate intersection of each pair of distinct layers
# #   as percentage:: L1.intersect(L2).edges / L1.edges.

# inter_close=[]
# inter_ext=[]
# inter_house=[]
# inter_edu=[]
# inter_nbr=[]
# inter_work=[]

# # Close family 2017:
# close_df=pd.read_csv(f"{csv_path}/close_family2017.csv")
# # Extended family 2017:
# ext_df=pd.read_csv(f"{csv_path}/extended_family2017.csv")
# # Household 2017:
# house_df=pd.read_csv(f"{csv_path}/household2017.csv")
# # Education 2017:
# edu_df=pd.read_csv(f"{csv_path}/education2017.csv")
# # Neighbourhood 2017:
# nbr_df=pd.read_csv(f"{csv_path}/neighbourhood2017.csv")
# # Work 2017:
# work_df=pd.read_csv(f"{csv_path}/work2017.csv")


# # Intersection C/C
# inter_close.append(1.0)
# # Intersection C/E
# inter_ce=pd.merge(close_df,ext_df,how="inner",on=["PersonNr","PersonNr2"])
# inter_close.append(len(inter_ce)/len(close_df))
# inter_ext.append(len(inter_ce)/len(ext_df))
# # Intersection C/H
# inter_ch=pd.merge(close_df,house_df,how="inner",on=["PersonNr","PersonNr2"])
# inter_close.append(len(inter_ch)/len(close_df))
# inter_house.append(len(inter_ch)/len(house_df))
# # Intersection C/N
# inter_cn=pd.merge(close_df,nbr_df,how="inner",on=["PersonNr","PersonNr2"])
# inter_close.append(len(inter_cn)/len(close_df))
# inter_nbr.append(len(inter_cn)/len(nbr_df))
# # Intersection C/S
# inter_cs=pd.merge(close_df,edu_df,how="inner",on=["PersonNr","PersonNr2"])
# inter_close.append(len(inter_cs)/len(close_df))
# inter_edu.append(len(inter_cs)/len(edu_df))
# # Intersection C/W
# inter_cw=pd.merge(close_df,work_df,how="inner",on=["PersonNr","PersonNr2"])
# inter_close.append(len(inter_cw)/len(close_df))
# inter_work.append(len(inter_cw)/len(work_df))

# # Garbage
# close_df=None
# inter_ce=None
# inter_ch=None
# inter_cs=None
# inter_cn=None
# inter_cw=None
# gc.collect()

# #############################################

# # Intersection E/E
# inter_ext.append(1.0)
# # Intersection E/H
# inter_eh=pd.merge(ext_df,house_df,how="inner",on=["PersonNr","PersonNr2"])
# inter_ext.append(len(inter_eh)/len(ext_df))
# inter_house.append(len(inter_eh)/len(house_df))
# # Intersection E/N
# inter_en=pd.merge(ext_df,nbr_df,how="inner",on=["PersonNr","PersonNr2"])
# inter_ext.append(len(inter_en)/len(ext_df))
# inter_nbr.append(len(inter_en)/len(nbr_df))
# # Intersection E/S
# inter_es=pd.merge(ext_df,edu_df,how="inner",on=["PersonNr","PersonNr2"])
# inter_ext.append(len(inter_es)/len(ext_df))
# inter_edu.append(len(inter_es)/len(edu_df))
# # Intersection E/W
# inter_ew=pd.merge(ext_df,work_df,how="inner",on=["PersonNr","PersonNr2"])
# inter_ext.append(len(inter_ew)/len(ext_df))
# inter_work.append(len(inter_ew)/len(work_df))

# # Garbage
# ext_df=None
# inter_eh=None
# inter_es=None
# inter_en=None
# inter_ew=None
# gc.collect()


# #############################################

# # Intersection H/H
# inter_house.append(1.0)
# # Intersection H/N
# inter_hn=pd.merge(house_df,nbr_df,how="inner",on=["PersonNr","PersonNr2"])
# inter_house.append(len(inter_hn)/len(house_df))
# inter_nbr.append(len(inter_hn)/len(nbr_df))
# # Intersection H/S
# inter_hs=pd.merge(house_df,edu_df,how="inner",on=["PersonNr","PersonNr2"])
# inter_house.append(len(inter_hs)/len(house_df))
# inter_edu.append(len(inter_hs)/len(edu_df))
# # Intersection H/W
# inter_hw=pd.merge(house_df,work_df,how="inner",on=["PersonNr","PersonNr2"])
# inter_house.append(len(inter_hw)/len(house_df))
# inter_work.append(len(inter_hw)/len(work_df))

# # Garbage
# house_df=None
# inter_hs=None
# inter_hn=None
# inter_hw=None
# gc.collect()


# #############################################

# # Intersection N/N
# inter_nbr.append(1.0)
# # Intersection N/S
# inter_ns=pd.merge(nbr_df,edu_df,how="inner",on=["PersonNr","PersonNr2"])
# inter_nbr.append(len(inter_ns)/len(nbr_df))
# inter_edu.append(len(inter_ns)/len(edu_df))
# # Intersection N/W
# inter_nw=pd.merge(nbr_df,work_df,how="inner",on=["PersonNr","PersonNr2"])
# inter_nbr.append(len(inter_nw)/len(nbr_df))
# inter_work.append(len(inter_nw)/len(work_df))

# # Garbage
# nbr_df=None
# inter_ns=None
# inter_nw=None
# gc.collect()


# #############################################

# # Intersection S/S
# inter_edu.append(1.0)
# # Intersection S/W
# inter_sw=pd.merge(edu_df,work_df,how="inner",on=["PersonNr","PersonNr2"])
# inter_edu.append(len(inter_sw)/len(edu_df))
# inter_work.append(len(inter_sw)/len(work_df))

# # Garbage
# edu_df=None
# inter_sw=None
# gc.collect()

# #############################################

# # Intersection Work / Work
# inter_work.append(1.0)

# work_df=None
# gc.collect()

# # Create dataframe
# #table_1b=pd.DataFrame(columns=["F","E","N","W"])
# c_df=pd.DataFrame({"C":[inter_close[0]],"E":[inter_close[1]],"H":[inter_close[2]],"N":[inter_close[3]],"S":[inter_close[4]],"W":[inter_close[5]]})
# e_df=pd.DataFrame({"C":[inter_ext[0]],"E":[inter_ext[1]],"H":[inter_ext[2]],"N":[inter_ext[3]],"S":[inter_ext[4]],"W":[inter_ext[5]]})
# h_df=pd.DataFrame({"C":[inter_house[0]],"E":[inter_house[1]],"H":[inter_house[2]],"N":[inter_house[3]],"S":[inter_house[4]],"W":[inter_house[5]]})
# n_df=pd.DataFrame({"C":[inter_nbr[0]],"E":[inter_nbr[1]],"H":[inter_nbr[2]],"N":[inter_nbr[3]],"S":[inter_nbr[4]],"W":[inter_nbr[5]]})
# s_df=pd.DataFrame({"C":[inter_edu[0]],"E":[inter_edu[1]],"H":[inter_edu[2]],"N":[inter_edu[3]],"S":[inter_edu[4]],"W":[inter_edu[5]]})
# w_df=pd.DataFrame({"C":[inter_work[0]],"E":[inter_work[1]],"H":[inter_work[2]],"N":[inter_work[3]],"S":[inter_work[4]],"W":[inter_work[5]]})
# table_1b=pd.concat([c_df,e_df,h_df,n_df,s_df,w_df],axis=0)

# # Save dataframe
# table_1b.to_csv(f"{plot_path}/table_1b.csv",index=False)

