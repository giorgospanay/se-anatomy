import os, sys, glob, parse, pickle, gc
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


# Read cmd args
args=sys.argv[1:]
mode=""
if len(args)>=1:
	mode=args[0]


# Read node_final: node statistics plus attributes
print("Reading node_final")
node_df=pd.read_csv(f"{log_path}/node_final_2017.csv",index_col="PersonNr",header=0)

print(node_df[node_df["actual_triangles"]<node_df["pure_triangles"]][["actual_triangles","pure_triangles","excess_closure","lcc"]])

# ------------------------------------------------------------------------

# Fig. 3A: Embeddedness (hist, bins of 4?)
print("Figure 3A")

# ------------------------------------------------------------------------

# Fig. 3B: Tie range, where embeddedness=0.
print("Figure 3B")

# ------------------------------------------------------------------------

# Fig. 4A: # nodes vs. closure score (lcc, excess)
print("Figure 4A")

fig4a, ax4a = plt.subplots()

# Obtain histograms in 200 bins
hist_lcc=node_df["lcc"].value_counts(bins=200).sort_index()
print(hist_lcc)
hist_exc=node_df["excess_closure"].value_counts(bins=200).sort_index()
print(node_df["excess_closure"].min())
print(node_df["excess_closure"].max())

print(hist_exc)

# Plot clustering coefficient histogram, fill under curve
ax4a.plot(hist_lcc,color="blue",marker=",",linestyle="solid")
ax4a.fill_between(len(hist_lcc),hist_lcc,color="blue")
# Plot excess closure histogram, fill under curve
ax4a.plot(hist_exc,color="red",marker=",",linestyle="solid")
ax4a.fill_between(len(hist_exc),hist_exc,color="red")

ax4a.set_xlabel("Closure")
ax4a.set_ylabel("Number of nodes")
ax4a.set_yscale("log")
ax4a.set_yticks([1,10,100,1000,10000,100000,1000000],labels=["1","10","100","1K","10K","100K","1M"])

fig4a.legend(labels=["Clustering coefficient","Excess closure"],loc="upper center",alignment="center",ncols=2)
fig4a.savefig(f"{plot_path}/fig4a.png",bbox_inches='tight',dpi=300)

# ------------------------------------------------------------------------

# Fig. 4B: closure score (lcc, excess) vs degree (plus percentiles)
print("Figure 4B")

fig4b, ax4b = plt.subplots()

# Get stats for clustering coefficient
result_lcc=node_df.groupby("deg_total")["lcc"].agg(
	mean_value=('mean'),                   # Mean of B for each A
	percentile_25=('quantile', 0.25),      # 25th percentile of B for each A
	percentile_75=('quantile', 0.75)       # 75th percentile of B for each A
)
# Get stats for excess closure
result_exc=node_df.groupby("deg_total")["excess_closure"].agg(
	mean_value=('mean'),                   # Mean of B for each A
	percentile_25=('quantile', 0.25),      # 25th percentile of B for each A
	percentile_75=('quantile', 0.75)       # 75th percentile of B for each A
)
# Get degree index
result_index=result.index

# Plot clustering coefficient histogram, fill under curve
ax4b.plot(result_lcc["mean_value"],color="blue",marker=",",linestyle="solid")
ax4b.fill_between(result_index,result_lcc["percentile_25"],y2=result_lcc["percentile_75"],color="lightskyblue")
# Plot excess closure histogram, fill under curve
ax4b.plot(result_exc["mean_value"],color="red",marker=",",linestyle="solid")
ax4b.fill_between(result_index,result_exc["percentile_25"],y2=result_exc["percentile_75"],color="lightcoral")

ax4b.set_ylabel("Closure")
ax4b.set_xlabel("Degree")
ax4b.set_xscale("log")

ax4b.set_yticks([1,10,100,1000,10000,100000],labels=["1","10","100","1K","10K","100K"])

fig4b.legend(labels=["Clustering coefficient","Excess closure"],loc="upper center",alignment="center",ncols=2)
fig4b.savefig(f"{plot_path}/fig4b.png",bbox_inches='tight',dpi=300)


# ------------------------------------------------------------------------

# Fig. 5: income, education, urbanization x degree, excess closure, closeness vs. age
print("Figure 5")


# ------------------------------------------------------------------------