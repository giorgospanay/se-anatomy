import os, sys, glob, parse, pickle, gc
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
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

# ------------------------------------------------------------------------

# Read node_final: node statistics plus attributes
print("Reading node_final")
node_df=pd.read_csv(f"{log_path}/node_final_2017.csv",index_col="PersonNr",header=0)

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
hist_lcc=node_df["lcc"].value_counts(bins=50).sort_index()
hist_exc=node_df["excess_closure"].value_counts(bins=50).sort_index()

# Plot clustering coefficient histogram, excess closure histogram
x_lcc=hist_lcc.index.mid
# ax4a.plot(x_lcc,hist_lcc,color="blue",marker=" ",linestyle="solid")
# ax4a.plot(x_lcc,hist_exc,color="red",marker=" ",linestyle="solid")
# ax4a.plot(x_lcc,hist_lcc,color="lightskyblue",marker=" ",linestyle="solid")
# ax4a.plot(x_lcc,hist_exc,color="lightcoral",marker=" ",linestyle="solid")

# Fill under curves. Overlap for both?
ax4a.fill_between(x_lcc,hist_lcc,color="lightskyblue",interpolate=True)
ax4a.fill_between(x_lcc,hist_exc,color="lightcoral",interpolate=True)


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
	mean_value='mean',                   		# Mean of B for each A
	percentile_25=lambda x: x.quantile(0.25),     # 25th percentile of B for each A
	percentile_75=lambda x: x.quantile(0.75)      # 75th percentile of B for each A
)
print(result_lcc)
# Get stats for excess closure
result_exc=node_df.groupby("deg_total")["excess_closure"].agg(
	mean_value='mean',                   		# Mean of B for each A
	percentile_25=lambda x: x.quantile(0.25),     # 25th percentile of B for each A
	percentile_75=lambda x: x.quantile(0.75)      # 75th percentile of B for each A
)
print(result_exc)

# Get degree index
result_index=result_lcc.index

# Plot clustering coefficient histogram
ax4b.plot(result_lcc["mean_value"],color="blue",marker=",",linestyle="solid")
# Plot excess closure histogram
ax4b.plot(result_exc["mean_value"],color="red",marker=",",linestyle="solid")
# Fill under curves
ax4b.fill_between(result_index,result_lcc["percentile_25"],y2=result_lcc["percentile_75"],color="lightskyblue")
ax4b.fill_between(result_index,result_exc["percentile_25"],y2=result_exc["percentile_75"],color="lightcoral")


ax4b.set_ylabel("Closure")
ax4b.set_xlabel("Degree")
ax4b.set_xscale("log")

ax4b.set_xticks([1,10,100,1000],labels=["1","10","100","1K"])

fig4b.legend(labels=["Clustering coefficient","Excess closure"],loc="upper center",alignment="center",ncols=2)
fig4b.savefig(f"{plot_path}/fig4b.png",bbox_inches='tight',dpi=300)

# ------------------------------------------------------------------------

# Fig. 5: income, education, urbanization x degree, excess closure, closeness vs. age
print("Figure 5")

mpl.rcParams['agg.path.chunksize'] = 10000

# Set up the 3x3 grid
fig5, axes = plt.subplots(3,3,figsize=(15,15))

row_values = ["income_group","education_level","DeSO"]
column_pairs = [("age","deg_total"),("age","excess_closure"),("age","closeness")]

# Plot each row and column
for i, row_value in enumerate(row_values):
	# Filter data for each row label
	filter_data = node_df[node_df[row_value].notna()]

	
	for j, (x_col, y_col) in enumerate(column_pairs):
		ax = axes[i, j]

		# Group values per (row_value). Find mean (y_col) and sort
		row_data=filter_data.groupby(row_value).count()
		print(row_data)
		
		# Get colormap to be used
		cm_lbl=""
		if j==0:
			cm_lbl="Reds"
		elif j==1:
			cm_lbl="Blues"
		elif j==2:
			cm_lbl="Greens"

		# Get labels to be used
		tx_lbl=""
		val_lbl=[]
		if i==0:
			tx_lbl="Income decile"
			val_lbl=list(range(1,11))
		elif i==1:
			tx_lbl="Highest education level"
			val_lbl=["Primary","Secondary","Tertiary"]
		elif i==2:
			tx_lbl="Urbanization level"
			val_lbl=["Not urban","","Strongly urban"]

		# Get colormap and split into number of unique values left.
		cmap=plt.get_cmap(cm_lbl)
		cm_range=np.arange(0,len(row_data.index))
		color=cmap(cm_range)


		# Plot each unique value in the current row's column
		for idx,unique_val in enumerate(row_data.index):
			# Get data for unique_val
			plot_data=filter_data[filter_data[row_value]==unique_val]
			print(plot_data)

			# Find mean on x_col for every age
			plot_mean=plot_data.groupby("age")[y_col].mean()
			print(plot_mean)


			# Plot line from heatmap
			ax.set_xlim(left=0,right=90)
			ax.plot(plot_mean,color=color[idx],marker=" ",label=f'{row_value}={unique_val}')
			
		# Set labels
		y_lbl=""
		if y_col=="deg_total": 
			y_lbl="Degree"
		elif y_col=="closeness": 
			y_lbl="Closeness centrality"
		elif y_col=="excess_closure": 
			y_lbl="Excess closure"

		# Set labels
		ax.set_xlabel("Age")
		ax.set_ylabel(y_lbl)
		ax.set_xticks([0,20,40,60,80])
		
		# Add heatmap used as legend on top of figure
		cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.BoundaryNorm(cm_range,cmap.N),cmap=cmap),ax=ax,location="top",label=tx_lbl)
		cbar.set_ticks(ticks=cm_range,labels=val_lbl)

#fig5.savefig(f"{plot_path}/fig5.png",bbox_inches='tight',dpi=300)
fig5.savefig(f"{plot_path}/fig5.png",dpi=300)

# ------------------------------------------------------------------------