import os, glob, parse, pickle, sys, gc
import numpy as np
import pandas as pd

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
top=""
if len(args)>=1:
	mode=args[0]
	if len(args)>=2:
		top=args[1]


# Removes self-loops from the dataframe (assumes names are ["PersonNr","PersonNr2"])
def remove_self_loops(df):
	index_mask=df[(df["PersonNr"]==df["PersonNr2"])].index
	return df.drop(index_mask).reset_index(drop=True)

if mode=="fix":
	print("Quick fix")
	df=pd.read_csv(f"{csv_path}/family2017.csv")
	df=df[["PersonNr","PersonNr2"]].astype({"PersonNr":"int","PersonNr2":"int"})
	df.to_csv(f"{csv_path}/family2017.csv")
	df.to_csv(f"{csv_path}/edgelist_family2017.csv",sep=" ",index=False,header=False)


if mode=="fam":
	# Read full family layer
	print("Reading full family network")
	fam_df=read_in_network(pd.read_csv(f"{csv_path}/final_network2017.csv"),"PersonNr")
	print("Making edgelist")
	df=make_entire_edge_list(fam_df)
	print("Removing self loops")
	df=remove_self_loops(df)
	df=df[["PersonNr","PersonNr2"]].astype({"PersonNr":"int","PersonNr2":"int"})
	df.to_csv(f"{csv_path}/family2017.csv")
	df.to_csv(f"{csv_path}/edgelist_family2017.csv",sep=" ",index=False,header=False)

	print("df:")
	print(df)

	# Set correct labels here
	close_labels=["parent","siblings","child","partners"]
	ext_labels=["grandparents","grandchildren","aunts_uncles","niece_nephews","cousins"]
	house_labels=["family_household"]

	# Close family
	print("Make close family")
	close_mask=df["connection"].isin(close_labels)
	close_df=df[close_mask][["PersonNr","PersonNr2"]].astype({"PersonNr":"int","PersonNr2":"int"})
	close_df.to_csv(f"{csv_path}/close_family2017.csv")
	close_df.to_csv(f"{csv_path}/edgelist_close_family2017.csv",sep=" ",index=False,header=False)

	print("close df:")
	print(close_df)

	# Extended family
	print("Make extended family")
	ext_mask=df["connection"].isin(ext_labels)
	ext_df=df[ext_mask][["PersonNr","PersonNr2"]].astype({"PersonNr":"int","PersonNr2":"int"})
	ext_df.to_csv(f"{csv_path}/extended_family2017.csv")
	ext_df.to_csv(f"{csv_path}/edgelist_extended_family2017.csv",sep=" ",index=False,header=False)

	print("extended df:")
	print(ext_df)

	# Household
	print("Make household")
	house_mask=df["connection"].isin(house_labels)
	house_df=df[house_mask][["PersonNr","PersonNr2"]].astype({"PersonNr":"int","PersonNr2":"int"})
	house_df.to_csv(f"{csv_path}/household2017.csv")
	house_df.to_csv(f"{csv_path}/edgelist_household2017.csv",sep=" ",index=False,header=False)

	print("household df:")
	print(house_df)


# -----------------------------------------------------------------------------------

if mode=="nbr":
	# Read old neighbourhood
	print("Removing loops from neighbourhood")
	n_df=pd.read_csv(f"{csv_path}/old_neighbourhood2017.csv").astype({"PersonNr":"int","PersonNr2":"int"})
	n_df=remove_self_loops(n_df)
	n_df.to_csv(f"{csv_path}/neighbourhood2017.csv")
	n_df.to_csv(f"{csv_path}/edgelist_neighbourhood2017.csv",sep=" ",index=False,header=False)


# Read old education
if mode=="edu":
	print("Removing loops from education")
	e_df=pd.read_csv(f"{csv_path}/old_education2017.csv").astype({"PersonNr":"int","PersonNr2":"int"})
	e_df=remove_self_loops(e_df)
	e_df.to_csv(f"{csv_path}/education2017.csv")
	e_df.to_csv(f"{csv_path}/edgelist_education2017.csv",sep=" ",index=False,header=False)


# Read old work
if mode=="work":
	print("Removing loops from work")
	w_df=pd.read_csv(f"{csv_path}/old_work2017.csv").astype({"PersonNr":"int","PersonNr2":"int"})
	w_df=remove_self_loops(w_df)
	w_df.to_csv(f"{csv_path}/work2017.csv")
	w_df.to_csv(f"{csv_path}/edgelist_work2017.csv",sep=" ",index=False,header=False)




