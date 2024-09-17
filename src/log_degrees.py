import os, glob, parse, pickle, sys
import networkx as nx
import numpy as np
import pandas as pd

# Local imports
from simplify_family import read_in_network, simplify_family_layer, make_entire_edge_list



#### Uncomment for synth
#
# csv_path="../synth_test"
# log_path="../result_test"
# obj_path=csv_path


#### Uncomment for server data
#
csv_path="../results2"
log_path="../result_logs"
obj_path=csv_path


work_all=nx.Graph()
fam_all=nx.Graph()
nbr_all=nx.Graph()
edu_all=nx.Graph()


# Flatten nx layers
def flatten_layers(l1,l2):
	# Flattening on nx can be done using compose:
	flat=nx.compose(l1,l2)

	# # Attributes of l2 will take precedence. Workaround to sum weights on flattened:
	# edge_data = {
	#     e: l1.edges[e]["weight"] + l2.edges[e]["weight"] for e in l1.edges & l2.edges
	# }
	# nx.set_edge_attributes(flat, edge_data, "weight")

	return flat


args=sys.argv[1:]
mode=""

if len(args)>=1:
	mode=args[0]


# Open all csv files in path
for filename in glob.glob(f"{csv_path}/*.csv"):
	with open(filename,"r") as f:
		# Parse filename to determine type and year of layer
		layer_type=None
		layer_year=0
		print(filename)
		if "education" in filename:
			layer_type="education"
			layer_year=int(parse.parse(csv_path+"/education{}.csv",filename)[0])
		# final_network before "work" so that it is not recognized as work :)
		elif "final" in filename:
			layer_type="family"
			layer_year=int(parse.parse(csv_path+"/final_network{}.csv",filename)[0])
		elif "work" in filename:
			layer_type="work"
			#print(csv_path+"/work{}.csv")
			layer_year=int(parse.parse(csv_path+"/work{}.csv",filename)[0])
		elif "neighbourhood" in filename:
			layer_type="neighbourhood"
			layer_year=int(parse.parse(csv_path+"/neighbourhood{}.csv",filename)[0])
		else: continue


		# Set here flags to ignore years / layer types etc.
		if layer_type!=mode:
			print(f"{filename} skipped.")
			continue

		if layer_year>2018 or layer_year<2000: 
			print(f"{filename} skipped.")
			continue

		if layer_type=="family" and layer_year<2018:
			print(f"{filename} skipped.")
			continue


		net_year=None

		# If not family layer csv: read edgelist
		if layer_type!="family":
			# Read csv
			df=pd.read_csv(f)
			net_year=nx.from_pandas_edgelist(df,source="PersonNr",target="PersonNr2")

			df=None

		# Otherwise for family: 
		else:
			# Use pd and helper functions from simplify_family
			fam_raw=pd.read_csv(f)
			fam_df=read_in_network(fam_raw,"PersonNr")
			fam_edgelist = make_entire_edge_list(fam_df)
			# @TODO: Save to csv here if necessary

			fam_raw=None
			fam_df=None

			# Create network from edgelist
			net_year=nx.from_pandas_edgelist(fam_edgelist,source="PersonNr",target="PersonNr2",edge_attr="connection")

			fam_edgelist=None

		# Calculate degrees & deg. histogram and save to a file
		degs=net_year.degree()
		with open(f"{log_path}/degrees_{layer_type}{layer_year}.txt","w") as d_wf:
			for n,d in degs:
				d_wf.write(f"{n} {d}")
		degs=None

		deg_hist=nx.degree_histogram(net_year)
		with open(f"{log_path}/histogram_{layer_type}{layer_year}.txt","w") as h_wf:
			h_wf.write(f"{deg_hist}")
		deg_hist=None


		# Flatten net_year with overall
		if layer_type=="education":
			edu_all=flatten_layers(edu_all,net_year)
		elif layer_type=="work":
			work_all=flatten_layers(work_all,net_year)
		elif layer_type=="family":
			fam_all=flatten_layers(fam_all,net_year)
		elif layer_type=="neighbourhood":
			nbr_all=flatten_layers(nbr_all,net_year)
		else: continue

		net_year=None


# Calculate degree & histogram for x_all networks and save file
if mode=="family":
	degs=fam_all.degree()
	with open(f"{log_path}/degrees_fam_all.txt","w") as d_wf:
		for n,d in degs:
			d_wf.write(f"{n} {d}")
	hist=nx.degree_histogram(fam_all)
	with open(f"{log_path}/histogram_fam_all.txt","w") as h_wf:
		h_wf.write(f"{hist}")
	with open(f"{obj_path}/fam_all.nx","wb") as n_out:
		pickle.dump(fam_all,n_out)

elif mode=="neighbourhood":
	degs=nbr_all.degree()
	with open(f"{log_path}/degrees_nbr_all.txt","w") as d_wf:
		for n,d in degs:
			d_wf.write(f"{n} {d}")
	hist=nx.degree_histogram(nbr_all)
	with open(f"{log_path}/histogram_nbr_all.txt","w") as h_wf:
		h_wf.write(f"{hist}")
	with open(f"{obj_path}/nbr_all.nx","wb") as n_out:
		pickle.dump(nbr_all,n_out)


elif mode=="education":
	degs=edu_all.degree()
	with open(f"{log_path}/degrees_edu_all.txt","w") as d_wf:
		for n,d in degs:
			d_wf.write(f"{n} {d}")
	hist=nx.degree_histogram(edu_all)
	with open(f"{log_path}/histogram_edu_all.txt","w") as h_wf:
		h_wf.write(f"{hist}")
	with open(f"{obj_path}/edu_all.nx","wb") as n_out:
		pickle.dump(edu_all,n_out)

elif mode=="work":
	degs=work_all.degree()
	with open(f"{log_path}/degrees_work_all.txt","w") as d_wf:
		for n,d in degs:
			d_wf.write(f"{n} {d}")
	hist=nx.degree_histogram(work_all)
	with open(f"{log_path}/histogram_work_all.txt","w") as h_wf:
		h_wf.write(f"{hist}")
	with open(f"{obj_path}/work_all.nx","wb") as n_out:
		pickle.dump(work_all,n_out)


# If mode=flat: flatten all (flattened) networks one-by-one and produce degs
if mode=="flat":
	flat_all=None
	with open(f"{obj_path}/work_all.nx","rb") as n_out:
		flat_all=pickle.load(n_out)
	with open(f"{obj_path}/edu_all.nx","rb") as n_out:
		l2=pickle.load(n_out)
		flat_all=flatten_layers(flat_all,l2)
	with open(f"{obj_path}/nbr_all.nx","rb") as n_out:
		l2=pickle.load(n_out)
		flat_all=flatten_layers(flat_all,l2)
	with open(f"{obj_path}/fam_all.nx","rb") as n_out:
		l2=pickle.load(n_out)
		flat_all=flatten_layers(flat_all,l2)

	degs=flat_all.degree()
	with open(f"{log_path}/degrees_flat_all.txt","w") as d_wf:
		for n,d in degs:
			d_wf.write(f"{n} {d}")
	hist=nx.degree_histogram(flat_all)
	with open(f"{log_path}/histogram_flat_all.txt","w") as h_wf:
		h_wf.write(f"{hist}")

	# Save to pickle
	with open(f"{obj_path}/flat_all.nx","wb") as n_out:
		pickle.dump(flat_all,n_out)

