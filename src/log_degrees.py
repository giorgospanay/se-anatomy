import os, glob, parse, pickle, sys, gc
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


work_all=pd.DataFrame({"PersonNr":[], "PersonNr2":[]})
fam_all=pd.DataFrame({"PersonNr":[], "PersonNr2":[]})
nbr_all=pd.DataFrame({"PersonNr":[], "PersonNr2":[]})
edu_all=pd.DataFrame({"PersonNr":[], "PersonNr2":[]})


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

# Flatten layers using pandas
def pd_flatten_layers(l1,l2):
	return pd.concat([l1,l2],copy=False).drop_duplicates().reset_index(drop=True)



args=sys.argv[1:]
mode=""
half=""

if len(args)>=1:
	mode=args[0]
	if len(args)>=2:
		half=args[1]


# Open all csv files in path
for filename in glob.glob(f"{csv_path}/*.csv"):
	with open(filename,"r") as f:
		# Parse filename to determine type and year of layer
		layer_type=None
		layer_year=0
		if "education" in filename:
			layer_type="education"
			layer_year=int(parse.parse(csv_path+"/education{}.csv",filename)[0])
		# final_network before "work" so that it is not recognized as work :)
		elif "final" in filename:
			layer_type="family"
			layer_year=int(parse.parse(csv_path+"/final_network{}.csv",filename)[0])
		elif "work" in filename:
			if "work_" in filename:
				continue
			layer_type="work"
			#print(csv_path+"/work{}.csv")
			layer_year=int(parse.parse(csv_path+"/work{}.csv",filename)[0])
		elif "neighbourhood" in filename:
			layer_type="neighbourhood"
			layer_year=int(parse.parse(csv_path+"/neighbourhood{}.csv",filename)[0])
		else: continue


		# Set here flags to ignore years / layer types etc.
		if layer_type!=mode:
			#print(f"{filename} skipped.")
			continue

		# Flags for top/bot
		if half=="top" and layer_year>=2009:
			#print(f"{filename} skipped.")
			continue
		if half=="bot" and layer_year<=2010:
			#print(f"{filename} skipped.")
			continue

		# Flags for quarters
		if half=="1" and layer_year>2004:
			#print(f"{filename} skipped.")
			continue
		if half=="2" and (layer_year<=2004 or layer_year>2008):
			#print(f"{filename} skipped.")
			continue
		if half=="3" and (layer_year<=2008 or layer_year>2012):
			#print(f"{filename} skipped.")
			continue
		if half=="4" and layer_year<=2012:
			#print(f"{filename} skipped.")
			continue

		# All year-flags
		if layer_year>2017 or layer_year<2000: 
			#print(f"{filename} skipped.")
			continue

		# Family flag
		if layer_type=="family" and layer_year<2017:
			#print(f"{filename} skipped.")
			continue

		print(f"Processing {filename}.")


		df=None

		# If not family layer csv: read edgelist
		if layer_type!="family":
			# Read csv
			df=pd.read_csv(f)
			# Convert to nx?
			#net_year=nx.from_pandas_edgelist(df,source="PersonNr",target="PersonNr2")

		# Otherwise for family: 
		else:
			# Use pd and helper functions from simplify_family
			fam_df=read_in_network(pd.read_csv(f),"PersonNr")
			df = make_entire_edge_list(fam_df)[["PersonNr","PersonNr2"]]
			# @TODO: Save to csv here if necessary

			fam_df=None
			fam_edgelist=None

			# Create network from edgelist
			#net_year=nx.from_pandas_edgelist(fam_edgelist,source="PersonNr",target="PersonNr2",edge_attr="connection")
			


		# Get the garbage!
		gc.collect()

		net_year=nx.from_pandas_edgelist(df,source="PersonNr",target="PersonNr2")

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

		
		# Get the garbage!
		net_year=None
		gc.collect()

		# Flatten net_year with overall
		if layer_type=="education":
			edu_all=pd_flatten_layers(edu_all,df)
		elif layer_type=="work":
			work_all=pd_flatten_layers(work_all,df)
		elif layer_type=="family":
			fam_all=pd_flatten_layers(fam_all,df)
		elif layer_type=="neighbourhood":
			nbr_all=pd_flatten_layers(nbr_all,df)
		else: continue

		# Get the garbage!
		df=None
		gc.collect()


# Save half networks
if mode=="family":
	# Save to csv
	fam_all.to_csv(f"{csv_path}/fam_{half}.csv")
			
elif mode=="neighbourhood":
	# Save to csv
	nbr_all.to_csv(f"{csv_path}/nbr_{half}.csv")
	
elif mode=="education":
	# Save to csv
	edu_all.to_csv(f"{csv_path}/edu_{half}.csv")

elif mode=="work":
	# Save to csv
	work_all.to_csv(f"{csv_path}/work_{half}.csv")
	


# Mode for flattening halves:
if mode=="family-flat":
	# Read top/bot from csv
	fam_top=pd.read_csv(f"{csv_path}/fam_top.csv")
	fam_bot=pd.read_csv(f"{csv_path}/fam_bot.csv")
	fam_all=pd_flatten_layers(fam_top,fam_bot)

	fam_top=None
	fam_bot=None
	gc.collect()

	# Save to csv
	fam_all.to_csv(f"{csv_path}/fam_all.csv")
	# Calc degs
	net_all=nx.from_pandas_edgelist(fam_all,source="PersonNr",target="PersonNr2")
	degs=net_all.degree()
	with open(f"{log_path}/degrees_fam_all.txt","w") as d_wf:
		for n,d in degs:
			d_wf.write(f"{n} {d}")
	hist=nx.degree_histogram(net_all)
	with open(f"{log_path}/histogram_fam_all.txt","w") as h_wf:
		h_wf.write(f"{hist}")
		

elif mode=="neighbourhood-flat":
	# Read top/bot from csv
	nbr_all=pd_flatten_layers(pd.read_csv(f"{csv_path}/nbr_1.csv"),pd.read_csv(f"{csv_path}/nbr_2.csv"))
	nbr_all=pd_flatten_layers(nbr_all,pd.read_csv(f"{csv_path}/nbr_3.csv"))
	nbr_all=pd_flatten_layers(nbr_all,pd.read_csv(f"{csv_path}/nbr_4.csv"))


	# Save to csv
	nbr_all.to_csv(f"{csv_path}/nbr_all.csv")
	
	net_all=nx.from_pandas_edgelist(nbr_all,source="PersonNr",target="PersonNr2")

	nbr_all=None
	gc.collect()

	# Calc degs
	degs=net_all.degree()
	with open(f"{log_path}/degrees_nbr_all.txt","w") as d_wf:
		for n,d in degs:
			d_wf.write(f"{n} {d}")
	hist=nx.degree_histogram(net_all)
	with open(f"{log_path}/histogram_nbr_all.txt","w") as h_wf:
		h_wf.write(f"{hist}")


elif mode=="education-flat":
	# Read top/bot from csv
	edu_top=pd.read_csv(f"{csv_path}/edu_top.csv")
	edu_bot=pd.read_csv(f"{csv_path}/edu_bot.csv")
	edu_all=pd_flatten_layers(edu_top,edu_bot)

	edu_top=None
	edu_bot=None
	gc.collect()

	# Save to csv
	edu_all.to_csv(f"{csv_path}/edu_all.csv")
	# Calc degs
	net_all=nx.from_pandas_edgelist(edu_all,source="PersonNr",target="PersonNr2")
	degs=net_all.degree()
	with open(f"{log_path}/degrees_edu_all.txt","w") as d_wf:
		for n,d in degs:
			d_wf.write(f"{n} {d}")
	hist=nx.degree_histogram(net_all)
	with open(f"{log_path}/histogram_edu_all.txt","w") as h_wf:
		h_wf.write(f"{hist}")

elif mode=="work-flat":
	# Read top/bot from csv
	work_all=pd_flatten_layers(pd.read_csv(f"{csv_path}/work_1.csv"),pd.read_csv(f"{csv_path}/work_2.csv"))
	work_all=pd_flatten_layers(work_all,pd.read_csv(f"{csv_path}/work_3.csv"))
	work_all=pd_flatten_layers(work_all,pd.read_csv(f"{csv_path}/work_4.csv"))

	# Save to csv
	work_all.to_csv(f"{csv_path}/work_all.csv")

	# Calc degs
	net_all=nx.from_pandas_edgelist(work_all,source="PersonNr",target="PersonNr2")
	degs=net_all.degree()
	with open(f"{log_path}/degrees_work_all.txt","w") as d_wf:
		for n,d in degs:
			d_wf.write(f"{n} {d}")
	hist=nx.degree_histogram(net_all)
	with open(f"{log_path}/histogram_work_all.txt","w") as h_wf:
		h_wf.write(f"{hist}")


# If mode=flat-2017: flatten all networks for the 2017 year, produce degs
if mode=="flat-2017":

	fam_df=read_in_network(pd.read_csv(f"{csv_path}/final_network2017.csv"),"PersonNr")
	df = make_entire_edge_list(fam_df)
	print(df)
	df=df[["PersonNr","PersonNr2"]]

	flat_all=pd_flatten_layers(df,pd.read_csv(f"{csv_path}/education2017.csv"))
	flat_all=pd_flatten_layers(flat_all,pd.read_csv(f"{csv_path}/neighbourhood2017.csv"))
	flat_all=pd_flatten_layers(flat_all,pd.read_csv(f"{csv_path}/work2017.csv"))

	# Save to csv
	flat_all.to_csv(f"{csv_path}/flat_2017.csv")


	net_all=nx.from_pandas_edgelist(flat_all,source="PersonNr",target="PersonNr2")


	# with open(f"{obj_path}/work_all.nx","rb") as n_out:
	# 	flat_all=pickle.load(n_out)
	# with open(f"{obj_path}/edu_all.nx","rb") as n_out:
	# 	l2=pickle.load(n_out)
	# 	flat_all=flatten_layers(flat_all,l2)
	# with open(f"{obj_path}/nbr_all.nx","rb") as n_out:
	# 	l2=pickle.load(n_out)
	# 	flat_all=flatten_layers(flat_all,l2)
	# with open(f"{obj_path}/fam_all.nx","rb") as n_out:
	# 	l2=pickle.load(n_out)
	# 	flat_all=flatten_layers(flat_all,l2)

	degs=net_all.degree()
	with open(f"{log_path}/degrees_flat_all.txt","w") as d_wf:
		for n,d in degs:
			d_wf.write(f"{n} {d}")
	hist=nx.degree_histogram(net_all)
	with open(f"{log_path}/histogram_flat_all.txt","w") as h_wf:
		h_wf.write(f"{hist}")


# If mode=flat: flatten all (flattened) networks one-by-one and produce degs
if mode=="flat":
	flat_all=pd.DataFrame({"PersonNr":[], "PersonNr2":[]})

	flat_all=pd_flatten_layers(flat_all,pd.read_csv(f"{csv_path}/fam_all.csv"))
	flat_all=pd_flatten_layers(flat_all,pd.read_csv(f"{csv_path}/edu_all.csv"))
	flat_all=pd_flatten_layers(flat_all,pd.read_csv(f"{csv_path}/nbr_all.csv"))
	flat_all=pd_flatten_layers(flat_all,pd.read_csv(f"{csv_path}/work_all.csv"))

	# Save to csv
	flat_all.to_csv(f"{csv_path}/flat_all.csv")


	net_all=nx.from_pandas_edgelist(work_all,source="PersonNr",target="PersonNr2")


	# with open(f"{obj_path}/work_all.nx","rb") as n_out:
	# 	flat_all=pickle.load(n_out)
	# with open(f"{obj_path}/edu_all.nx","rb") as n_out:
	# 	l2=pickle.load(n_out)
	# 	flat_all=flatten_layers(flat_all,l2)
	# with open(f"{obj_path}/nbr_all.nx","rb") as n_out:
	# 	l2=pickle.load(n_out)
	# 	flat_all=flatten_layers(flat_all,l2)
	# with open(f"{obj_path}/fam_all.nx","rb") as n_out:
	# 	l2=pickle.load(n_out)
	# 	flat_all=flatten_layers(flat_all,l2)

	degs=net_all.degree()
	with open(f"{log_path}/degrees_flat_all.txt","w") as d_wf:
		for n,d in degs:
			d_wf.write(f"{n} {d}")
	hist=nx.degree_histogram(net_all)
	with open(f"{log_path}/histogram_flat_all.txt","w") as h_wf:
		h_wf.write(f"{hist}")

	# # Save to pickle
	# with open(f"{obj_path}/flat_all.nx","wb") as n_out:
	# 	pickle.dump(flat_all,n_out)

