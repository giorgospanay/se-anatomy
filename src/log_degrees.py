import os, glob, parse, pickle
import networkx as nx
import numpy as np



#### Uncomment for synth
#
csv_path="../synth_test"
log_path="../result_test"
obj_path=csv_path


#### Uncomment for server data
#
# csv_path="../results2/"
# log_path="../result_logs/"
# obj_path=csv_path


work_all=nx.Graph()
fam_all=nx.Graph()
nbr_all=nx.Graph()
edu_all=nx.Graph()


# Flatten nx layers
def flatten_layers(l1,l2):
	# Flattening on nx can be done using compose:
	flat=nx.compose(l1,l2)

	# Attributes of l2 will take precedence. Workaround to sum weights on flattened:
	edge_data = {
	    e: l1.edges[e]["weight"] + l2.edges[e]["weight"] for e in l1.edges & l2.edges
	}
	nx.set_edge_attributes(flat, edge_data, "weight")
	return flat


# Open all csv files in path
for filename in glob.glob('*.csv'):
	with open(os.path.join(os.getcwd(), filename), 'r') as f:
		# Parse filename to determine type and year of layer
		layer_type=None
		layer_year=0
		print(filename)
		if "education" in filename:
			layer_type="education"
			layer_year=int(parse.parse("education{}.csv",filename)[0])

		elif "work" in filename:
			layer_type="work"
			layer_year=int(parse.parse("work{}.csv",filename)[0])

		elif "neighbourhood" in filename:
			layer_type="neighbourhood"
			layer_year=int(parse.parse("neighbourhood{}.csv",filename)[0])

		elif "final" in filename:
			layer_type="family"
			layer_year=int(parse.parse("final_network{}.csv",filename)[0])

		else: continue


		# Set here flags to ignore years / layer types etc.
		#if layer_year>2018: continue



		# Read and strip all lines.
		edge_lines = [line.rstrip() for line in f]
		net_year=None

		# If not family layer csv: read edgelist
		if layer_type!="family":
			# Parse from edgelist. Skip header line
			net_year=nx.parse_adjlist(edge_lines[1:],delimiter=",")
		# Otherwise for family: 
		else:
			net_year=nx.Graph()
			# Tokenize lines, set edges on all params set.
			# Fmt:: node,parent,child,partners,siblings,grandparents,grandchildren,aunts_uncles,niece_nephews,cousins,family_household
			for ln in edge_lines[1:]:
				tok=ln.split(",")
				n1=tok[0]
				for i in range(1,len(tok)):
					# Evaluate set values
					tok_s=eval(tok[i])
					# For eveny n2, add (n1,n2) if does not exist in G:
					for n2 in tok_s:
						if n2 not in G[n1]:
							net_year.add_edge(n1,n2)


		# Calculate degrees & deg. histogram and save to a file
		degs=net_year.degree()
		with open(f"{log_path}/degrees_{layer_type}{layer_year}.txt","w") as d_wf:
			for n,d in degs:
				d_wf.write(f"{n} {d}")

		deg_hist=net_year.degree_histogram()
		with open(f"{log_path}/histogram_{layer_type}{layer_year}.txt","w") as h_wf:
			h_wf.write(f"{deg_hist}")


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


# Calculate degree & histogram for x_all networks and save file
degs=fam_all.degree()
with open(f"{log_path}/degrees_fam_all.txt","w") as d_wf:
	for n,d in degs:
		d_wf.write(f"{n} {d}")
hist=fam_all.degree_histogram()
with open(f"{log_path}/histogram_fam_all.txt","w") as h_wf:
	h_wf.write(f"{hist}")

degs=nbr_all.degree()
with open(f"{log_path}/degrees_nbr_all.txt","w") as d_wf:
	for n,d in degs:
		d_wf.write(f"{n} {d}")
hist=nbr_all.degree_histogram()
with open(f"{log_path}/histogram_nbr_all.txt","w") as h_wf:
	h_wf.write(f"{hist}")

degs=edu_all.degree()
with open(f"{log_path}/degrees_edu_all.txt","w") as d_wf:
	for n,d in degs:
		d_wf.write(f"{n} {d}")
hist=edu_all.degree_histogram()
with open(f"{log_path}/histogram_edu_all.txt","w") as h_wf:
	h_wf.write(f"{hist}")

degs=work_all.degree()
with open(f"{log_path}/degrees_work_all.txt","w") as d_wf:
	for n,d in degs:
		d_wf.write(f"{n} {d}")
hist=work_all.degree_histogram()
with open(f"{log_path}/histogram_work_all.txt","w") as h_wf:
	h_wf.write(f"{hist}")


# Save x_all networks as pickle
with open(f"{obj_path}/fam_all.nx","wb") as n_out:
	pickle.dump(fam_all,n_out)
with open(f"{obj_path}/nbr_all.nx","wb") as n_out:
	pickle.dump(nbr_all,n_out)
with open(f"{obj_path}/edu_all.nx","wb") as n_out:
	pickle.dump(edu_all,n_out)
with open(f"{obj_path}/work_all.nx","wb") as n_out:
	pickle.dump(work_all,n_out)



