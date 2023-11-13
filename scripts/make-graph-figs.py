#!/bin/python

import csv
import matplotlib.pyplot as plt
import numpy as np
import os

batch_sizes = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
graph_names = ["LJ", "CO", "ER", "TW", "FS"]
small_graph_names = ["lj", "co", "er", "tw", "fs"]
num_algs = 3
num_graphs = 5
pr_col = 0
cc_col = 1
bc_col = 2
cpma_alg_idx = 2

# create empty matrices with rows = graphs and cols = algs to store times
cpma_alg_times = [[0 for x in range(num_algs)] for y in range(num_graphs)] 
cpac_alg_times = [[0 for x in range(num_algs)] for y in range(num_graphs)] 
aspen_alg_times = [[0 for x in range(num_algs)] for y in range(num_graphs)] 

cpma_sizes = [0]*num_graphs
cpma_batches = [0]*len(batch_sizes)
cpac_sizes = [0]*num_graphs
cpac_batches = [0]*len(batch_sizes)
aspen_sizes = [0]*num_graphs
aspen_batches = [0]*len(batch_sizes)

# read f-graph times
for i in range(0, num_graphs):
    with open(os.path.join('outputs/graph-eval/', small_graph_names[i] +'.out'), newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        idx = 0
        for row in reader:
            if row[0].startswith('size'):
                size_str = (row[0].split('='))[1]
                size_in_bytes = float(size_str.split(' ')[1])
                cpma_sizes[i] = size_in_bytes/1024**3
            elif row[0].startswith('F-Graph'):
                if row[cpma_alg_idx].strip() == 'PageRank':
                    cpma_alg_times[i][pr_col] = float(row[-1])
                elif row[cpma_alg_idx].strip() == 'Components':
                    cpma_alg_times[i][cc_col] = float(row[-1])
                elif row[cpma_alg_idx].strip() == 'BC':
                    cpma_alg_times[i][bc_col] = float(row[-1])
            elif row[0].isnumeric():
                cpma_batches[idx] = float(row[1])
                idx = idx + 1

print('CPMA')
print(cpma_alg_times)
print(cpma_sizes)
print(cpma_batches)

# read cpac algs and sizes
for i in range(0, num_graphs):
    with open(os.path.join('../other_systems/CPAM/examples/graphs/algorithms/outputs', small_graph_names[i] +'.out'), newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=':')
        idx = 0
        edge_size = 0
        node_size = 0
        for row in reader:
            if row[0].startswith('Edge'):
                edge_size = float((row[0].split('='))[1].strip())
            elif row[0].startswith('Vertex'):
                node_size = float((row[0].split('='))[1].strip())
                cpac_sizes[i] = (edge_size + node_size)/1024**3
            elif row[0].startswith('PR'):
                idx = pr_col
            elif row[0].startswith('BC'):
                idx = bc_col
            elif row[0].startswith('CC'):
                idx = cc_col
            elif row[0].startswith('#'):
                cpac_alg_times[i][idx] = float(row[1].strip())
print('\nCPAC')
print(cpac_alg_times)
print(cpac_sizes)

# read cpac batches
with open(os.path.join('../other_systems/CPAM/examples/graphs/run_batch_updates/outputs/batch_inserts.out'), newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    idx = 0
    for row in reader:
        if row[0] == 'RESULT: Insert' and idx < len(batch_sizes):
            cpac_batches[idx] = float(row[-1])
            idx = idx + 1

print(cpac_batches)

# read aspen times
for i in range(0, num_graphs):
    with open(os.path.join('../other_systems/aspen/code/outputs/algs/', small_graph_names[i] +'.out'), newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        idx = 0
        for row in reader:
            if row[0] == 'PR':
                idx = pr_col
            elif row[0] == 'BC':
                idx = bc_col
            elif row[0] == 'CC':
                idx = cc_col
            elif row[0] == 'RESULT':
                aspen_alg_times[i][idx] = float((row[-1].split('\t'))[0])


with open(os.path.join('../other_systems/aspen/code/outputs/sizes.out'), newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    idx = 0
    for row in reader:
        aspen_sizes[idx] = float(row[-1])
        idx = idx + 1

print('\nASPEN')
print(aspen_alg_times)
print(aspen_sizes)

with open(os.path.join('../other_systems/aspen/code/outputs/batches.out'), newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=':')
    idx = 0
    for row in reader:
        if(row[0] == 'Avg insert') and idx < len(batch_sizes):
            aspen_batches[idx] = float(row[-1])
            idx = idx + 1
print(aspen_batches)


# plot algs
# cpac time / x time = speedup over cpac
cpma_alg_ratios = [[0 for x in range(num_algs)] for y in range(num_graphs)] 
aspen_alg_ratios = [[0 for x in range(num_algs)] for y in range(num_graphs)] 

cpma_geomeans = [0] * (num_algs + 1)
aspen_geomeans = [0] * (num_algs + 1)

total_cpma_geomean = 1
total_aspen_geomean = 1
for j in range(0, num_algs):
    alg_cpma_geomean = 1
    alg_aspen_geomean = 1
    for i in range(0, num_graphs):

        cpma_alg_ratios[i][j] = cpac_alg_times[i][j] / cpma_alg_times[i][j]
        total_cpma_geomean = total_cpma_geomean * cpma_alg_ratios[i][j]
        alg_cpma_geomean = alg_cpma_geomean * cpma_alg_ratios[i][j]

        aspen_alg_ratios[i][j] = cpac_alg_times[i][j] / aspen_alg_times[i][j]
        total_aspen_geomean = total_aspen_geomean * aspen_alg_ratios[i][j]
        alg_aspen_geomean = alg_aspen_geomean * aspen_alg_ratios[i][j]

    # a.prod()**(1.0/len(a))
    cpma_geomeans[j] = alg_cpma_geomean ** (1.0/num_graphs)
    aspen_geomeans[j] = alg_aspen_geomean ** (1.0/num_graphs)

cpma_geomeans[-1] = total_cpma_geomean ** (1.0 / (num_graphs * num_algs))
aspen_geomeans[-1] = total_aspen_geomean ** (1.0 / (num_graphs * num_algs))
print('\nalg geomeans')
print(cpma_geomeans)
print(aspen_geomeans)

cpac_geomeans = [1]*(num_algs + 1)

algs_names = ['PR', 'CC', 'BC', 'Avg']

barWidth = 0.25
br1 = np.arange(num_algs + 1)
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2]

fig1 = plt.figure() 

# Make the plot
plt.bar(br1, cpac_geomeans, color ='lightcoral', width = barWidth, label ='C-PAC')
plt.bar(br2, cpma_geomeans, color ='cornflowerblue', width = barWidth, 
                hatch='/', label ='F-Graph') 
plt.bar(br3, aspen_geomeans, color ='mediumpurple', width = barWidth, 
                        hatch='\\', label ='Aspen') 
                         
# Adding Xticks 
plt.xlabel('Algorithm') 
plt.ylabel('Speedup over C-PaC')
plt.xticks([r + barWidth for r in range(len(algs_names))], 
         algs_names)
plt.legend()
plt.savefig("./plots/graph_algs.pdf", format="pdf")


# plot the inserts
cpac_throughputs = [0]*len(batch_sizes)
cpma_throughputs = [0]*len(batch_sizes)
aspen_throughputs = [0]*len(batch_sizes)

for i in range(0, len(batch_sizes)):
        cpac_throughputs[i] = batch_sizes[i]/cpac_batches[i]
        cpma_throughputs[i] = batch_sizes[i]/cpma_batches[i]
        aspen_throughputs[i] = batch_sizes[i]/aspen_batches[i]

fig2 = plt.figure() 

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Batch size")
plt.ylabel("Throughput (inserts/s)")
plt.plot(batch_sizes, cpma_throughputs, label="F-Graph", linestyle='solid')
plt.plot(batch_sizes, cpac_throughputs, label="C-PaC", linestyle='dashed')
plt.plot(batch_sizes, aspen_throughputs, label="Aspen", linestyle='dashdot')

plt.legend(loc="upper left")
plt.savefig("./plots/graph_batch_insert.pdf", format="pdf")


# make size CSV
N = [4.8, 3.1, 10, 62, 125]
M = [86, 234, 1000, 2405, 3612]

with open('./csvs/graph_sizes.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ['Graph', 'N', 'M', 'F-Graph', 'C-PaC', 'Aspen', 'F/C', 'F/A']
    writer.writerow(field)
    for i in range(0, num_graphs):
        writer.writerow([graph_names[i], str(N[i]), str(M[i]),"{:.2f}".format(cpma_sizes[i]), "{:.2f}".format(cpac_sizes[i]), "{:.2f}".format(aspen_sizes[i]), "{:.2f}".format(cpma_sizes[i]/cpac_sizes[i]), "{:.2f}".format(cpma_sizes[i]/aspen_sizes[i])])

# make graph alg csv
with open('./csvs/graph_algs.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ['PR', '', '', '', '', '', 'CC', '', '', '', '', '', 'BC', '', '', '', '', '']
    writer.writerow(field)
    field = ['Graph', 'Aspen', 'C-PaC', 'F-Graph', 'A/F', 'C/F', 'Aspen', 'C-PaC', 'F-Graph', 'A/F', 'C/F', 'Aspen', 'C-PaC', 'F-Graph', 'A/F', 'C/F']
    writer.writerow(field)
    for i in range(0, num_graphs):
        writer.writerow([graph_names[i], "{:.2f}".format(aspen_alg_times[i][pr_col]), "{:.2f}".format(cpac_alg_times[i][pr_col]), "{:.2f}".format(cpma_alg_times[i][pr_col]), "{:.2f}".format(aspen_alg_times[i][pr_col]/cpma_alg_times[i][pr_col]), "{:.2f}".format(cpac_alg_times[i][pr_col]/cpma_alg_times[i][pr_col]),  "{:.2f}".format(aspen_alg_times[i][cc_col]), "{:.2f}".format(cpac_alg_times[i][cc_col]), "{:.2f}".format(cpma_alg_times[i][cc_col]), "{:.2f}".format(aspen_alg_times[i][cc_col]/cpma_alg_times[i][cc_col]), "{:.2f}".format(cpac_alg_times[i][cc_col]/cpma_alg_times[i][cc_col]),  "{:.2f}".format(aspen_alg_times[i][bc_col]), "{:.2f}".format(cpac_alg_times[i][bc_col]), "{:.2f}".format(cpma_alg_times[i][bc_col]), "{:.2f}".format(aspen_alg_times[i][bc_col]/cpma_alg_times[i][bc_col]), "{:.2f}".format(cpac_alg_times[i][bc_col]/cpma_alg_times[i][bc_col])]) 

# make graph batch insert csv

with open('./csvs/graph_batch_insert.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ['Graph', 'Aspen', 'C-PaC', 'F-Graph', 'F/A', 'F/C']
    writer.writerow(field)
    for i in range(0, num_graphs):
        writer.writerow([graph_names[i], "{:.2f}".format(aspen_throughputs[i]), "{:.2f}".format(cpac_throughputs[i]), "{:.2f}".format(cpma_throughputs[i]), "{:.2f}".format(cpma_throughputs[i]/aspen_throughputs[i]), "{:.2f}".format(cpma_throughputs[i]/cpac_throughputs[i])])


