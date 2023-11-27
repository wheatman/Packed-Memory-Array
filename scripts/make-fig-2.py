#!/bin/python

import csv
import matplotlib.pyplot as plt
import numpy as np

# TODO: update num_* later for bigger test
num_entries = 21
num_elts = 100000000
num_queries = 100000
universe_size = 2**40
start_range_size = 14 # 2^14 = 16384

range_sizes = [0] * num_entries
exp_finds_per_search = [0] * num_entries
exp_finds_total = [0] * num_entries
for i in range(0, num_entries):
    range_idx = i + start_range_size
    range_sizes[i] = 2**range_idx
    exp_finds_per_search[i] = (num_elts * range_sizes[i])/universe_size
    exp_finds_total[i] = exp_finds_per_search[i] * num_queries
print(exp_finds_per_search)
print(exp_finds_total)

pam_times = [0] * num_entries
upac_times = [0] * num_entries
cpac_times = [0] * num_entries
pma_times = [0] * num_entries
cpma_times = [0] * num_entries

# TODO: make this ignore the small points with expected finds < 1
# probably something like 
# for row: if idx < start_range_size: continue else: add it to the list of times
# read PMA times
with open('outputs/pma_parallel_map_range.out', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    idx = 0
    for row in reader:
        if idx >= start_range_size:
            pma_times[idx - start_range_size] = float(row[-1])
        idx = idx + 1
print('pma times')
print(pma_times)

# read CPMA times
with open('outputs/cpma_parallel_map_range.out', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    idx = 0
    for row in reader:
        if idx >= start_range_size:
            cpma_times[idx - start_range_size] = float(row[-1])
        idx = idx + 1

print('cpma times')
print(cpma_times)


# read pam / cpam times
with open('../other_systems/CPAM/examples/microbenchmarks/outputs/pam_parallel_map_range.out') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    idx = 0
    for row in reader:
        pam_times[idx] = float(row[0])
        idx = idx + 1

print('pam times')
print(pam_times)

with open('../other_systems/CPAM/examples/microbenchmarks/outputs/upac_parallel_map_range.out') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    idx = 0
    for row in reader:
        upac_times[idx] = float(row[0])
        idx = idx + 1

print('upac times')
print(upac_times)

with open('../other_systems/CPAM/examples/microbenchmarks/outputs/cpac_parallel_map_range.out') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    idx = 0
    for row in reader:
        cpac_times[idx] = float(row[0])
        idx = idx + 1
print('cpac times')
print(cpac_times)

# calculate throughputs
pam_throughputs = [0] * num_entries
upac_throughputs = [0] * num_entries
cpac_throughputs = [0] * num_entries
pma_throughputs = [0] * num_entries
cpma_throughputs = [0] * num_entries

for i in range(0, num_entries):
    pma_throughputs[i] = exp_finds_total[i] / pma_times[i]
    cpma_throughputs[i] = exp_finds_total[i] / cpma_times[i]
    pam_throughputs[i] = exp_finds_total[i] / pam_times[i]
    upac_throughputs[i] = exp_finds_total[i]  / upac_times[i]
    cpac_throughputs[i] = exp_finds_total[i]  / cpac_times[i]

# plot it
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Expected elts per search")
plt.ylabel("Throughput (expected elts/s)")
plt.plot(exp_finds_per_search, pma_throughputs, label="PMA", linestyle=(0, (1, 10)))
plt.plot(exp_finds_per_search, cpma_throughputs, label="CPMA", linestyle=(0, (1, 1)))
plt.plot(exp_finds_per_search, pam_throughputs, label="PAM", linestyle='solid')
plt.plot(exp_finds_per_search, upac_throughputs, label="UPAC", linestyle='dashed')
plt.plot(exp_finds_per_search, cpac_throughputs, label="CPAC", linestyle='dashdot')

plt.legend(loc="upper left")
plt.savefig("./plots/map_range_micro.pdf", format="pdf")

# make the CSV

with open('./csvs/map_range_micro.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["Batch_size", "P-trees", "U-PaC", "PMA", "PMA/P-trees", "PMA/U-PaC", "C-PaC", "CPMA", "CPMA/C-PaC", "CPMA/PMA"]
    writer.writerow(field)
    for i in range(0, num_entries):
        finds_per_search = exp_finds_per_search[i]
        pam = pam_throughputs[i]
        upac = upac_throughputs[i]
        pma = pma_throughputs[i]
        cpac = cpac_throughputs[i]
        cpma = cpma_throughputs[i]

        writer.writerow([str(finds_per_search), str(int(pam)), str(int(upac)), str(int(pma)), "{:.2f}".format(pma/pam),"{:.2f}".format(pma/upac), str(int(cpac)), str(int(cpma)), "{:.2f}".format(cpma/cpac), "{:.2f}".format(cpma/pma)])
