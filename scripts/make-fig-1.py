#!/bin/python

import csv
import matplotlib.pyplot as plt
import numpy as np

num_entries = 8 # TODO: update later with bigger test
num_elts = 1e8
batch_sizes = [1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
pam_times = [0] * num_entries
upac_times = [0] * num_entries
cpac_times = [0] * num_entries
pma_times = [0] * num_entries
cpma_times = [0] * num_entries

# make the CSV
# read PMA times
with open('outputs/pma_parallel_batch.out', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    idx = 0
    for row in reader:
        pma_times[idx] = float(row[1]) / 1e6
        idx = idx + 1
print('pma times')
print(pma_times)

# read CPMA times
with open('outputs/cpma_parallel_batch.out', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    idx = 0
    for row in reader:
        cpma_times[idx] = float(row[1]) / 1e6
        idx = idx + 1
print('cpma times')
print(cpma_times)

with open('../other_systems/CPAM/examples/microbenchmarks/outputs/pam_parallel_batch.out') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        for i in range(0, num_entries):
            pam_times[num_entries - i - 1] = float(row[i])
print('pam times')
print(pam_times)

with open('../other_systems/CPAM/examples/microbenchmarks/outputs/upac_parallel_batch.out') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        for i in range(0, num_entries):
            upac_times[num_entries - i - 1] = float(row[i])
print('upac times')
print(upac_times)

with open('../other_systems/CPAM/examples/microbenchmarks/outputs/cpac_parallel_batch.out') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        for i in range(0, num_entries):
            cpac_times[num_entries - i - 1] = float(row[i])
print('cpac times')
print(cpac_times)

# calculate throughputs
pam_throughputs = [0] * num_entries
upac_throughputs = [0] * num_entries
cpac_throughputs = [0] * num_entries
pma_throughputs = [0] * num_entries
cpma_throughputs = [0] * num_entries

for i in range(0, num_entries):
    pma_throughputs[i] = num_elts / pma_times[i]
    cpma_throughputs[i] = num_elts / cpma_times[i]
    pam_throughputs[i] = num_elts / pam_times[i]
    upac_throughputs[i] = num_elts / upac_times[i]
    cpac_throughputs[i] = num_elts / cpac_times[i]

# plot it
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Batch size")
plt.ylabel("Throughput (inserts/s)")
plt.plot(batch_sizes, pma_throughputs, label="PMA", linestyle=(0, (1, 10)))
plt.plot(batch_sizes, cpma_throughputs, label="CPMA", linestyle=(0, (1, 1)))
plt.plot(batch_sizes, pam_throughputs, label="PAM", linestyle='solid')
plt.plot(batch_sizes, upac_throughputs, label="UPAC", linestyle='dashed')
plt.plot(batch_sizes, cpac_throughputs, label="CPAC", linestyle='dashdot')

plt.legend(loc="upper left")
plt.savefig("./plots/batch_insert_micro.pdf", format="pdf")

# make the CSV

with open('./csvs/batch_insert_micro.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["Batch_size", "P-trees", "U-PaC", "PMA", "PMA/P-trees", "PMA/U-PaC", "C-PaC", "CPMA", "CPMA/C-PaC", "CPMA/PMA"]
    writer.writerow(field)
    for i in range(0, num_entries):
        batch_size = batch_sizes[i]
        pam = pam_throughputs[i]
        upac = upac_throughputs[i]
        pma = pma_throughputs[i]
        cpac = cpac_throughputs[i]
        cpma = cpma_throughputs[i]
        writer.writerow([str(batch_size), str(int(pam)), str(int(upac)), str(int(pma)), "{:.2f}".format(pma/pam),"{:.2f}".format(pma/upac), str(int(cpac)), str(int(cpma)), "{:.2f}".format(cpma/cpac), "{:.2f}".format(cpma/pma)])
