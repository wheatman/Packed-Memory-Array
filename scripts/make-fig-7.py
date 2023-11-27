#!/bin/python

import csv
import matplotlib.pyplot as plt
import numpy as np

num_entries = 8
num_elts = 100000000

num_threads = [0] * num_entries
for i in range(0, num_entries):
    num_threads[i] = 2**i

pma_times = [0] * num_entries
cpma_times = [0] * num_entries

# make the CSV
# read PMA times
with open('outputs/pma_batch_scaling.out', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    idx = 0
    for row in reader:
        pma_times[idx] = float(row[0])
        idx = idx + 1
print('pma times')
print(pma_times)

# read CPMA times
with open('outputs/cpma_batch_scaling.out', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    idx = 0
    for row in reader:
        cpma_times[idx] = float(row[0])
        idx = idx + 1
print('cpma times')
print(cpma_times)

# calculate throughputs
pma_scaling = [0] * num_entries
cpma_scaling = [0] * num_entries
pma_throughput = [0] * num_entries
cpma_throughput = [0] * num_entries

for i in range(0, num_entries):
    pma_scaling[i] = pma_times[0] / pma_times[i]
    cpma_scaling[i] = cpma_times[0] / cpma_times[i]
    pma_throughput[i] = num_elts / pma_times[i]
    cpma_throughput[i] = num_elts / cpma_times[i]

print('pma scaling')
print(pma_scaling)
print('cpma scaling')
print(cpma_scaling)

# plot it
print(num_threads)
plt.xscale("log", base=2)
plt.xlabel("Cores")
plt.ylabel("Speedup")
plt.plot(num_threads, pma_scaling, label="PMA", linestyle=(0, (1, 10)))
plt.plot(num_threads, cpma_scaling, label="CPMA", linestyle=(0, (1, 1)))

plt.legend(loc="upper left")
plt.savefig("./plots/batch_insert_scaling.pdf", format="pdf")

# make the CSV

with open('./csvs/batch_insert_scaling.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["Cores", "PMA throughput", "PMA speedup", "CPMA throughput", "CPMA speedup"]
    writer.writerow(field)
    for i in range(0, num_entries):
        writer.writerow([str(num_threads[i]), str(int(pma_throughput[i])), "{:.2f}".format(pma_scaling[i]), str(int(cpma_throughput[i])), "{:.2f}".format(cpma_scaling[i])])
