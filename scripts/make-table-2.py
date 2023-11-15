#!/bin/python

import csv
import matplotlib.pyplot as plt
import numpy as np

num_entries = 8
num_elts = 1e8
batch_sizes = [1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
pma_serial_times = [0] * num_entries
cpma_serial_times = [0] * num_entries
pma_parallel_times = [0] * num_entries
cpma_parallel_times = [0] * num_entries

# make the CSV
# read PMA times
with open('outputs/pma_serial_batch.out', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    idx = 0
    for row in reader:
        pma_serial_times[idx] = float(row[1]) / 1e6
        idx = idx + 1
print('pma serial times')
print(pma_serial_times)

with open('outputs/pma_parallel_batch.out', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    idx = 0
    for row in reader:
        pma_parallel_times[idx] = float(row[1]) / 1e6
        idx = idx + 1
print('pma parallel times')
print(pma_parallel_times)

# calculate throughputs
pma_serial_throughputs = [0] * num_entries
pma_parallel_throughputs = [0] * num_entries

for i in range(0, num_entries):
    pma_serial_throughputs[i] = num_elts / pma_serial_times[i]
    pma_parallel_throughputs[i] = num_elts / pma_parallel_times[i]

# remove anything slower than serial point
for i in range(1, num_entries):
    if pma_serial_throughputs[i] < pma_serial_throughputs[0]:
        pma_serial_throughputs[i] = pma_serial_throughputs[0]
    if pma_parallel_throughputs[i] < pma_parallel_throughputs[0]:
        pma_parallel_throughputs[i] = pma_parallel_throughputs[0]

serial_speedups_over_point = [0] * num_entries
parallel_speedup_over_serial = [0] * num_entries
overall_speedup = [0] * num_entries

for i in range(0, num_entries):
    serial_speedups_over_point[i] = pma_serial_throughputs[i] / pma_serial_throughputs[0]
    parallel_speedup_over_serial[i] = pma_parallel_throughputs[i] / pma_serial_throughputs[i]
    overall_speedup[i] = pma_parallel_throughputs[i] / pma_serial_throughputs[0]

# make the CSV
with open('./csvs/pma_serial_parallel_batch.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["Batch size", "Serial TP", "Speedup over serial point", "Parallel TP", "Speedup over serial batch", "Overall speedup"]
    writer.writerow(field)
    for i in range(1, num_entries):
        batch_size = batch_sizes[i]
        if i == 1: batch_size = "1-10"
        serial = pma_serial_throughputs[i]
        parallel = pma_parallel_throughputs[i]
        su_serial_point = serial_speedups_over_point[i]
        su_parallel = parallel_speedup_over_serial[i]
        overall = overall_speedup[i]
        writer.writerow([str(batch_size), str(int(serial)), "{:.2f}".format(su_serial_point), str(int(parallel)), "{:.2f}".format(su_parallel), "{:.2f}".format(overall)])
