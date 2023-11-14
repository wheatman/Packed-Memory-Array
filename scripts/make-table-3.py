#!/bin/python

import csv
import matplotlib.pyplot as plt
import numpy as np

num_entries = 4 # TODO: update later with bigger test
num_elts = 1e8
batch_sizes = [1e4, 1e5, 1e6, 1e7]
pma_serial_times = [0] * num_entries
rma_serial_times = [0] * num_entries

# make the CSV
# read PMA times
with open('outputs/pma_serial_batch.out', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    idx = 0
    for row in reader:
        if int(row[0]) >= 1e4:
            pma_serial_times[idx] = float(row[1]) / 1e6
            idx = idx + 1
print('pma serial times')
print(pma_serial_times)

total_time_so_far = 0
trials_so_far = 0
idx = 0
with open('../other_systems/rma/rma_batch_insert.out', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if row[0].startswith('batch_size') and idx < num_entries:
            if trials_so_far > 0:
                rma_serial_times[num_entries - 1 - idx] = float(total_time_so_far) / trials_so_far
                total_time_so_far = 0
                trials_so_far = 0
                idx = idx + 1
        else:
            total_time_so_far = total_time_so_far + float((row[-1].split(' '))[0])
            trials_so_far = trials_so_far + 1

if idx < num_entries:
    rma_serial_times[num_entries - 1 - idx] = float(total_time_so_far) / trials_so_far

print('rma serial times')
print(rma_serial_times)


# calculate throughputs
pma_serial_throughputs = [0] * num_entries
rma_serial_throughputs = [0] * num_entries

for i in range(0, num_entries):
    pma_serial_throughputs[i] = num_elts / pma_serial_times[i]
    rma_serial_throughputs[i] = num_elts / rma_serial_times[i]

# make the CSV
with open('./csvs/batch_rma_pma.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["Batch size", "RMA", "PMA", "PMA/RMA"]
    writer.writerow(field)
    for i in range(0, num_entries):
        batch_size = batch_sizes[i]
        if i == 0: batch_size = "1-1e4"
        writer.writerow([str(batch_size), str(int(rma_serial_throughputs[i])), str(int(pma_serial_throughputs[i])), "{:.2f}".format(pma_serial_throughputs[i]/rma_serial_throughputs[i])])
