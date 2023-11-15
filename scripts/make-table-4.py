#!/bin/python

import csv
import matplotlib.pyplot as plt
import numpy as np

num_entries = 4
num_elts = [1e6, 1e7, 1e8, 1e9]

# sizes are all in bytes / elt
pma_sizes = [0] * num_entries
cpma_sizes = [0] * num_entries
upac_sizes = [0] * num_entries
cpac_sizes = [0] * num_entries

# make the CSV
# read PMA times
with open('outputs/pma_micro_sizes.out', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='=')
    idx = 0
    for row in reader:
        pma_sizes[idx] = float(row[-1])
        idx = idx + 1

with open('outputs/cpma_micro_sizes.out', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='=')
    idx = 0
    for row in reader:
        cpma_sizes[idx] = float(row[-1])
        idx = idx + 1

with open('../other_systems/CPAM/examples/microbenchmarks/outputs/upac_micro_sizes.out') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    idx = 0
    for row in reader:
        for i in range(0, num_entries):
            upac_sizes[i] = float(row[i]) / num_elts[i]
        idx = idx + 1

with open('../other_systems/CPAM/examples/microbenchmarks/outputs/cpac_micro_sizes.out') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    idx = 0
    for row in reader:
        for i in range(0, num_entries):
            cpac_sizes[i] = float(row[i]) / num_elts[i]
        idx = idx + 1

# make the CSV
with open('./csvs/micro_sizes.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["Num elts", "U-PaC", "PMA", "PMA/U-PaC", "C-PaC", "CPMA", "CPMA/C-PaC", "CPMA/PMA"]
    writer.writerow(field)
    for i in range(0, num_entries):
        writer.writerow([str(num_elts[i]), "{:.2f}".format(upac_sizes[i]),"{:.2f}".format(pma_sizes[i]),"{:.2f}".format(pma_sizes[i]/upac_sizes[i]),"{:.2f}".format(cpac_sizes[i]),"{:.2f}".format(cpma_sizes[i]),"{:.2f}".format(cpma_sizes[i]/cpac_sizes[i]),"{:.2f}".format(cpma_sizes[i]/pma_sizes[i])])
