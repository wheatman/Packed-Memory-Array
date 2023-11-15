#!/bin/bash

mkdir -p outputs
mkdir -p csvs
mkdir -p plots

N=100000000
logN=8 
# uncompressed PMA batch inserts
../build/basic_uint64_t_uncompressed_Eytzinger batch_bench $N 2>&1 | tail -$logN >> outputs/pma_serial_batch.out
