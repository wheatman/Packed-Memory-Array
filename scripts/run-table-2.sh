#!/bin/bash

mkdir -p outputs
mkdir -p csvs
mkdir -p plots

N=1000
logN=3 
# uncompressed PMA batch inserts
numactl -i all ../build/basic_uint64_t_uncompressed_Eytzinger batch_bench $N 2>&1 | tail -$logN >> outputs/pma_serial_batch.out
