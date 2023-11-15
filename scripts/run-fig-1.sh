#!/bin/bash

mkdir -p outputs
mkdir -p csvs
mkdir -p plots

N=100000000
logN=8
# uncompressed PMA batch inserts
numactl -i all ../build/basic_uint64_t_uncompressed_Eytzinger batch_bench $N 2>&1 | tail -$logN >> outputs/pma_parallel_batch.out

# compressed PMA batch inserts
numactl -i all ../build/basic_uint64_t_delta_compressed_Eytzinger batch_bench $N 2>&1 | tail -$logN >> outputs/cpma_parallel_batch.out
