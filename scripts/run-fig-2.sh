#!/bin/bash

mkdir -p outputs
mkdir -p csvs
mkdir -p plots

N=1000 # num elements -> 100M
Q=100 # num queries -> 100k
maxlogR=10 # max log of range size (goes up by power of 2) -> should be 34 in the final

# uncompressed PMA batch inserts
numactl -i all ../build/basic_uint64_t_uncompressed_Eytzinger map_range $N $Q $maxlogR 2>&1 | tail -$maxlogR >> outputs/pma_parallel_map_range.out

# compressed PMA batch inserts
numactl -i all ../build/basic_uint64_t_delta_compressed_Eytzinger map_range $N $Q $maxlogR 2>&1 | tail -$maxlogR >> outputs/cpma_parallel_map_range.out
