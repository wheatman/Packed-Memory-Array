#!/bin/bash

mkdir -p outputs
mkdir -p csvs
mkdir -p plots

N=100000000 # should be 100M
R=100000
RANGE=17179869184 # should eventually become 2^32 = 17179869184

# uncompressed PMA serial map range
../build/basic_uint64_t_uncompressed_Eytzinger map_range_single $N $R $RANGE 2>&1 | tail -1 >> outputs/pma_map_range_scaling.out

# compressed PMA batch inserts
../build/basic_uint64_t_delta_compressed_Eytzinger map_range_single $N $R $RANGE 2>&1 | tail -1 >> outputs/cpma_map_range_scaling.out
