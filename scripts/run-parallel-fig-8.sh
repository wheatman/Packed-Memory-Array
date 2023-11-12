#!/bin/bash

mkdir -p outputs
mkdir -p csvs
mkdir -p plots

N=1000 # should be 100M
R=100000
RANGE=128 # should eventually become 2^32 = 17179869184

for THREADS in 2 4 8 16 32 64 128
do
	# uncompressed PMA batch inserts
	PARLAY_NUM_THREADS=$THREADS numactl -i all ../build/basic_uint64_t_uncompressed_Eytzinger map_range_single $N $R $RANGE 2>&1 | tail -1 >> outputs/pma_map_range_scaling.out
	# map_range_single <num_elements_start = 100000000> <num_ranges = 100000> <range_size = 17179869184>

	# compressed PMA batch inserts
	PARLAY_NUM_THREADS=$THREADS numactl -i all ../build/basic_uint64_t_delta_compressed_Eytzinger map_range_single $N $R $RANGE 2>&1 | tail -1 >> outputs/cpma_map_range_scaling.out
done
