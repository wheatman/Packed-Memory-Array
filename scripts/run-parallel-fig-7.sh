#!/bin/bash

mkdir -p outputs
mkdir -p csvs
mkdir -p plots

N=100000000 # N=1000
BATCH=1000000 # BATCH=100

for THREADS in 2 4 8 16 32 64 128
do
	# uncompressed PMA batch inserts
	PARLAY_NUM_THREADS=$THREADS numactl -i all ../build/basic_uint64_t_uncompressed_Eytzinger batch $N $BATCH 2>&1 | tail -1 >> outputs/pma_batch_scaling.out

	# compressed PMA batch inserts
	PARLAY_NUM_THREADS=$THREADS numactl -i all ../build/basic_uint64_t_delta_compressed_Eytzinger batch $N $BATCH 2>&1 | tail -1 >> outputs/cpma_batch_scaling.out
done
