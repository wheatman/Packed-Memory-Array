#!/bin/bash

mkdir -p outputs
mkdir -p csvs
mkdir -p plots

# TODO: this should be 10^6 -> 10^9. it is small for testing purposes
#for N in 1000000 10000000 100000000 1000000000
for N in 100 1000 10000 100000
do
	# uncompressed PMA
	../build/basic_uint64_t_uncompressed_Eytzinger single $N 2>&1 | head -1 >> outputs/pma_micro_sizes.out

	# compressed PMA
	../build/basic_uint64_t_delta_compressed_Eytzinger single $N 2>&1 | head -1 >> outputs/cpma_micro_sizes.out
done
