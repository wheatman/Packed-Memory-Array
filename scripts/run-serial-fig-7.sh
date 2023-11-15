#!/bin/bash

mkdir -p outputs
mkdir -p csvs
mkdir -p plots

N=100000000
BATCH=1000000

# uncompressed PMA batch inserts
../build/basic_uint64_t_uncompressed_Eytzinger batch $N $BATCH 2>&1 | tail -1 >> outputs/pma_batch_scaling.out

# compressed PMA batch inserts
../build/basic_uint64_t_delta_compressed_Eytzinger batch $N $BATCH 2>&1 | tail -1 >> outputs/cpma_batch_scaling.out
