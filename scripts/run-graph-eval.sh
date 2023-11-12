#!/bin/bash


TRIALS=10
MAXBATCH=100000000

mkdir -p outputs/graph-eval

../build/basic_uint64_t_delta_compressed_Eytzinger graph ../graphs/lj.adj $TRIALS 0 0 2>&1 >> outputs/graph-eval/lj.out

../build/basic_uint64_t_delta_compressed_Eytzinger graph ../graphs/co.adj $TRIALS 1000 0 2>&1 >> outputs/graph-eval/co.out

../build/basic_uint64_t_delta_compressed_Eytzinger graph ../graphs/er.adj $TRIALS 0 0 2>&1 >> outputs/graph-eval/er.out

../build/basic_uint64_t_delta_compressed_Eytzinger graph ../graphs/tw.adj $TRIALS 12 0 2>&1 >> outputs/graph-eval/tw.out

../build/basic_uint64_t_delta_compressed_Eytzinger graph ../graphs/fs.adj $TRIALS 100000 $MAXBATCH 2>&1 >> outputs/graph-eval/fs.out
