#!/bin/bash

./ht-scale 5 1 ../data/unif_10M 9988471 scalability 0
taskset -c 4 ./ht-scale 5 1 ../data/unif_10M 9988471 scalability 1

./ht-scale 5 1 ../data/zipf_4194304_0.400000_10000000 8997731 scalability 0
taskset -c 4 ./ht-scale 5 1 ../data/zipf_4194304_0.400000_10000000 8997731 scalability 1
