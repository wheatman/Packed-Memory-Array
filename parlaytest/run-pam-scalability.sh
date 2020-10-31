#!/bin/bash

./pam_scalability 5 ../data/unif_10M scalability 0
taskset -c 4 ./pam_scalability 5 ../data/unif_10M scalability 1

./pam_scalability 5 ../data/zipf_4194304_0.400000_10000000 scalability 0
taskset -c 4 ./pam_scalability 5 ../data/zipf_4194304_0.400000_10000000 scalability 1
