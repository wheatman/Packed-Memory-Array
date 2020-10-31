#!/bin/bash

# args in the form
# {unif, zipf} {n} {filename (ignored in zipf)
# ./a.out 1 1000000 unif_1M
# ./a.out 2 1000000 zipf


./a.out 1 10000000 unif_10M

./a.out 2 10000000 zipf

