#!/bin/python

import sys
import subprocess
from subprocess import call
from subprocess import Popen, PIPE

basedir = '../data/'

"""
inputs = ['zipf_33554432_0.600000_10000000',
'zipf_33554432_0.800000_10000000', 
'zipf_33554432_1.000000_10000000',
'zipf_33554432_1.200000_10000000',
'zipf_33554432_1.400000_10000000',
'unif_10M',
'twitter_max']

uniqs = [7291207,
5369807,
2525395,
636437,
132035,
9988471,
2997487
]

inputs = [
'zipf_67108864_0.400000_100000000',
'zipf_67108864_0.600000_100000000',
'zipf_67108864_0.800000_100000000',
'unif_100M'
]

uniqs = [
48484343,
43045705,
32792385,
98843504
]
"""
inputs = [
'unif_10M',
'unif_100M'
]

uniqs = [
9988471,
98843504
]


assert(len(inputs) == len(uniqs))

batch_start = 7
batch_end = 6

num_trials = 5

out = open(sys.argv[1], "a")
for j in range(0, len(inputs)):
  f = inputs[j]
  uniq = uniqs[j]
  print("\nfile: " + f)
  for i in range(batch_start, batch_end, -1):
    batch_size = 10**i
    p = Popen(["./ht", str(num_trials), str(batch_size), basedir + f, str(uniq)], stdout = out)
    p.wait()
