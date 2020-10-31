#!/bin/python

import sys
import subprocess
from subprocess import call
from subprocess import Popen, PIPE

basedir = '../data/'

inputs = [
'zipf_67108864_0.400000_10000000',
'zipf_67108864_0.600000_10000000',
'zipf_67108864_0.800000_10000000',
'unif_10M',
'twitter_max'
]

uniqs = [
8997731,
8144270,
6101025,
9988471,
2997487
]

assert(len(inputs) == len(uniqs))

batch_start = 7
batch_end = -1
outfile = sys.argv[1]

num_trials = 5

for j in range(0, len(inputs)):
  f = inputs[j]
  uniq = uniqs[j]
  print("\nfile: " + f)
  for i in range(batch_start, batch_end, -1):
    batch_size = 10**i
    p = Popen(["./ht", str(num_trials), str(batch_size), basedir + f, str(uniq), outfile])
    p.wait()
