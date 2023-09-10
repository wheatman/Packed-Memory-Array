# Optimized Packed Memory Array

This repo contains the code for an optimized packed memory array.

It contains 3 major optimization.  

1. Search Optimization.  Improving the search also speeds up inserts since inserts must perform a search to know where the element is being inserted.
2. Compression.  We can compress the elements so that the entire PMA takes much less space.  Also this helps the performance of several operations that were previously memory bound since we reduce the memory traffic requirements.
3. Batch updates.  A new algorithm which algorithmic support for batch updates enable these updates to be much faster and to be parallel.

## Compile
The library itself is header only, so you can just `#include` "[CPMA.h](include/PMA/CPMA.hpp)".  

You will need to download the submodules with
```
git submodule init
git submodule update
```

To build the testing code just use `make basic`.  By default, it will build the code in many different configurations.  These allow you to choose the following attributes

1. element size
2. compressed or uncompressed
3. the head structure to use

For example to build with compression, 64 bit elements, and `Eytzinger` heads build with 
```
make build/basic_uint64_t_delta_compressed_Eytzinger
```

You can also build with support for parallelism using either [parlaylib](https://cmuparlay.github.io/parlaylib/) or [cilk](https://www.opencilk.org/) with 

```
make build/basic_uint64_t_delta_compressed_Eytzinger CILK=1
```

```
make build/basic_uint64_t_delta_compressed_Eytzinger PARLAY=1
```

This with build a binary with the same name in the build folder which can then run the tests.

## Running the tests

### Tests for [SPMA](https://epubs.siam.org/doi/pdf/10.1137/1.9781611977561.ch13)

#### Mixed Insert Search Workload (Figure 2, Table 2)

```
basic_uint64_t_uncompressed_Eytzinger ycsb_a <filename>
```
The file should be of the form 
```
<INSERT/READ> <element #>
```

#### Parallel Search Throughput (Figure 3, Table 4)
```
basic_uint64_t_uncompressed_Eytzinger find <num_elements_start> <num_searches>
```

#### Mixed Insert Range Query Workload (Figure 5, Table 3)
```
basic_uint64_t_uncompressed_Eytzinger ycsb_e <filename>
```
The file should be of the form 
```
INSERT <element #> [
SCAN  <element #> <length>
```

#### Serial Insert Throughput, Uniform Random (Figure 7, Table 5)
```
basic_uint64_t_uncompressed_Eytzinger single <num_elements>
```
This will also print out how big the structure was at the end

#### Serial Insert Throughput, Partially sorted (Figure 8, Table 6)
```
basic_uint64_t_uncompressed_Eytzinger single_alt <num_elements> <percent ordered>
```

#### Parallel Scan (Figure 9, Table 7)
```
basic_uint64_t_uncompressed_Eytzinger scan <num_elements> <number of bits to use for the elements>
```

This also prints out the size of the total structure as well as the size used by the head structure

#### Range Queires (Figure 10, Table 8)
```
basic_uint64_t_uncompressed_Eytzinger map_range <num_elements> <num_ranges> <number of bits to use for the range size>
```


### Insert Throughput 
To run a test and time how long it takes to insert edges run with 

```
basic_uint64_t_delta_compressed_Eytzinger batch <num elements start> <batch size> <trials>
```


### Range Map
To run a test and time how long it takes to map a basic function over different sized ranges 

```
basic_uint64_t_delta_compressed_Eytzinger map_range <num elements start> <number of ranges> <log of max range size> <trials>
```

### Graph Queries
```
basic_uint64_t_delta_compressed_Eytzinger graph <path to graph> <trials> <start_node> <max batch to insert>
```