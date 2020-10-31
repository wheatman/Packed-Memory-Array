# Optimized Packed Memory Array

This repo contains the code for an optimized packed memory array.

It contains 3 major optimization.  

1. Search Optimization.  Improving the search also speeds up inserts since inserts must perform a search to know where the element is being inserted.
2. Compression.  We can compress the elements so that the entire PMA takes much less space.  Also this helps the performance of several operations that were previously memory bound since we reduce the memory traffic requirements.
3. Batch updates.  A new algorithm which algorithmic support for batch updates enable these updates to be much faster and to be parallel.

## Compile
The library itself is header only, so you can just `#include` it.  To build the testing code just use `make basic`.  By default, it will build the code in many configurations.  These allow you to choose the following attributes

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
