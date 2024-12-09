# Optimized Packed Memory Array

This repo contains the code for an optimized packed memory array (PMA).

It contains 3 major optimizations.  

1. Search Optimization.  Improving the search also speeds up inserts since inserts must perform a search to know where the element is being inserted.  We call the PMA with search optimizations an SPMA
2. Compression.  We can compress the elements so that the entire PMA takes much less space.  Also, this helps the performance of several operations that were previously memory bound since we reduce the memory traffic requirements.  We call a compressed pma a CPMA.
3. Batch updates.  A new algorithm with algorithmic support for batch updates enables these updates to be much faster and to be parallel.

It also contains the wrapper code to convert the PMA into a graph data structure, PCSR

This README file is broken into two parts, the first is using the Packed Memory Array in general and as part of other systems.  The second is how to run specific benchmarks that were used to test and evaluate the behavior of the PMA, CPMA, and SPMA.

To easily run and use the PMA as a graph data structure I recommend using the [BYO](https://github.com/wheatman/BYO) framework, which has many variations of these PMAs already integrated into the graph benchmark.

## Using the different variants of PMA's
The library itself is header only, so you can just `#include` "[CPMA.hpp](include/PMA/CPMA.hpp)".  

You will need to download the submodules with
```
git submodule init
git submodule update
```

The graph data structure can similarly be included with `#include` "[PCSR.hpp](include/PMA/PCSR.hpp)".

### PMA Template Parameters

When using a PMA you must select at compile time which variant of the PMA you want to use.  This is done with the template parameter.  The Parameter is of type PMA_traits and allows the selection of many different characteristics.  

There are a bunch of predefined ones for each of use, these can be found near the top of [CPMA.h](include/PMA/CPMA.hpp).  A few of the major ones are.

 - `pma_settings`: which is an uncompressed PMA without any search optimization.
 - `spmae_settings`: which is an uncompressed PMA with a search optimization with the heads stored in Eytzinger order.  This will be much faster for point queries, point updates, and small batch insertions.  There may be a small slowdown in some scan operations. 
 - `cpma_settings`: which is a compressed PMA without the search optimization
- `scpmae_settings`: which is a compressed PMA with a search optimization with the heads stored in Eytzinger order. This will be much faster for point queries, point updates, and small batch insertions.  There may be a small slowdown in some scan operations.

If you want you can also make your own version of the traits class to select your own set of options.

The options are as follows, as a note some of the options are not fully supported and should not be selected.

- `l`: leaf type, this selects which type of PMA leaf to use as a base.  The two currently supported are `uncompressed_leaf` and `delta_compressed_leaf`, but additional ones could be added.  These options can be further templated by the type of the individual elements.
- `h`: HeadForm, this selects what type of heads to use which corresponds to what type of search optimization to use.  The options are 
    - `InPlace`: the heads are stored with the leafs which give no search optimization
    - `Linear`: The heads are stored together, but in the same order as the heads
    - `Eytzinger`: The heads are stored together in Eytzinger order which is normally the fastest in practice.
    - `BNary`: The heads are stored together in a static b tree like structure, this has some theoretical benefits of Eytzinger order but in practice does not gain much benefit.  
 - `b`: If BNary heads are selected then this argument should select what size b to use.  It should be 1 more than a small power of 2, it has been tested with 5, 9, and 17.  If the heads are anything but BNary then this field should be 0.
 - the next two fields, `density` and `rank` should be false
 - `fixed_size`: This boolean specifies that the entire PMA should be stored in a fixed size memory blob on the stack.  This can be useful if you want to use the PMA for something like the nodes in another data structure.
 - `max_fixed_size`: Specifies how big the PMA is able to grow to if it is being stored in a fixed size blob.
 - `parallel_`: specifies that the pma is able to parallelize its internal operations 
 - `maintain_offsets`: This is used to turn the PMA into the PCSR graph data structure. I highly recommend using the through PCSR.hpp and not directly. 


 Some examples of defining a PMA from scratch can be found in the definitions of the predefined ones near the top of [CPMA.hpp](include/PMA/CPMA.hpp).

 ### PMA define options

 In addition there are a few compile time defines that impact the behavior of the PMA, these are
- `DEBUG`: which adds a significant amount of verification of the internal operations.  When this is enabled it will be much slower and no longer follow asymptotic bounds.
- `CILK` and `PARLAY`: these are used by the ParallelTools library, one of the submodules to control the parallelization scheme, only one should be enabled at any time, and if it is enabled it will be used to parallelize the internal operations.  
- `NO_TLX`: If this is defined a few internal maps will use `std::map` instead of `tlx::btree_map` which can slow down some redistributes for the benefit of having less dependencies.


### PMA API

- `uint64_t size()` : The number of elements being stored in the PMA
- `CPMA()`: construct an empty PMA
- `CPMA(key_type *start, key_type *end)`: construct a PMA with the elements in the given range
- `bool has(key_type e)`: return true if the key `e` is in the PMA
- `bool insert(element_type e)`: inserts the element e into the PMA, returns false if the key was already there.
- `uint64_t insert_batch(element_ptr_type e, uint64_t batch_size, bool sorted = false)`: inserts a batch of elements of size batch_size
- `uint64_t remove_batch(key_type *e, uint64_t batch_size, bool sorted = false)`: removes a batch of elements
- `bool remove(key_type e)`: removes the element with key `e`
- `uint64_t get_size()`: returns the amount of memory in bytes used by the PMA
- `uint64_t sum()`: Returns the sum of all elements in the PMA
- `key_type max()` / `min()`: returns the smallest or largest key stored in the PMA
- `bool map(F f)`: runs function f on all elements in the pma
- `parallel_map(F f)`: runs function f on all elements in the pma in parallel
- `bool map_range(F f, key_type start_key, key_type end_key)`: runs function f on all elements with keys between start_key and end_key
- `uint64_t map_range_length(F f, key_type start, uint64_t length)`: runs function f on at most length elements starting from key at least start
- The PMA also supports iteration as it has begin and end functions so you can perform operations like `for (auto el : pma)`. Note that this may be slower than using the map functions 



### PCSR Template Parameters
The PCSR settings are the same as the standard PMA settings though it sets the argument maintain_offsets to true.  For simplicity a few have already been defined.

- `simple_pcsr_settings`: which stores an unweighted undirected graph and takes in the single template parameter of which type to store the vertex ids as
- `simple_wpcsr_settings`: which stores a weighted undirected graph and takes in both the vertex id type and the weight type

A few notes:
- Compression does not work with PCSR
- The top bit from the vertex id is used by the system, so if you have more than 2^31 vertices then you need to use 64 bit vertex ids
- The graphs are stored undirected, if you want to have directed graphs simply use 2 PCSRs

### PCSR API
- `PCSR(T num_nodes)`: construct an empty PCSR with the specified number of nodes
- `PCSR(T num_nodes, R &edges, bool sorted = false, Projection projection = {})`: construct a PCSR with the specified number of nodes with the set of edges.  This can be faster than constructing an empty one and then inserting the edges as a batch.  `edges` is an arbitrary random access range.  `sorted` is if the edges are already sorted, first by source and then by dest.  `projection` allows a conversion function to be applied to each edge to make it appear as a tuple like type.
- `contains(T src, T dest)`: returns if the edge is in the structure 
- `insert(T src, T dest)`: adds the edge to the strucure and returns if it was added, it is not added if it is already there
- `insert(T src, T dest, value_type val)`: adds a weighted edge.  There is a template parameter to the entire strcuture which allows specifying how to combine weights if an edge is inserted that is already there, by default the old weight is overwritten
- `remove(T src, T dest)`: remove the specified edge
- `insert_batch(R &es, bool sorted = false, Projection projection = {})`: add a batch of edges. `sorted` is if the edges are already sorted, first by source and then by dest.  `projection` allows a conversion function to be applied to each edge to make it appear as a tuple like type.
- `remove_batch(R &es, bool sorted = false)`: removes a batch of edges. `sorted` is if the edges are already sorted, first by source and then by dest.
- `get_memory_size()`: returns the total memory usage in bytes of the structure
- `template <int early_exit, class F> void map_neighbors(uint64_t i, F f, unused, bool run_parallel)`: runs the given function on all edges of the given vertex. You can specify if early exiting from the map is possible along with if it should be run in parallel. 
- write_adj_file(filename): writes the graph being stored out to the given file in adj format


## Compile

This library requires some c++-20 support. It has been tested and works for g++ 11.4 and clang++ 14 and should work for later ones as well.

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

#### Range Queries (Figure 10, Table 8)
```
basic_uint64_t_uncompressed_Eytzinger map_range <num_elements> <num_ranges> <number of bits to use for the range size>
```


#### Insert Throughput 
To run a test and time how long it takes to insert edges run with 

```
basic_uint64_t_delta_compressed_Eytzinger batch <num elements start> <batch size> <trials>
```


#### Range Map
To run a test and time how long it takes to map a basic function over different sized ranges 

```
basic_uint64_t_delta_compressed_Eytzinger map_range <num elements start> <number of ranges> <log of max range size> <trials>
```

#### Graph Queries
```
basic_uint64_t_delta_compressed_Eytzinger graph <path to graph> <trials> <start_node> <max batch to insert>
```


### Tests for CPMA

#### Batch Insert/Delete throughput (Figure 1, Table 3, Table 4)

First you need to build the PMA with 
```
make build/basic_uint64_t_uncompressed_Eytzinger PARLAY=1 -B
```
Then run the test with

```
./build/basic_uint64_t_uncompressed_Eytzinger batch_bench <num_elements_start>
```

Then do the same for the CPMA with 
```
make build/basic_uint64_t_delta_compressed_Eytzinger PARLAY=1 -B
./build/basic_uint64_t_delta_compressed_Eytzinger batch_bench <num_elements_start>
```

To get the serial numbers rebuild without the flag `PARLAY=1` and run the test again

```
make build/basic_uint64_t_uncompressed_Eytzinger -B
./build/basic_uint64_t_uncompressed_Eytzinger batch_bench <num_elements_start>
```

The above tests will run with 40 bit uniform random numbers.  You can also test a zipfian distribution for a skewed input distribution by using the command
```
./build/basic_uint64_t_uncompressed_Eytzinger batch_bench_zip <num_elements_start> <trials> <verify> <alpha>
```


#### Range query throughput (Figure 2)

First for the PMA
```
make build/basic_uint64_t_uncompressed_Eytzinger PARLAY=1 -B
./build/basic_uint64_t_uncompressed_Eytzinger map_range <num_elements_start> <num_ranges = 100000> <max_log_range_size>
```

The same can be done for the CPMA
```
make build/basic_uint64_t_delta_compressed_Eytzinger PARLAY=1 -B
./build/basic_uint64_t_delta_compressed_Eytzinger map_range <num_elements_start> <num_ranges = 100000> <max_log_range_size>
```


#### Throughput of Serial and parallel batch inserts (Table 2)

You first need to build the PMA in serial mode with, not the lack of PARLAY=1 or CILK=1.  This requires a rebuild 
```
make build/basic_uint64_t_uncompressed_Eytzinger -B
```
Then run the same batch insert command from before 
```
./build/basic_uint64_t_uncompressed_Eytzinger batch_bench <num_elements_start>
```

The remaining items in the table can be gotten from the batch insert test described above.

#### Scalability of batch inserts (Figure 7)
To get the serial numbers (1 core) build as we did above 
```
make build/basic_uint64_t_uncompressed_Eytzinger -B
make build/basic_uint64_t_delta_compressed_Eytzinger -B
```
These only use a single batch size so we can use the command 

```
./build/basic_uint64_t_uncompressed_Eytzinger batch <num_elements_start = 100000000> <batch_size = 1000000>

./build/basic_uint64_t_delta_compressed_Eytzinger batch <num_elements_start = 100000000> <batch_size = 1000000>
```

Then after that we want to rebuild in parallel mode 

```
make build/basic_uint64_t_uncompressed_Eytzinger -B PARLAY=1
make build/basic_uint64_t_delta_compressed_Eytzinger -B PARLAY=1
```

And run the command for the various numbers of threads

PARLAY_NUM_THREADS

```
PARLAY_NUM_THREADS = 2 ./build/basic_uint64_t_uncompressed_Eytzinger batch <num_elements_start = 100000000> <batch_size = 1000000>
PARLAY_NUM_THREADS = 4 ./build/basic_uint64_t_uncompressed_Eytzinger batch <num_elements_start = 100000000> <batch_size = 1000000>
PARLAY_NUM_THREADS = 8 ./build/basic_uint64_t_uncompressed_Eytzinger batch <num_elements_start = 100000000> <batch_size = 1000000>
...

PARLAY_NUM_THREADS = 2 ./build/basic_uint64_t_delta_compressed_Eytzinger batch <num_elements_start = 100000000> <batch_size = 1000000>
...
```

#### Scalability of map_range (Figure 8)
We can do the same thing as above to test the scalability of map_range

you will need to build the binaries in both serial and parallel mode and specify the environment variable for the number of threads again.

The command to just run map range with a single setting is in the form
```
basic_uint64_t_uncompressed_Eytzinger map_range_single <num_elements_start = 100000000> <num_ranges = 100000> <range_size = 17179869184>
```


#### Space usage (Table 5)

The space usage will print out after some of the tests.  The best one to use is the single insertion tests.  Due to the amortized nature of the structure it is possible that depending on the insertion pattern the structure will be slightly different sizes depending on how the elements were inserted, hence recommending using the single insert test to build the structure to measure size.  The command is 
```
./build/basic_uint64_t_uncompressed_Eytzinger single <num elements>
```

#### Running Graph Algorithms and insert (Figure 9, Figure 10, Table 6)
Graph Algorithms can be run with
```
./build/basic_uint64_t_delta_compressed_Eytzinger graph <path to graph> <trials> <start_node> <max batch to insert>
```

This will print out the size of the graph, how long it took to run the different algorithms, and how long it took to perform batch inserts into it.


### RMA batch insert experiment (Table 3)

The serial batch insert experiments for RMA can be run with `other_systems/rma/batch_insert.sh`

This may require changing around some settings about huge pages which can be done with the following commands run with sudo
```
echo 4294967296 > /proc/sys/vm/nr_overcommit_hugepages
echo 1 > /proc/sys/vm/overcommit_memory
```
No other parts of the evaluation require any sort of privilege access.


### Micro benchmarks for C-Pac U-Pac and PAM (Figure 1, Figure 2, Table 5, Table 8, Table 9)

To run the tests for U-Pac and C-Pac cd into `other_systems/CPAM/examples/microbenchmarks`
the run run_tests.sh

This will build and run the following tests.

batch inserts for all three systems with both uniform and zipfian distributions.

Size tests for C-Pac and U-Pac

Map range tests for all three systems.

### Graph Evaluation for C-PaC

To run the graph benchmarks for C-PaC cd into `other_systems/CPAM/examples/graphs/algorithms` 
build with `make`
Then run all of the graphs and algorithms with the run_all.sh script.  You will need to edit this script to point at the locations of the graphs on your machine.  The file format is the same as it is for the PMA graph file format.

To run the graph batch updates go into `other_systems/CPAM/examples/graphs/run_batch_updates` 
build with `make`

Then run the test with 
```
./run_batch_updates-CPAM-CPAM-Diff -s <path to graph>
```

### Tests for PCSR
All of the PCSR experiments were run through [BYO](https://github.com/wheatman/BYO)

There is a simpler example test driver for PCSR that can be found at [run_pcsr.cpp](run_pcsr.cpp).  That will make a graph, run a few algorithms and perform some batch inserts.  However, the algorithm times will be slower than using the BYO framework do to worse algorithm implementations. 


### Graph Evaluation for Aspen
the script to run all the experiments for aspen can be found at `other_systems/aspen/code/run-all.sh`  this will test running on all of the graphs as well as batch updates.

### Graph File Format
The graph file format is .adj described at [https://www.cs.cmu.edu/~pbbs/benchmarks/graphIO.html](https://www.cs.cmu.edu/~pbbs/benchmarks/graphIO.html). For utilities including converters from other file formats see [https://github.com/jshun/ligra/tree/master](https://github.com/jshun/ligra/tree/master) 





# Citations 

For the search optimization please cite

```
Wheatman, B., Burns, R., Buluç, A., & Xu, H. (2023). Optimizing Search Layouts in Packed Memory Arrays. In 2023 Proceedings of the Symposium on Algorithm Engineering and Experiments (ALENEX) (pp. 148-161). Society for Industrial and Applied Mathematics.
```

bibtex
```
@inproceedings{wheatman2023optimizing,
  title={Optimizing Search Layouts in Packed Memory Arrays},
  author={Wheatman, Brian and Burns, Randal and Bulu{\c{c}}, Ayd{\i}n and Xu, Helen},
  booktitle={2023 Proceedings of the Symposium on Algorithm Engineering and Experiments (ALENEX)},
  pages={148--161},
  year={2023},
  organization={SIAM}
}
```

For the compression or the batch parallel inserts optimizations please cite 

```
Wheatman, B., Burns, R., Buluc, A., & Xu, H. (2024, March). CPMA: An efficient batch-parallel compressed set without pointers. In Proceedings of the 29th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (pp. 348-363).
```

bibtex
```
@inproceedings{wheatman2024cpma,
  title={CPMA: An efficient batch-parallel compressed set without pointers},
  author={Wheatman, Brian and Burns, Randal and Buluc, Aydin and Xu, Helen},
  booktitle={Proceedings of the 29th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming},
  pages={348--363},
  year={2024}
}
```

For PCSR please cite 
```
Brian Wheatman, Randal Burns, and Helen Xu, Batch-Parallel Compressed Sparse Row: A Locality-Optimized Dynamic-Graph Representation 2024 IEEE High Performance Extreme Computing Conference (HPEC). IEEE, 2024
```
bibtex
```
@inproceedings{wheatman2024bpcsr,
  title={Batch-Parallel Compressed Sparse Row: A Locality-Optimized Dynamic-Graph Representation},
  author={Wheatman, Brian and Burns, Randal and Xu, Helen},
  booktitle={2024 IEEE High Performance Extreme Computing Conference (HPEC)},
  year={2024}
}