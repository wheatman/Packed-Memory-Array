#include <pam/pam.h>
/*
#include <parlay/hash_table.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include "../helpers.h"
#include <limits>
#include <random>
#include <sys/time.h>
*/
#include "parlay/primitives.h"
#include <string.h>
#include <fstream>
#include <stdio.h>

#include "../io_util.h"
// take input as a vector
template <class T>
void test_pam_from_data(uint64_t batch_size, std::vector<T> input, int num_trials, char* filename, char* outfilename) {
  printf("test from data, batch size %lu, input size %lu\n", batch_size, input.size());
  uint64_t start, end;

  struct set_entry {
    using key_t = T;
    static inline bool comp(key_t a, key_t b) { return a < b;}
  };
  using set = pam_set<set_entry>;

  timer t;
  double total_insert_time = 0.0, total_delete_time = 0.0;
  for(int i = 0; i < num_trials; i++) {
    auto e = parlay::sequence<T>();

    // base set
    set s(e);

    T target_sum = 0;
    
    start = 0;
    t.start();

    // do inserts
    while (start < input.size()) {
      end = start + batch_size;
      end = std::min((uint64_t)input.size(), end);
      // printf("start %lu, end %lu\n", start, end);

      assert(start < input.size() && end <= input.size());
      auto batch = parlay::sequence<T>(input.begin() + start, input.begin() + end);

      s = set::multi_insert(s, batch); 
      start += batch_size;
    }
    total_insert_time += t.stop();

    // do deletes
    start = 0;
    t.start();
    while (start < input.size()) {
      end = start + batch_size;
      end = std::min((uint64_t)input.size(), end);
      // printf("start %lu, end %lu\n", start, end);

      assert(start < input.size() && end <= input.size());
      auto batch = parlay::sequence<T>(input.begin() + start, input.begin() + end);

      s = set::multi_delete(s, batch); 
      start += batch_size;
    }
    total_delete_time += t.stop();
  }
  // cout << "total insert time = " << total_insert_time << endl;
  auto avg_insert = total_insert_time / num_trials;
  auto avg_delete = total_delete_time / num_trials;
  // auto avg_throughput = input.size() / avg_insert;
  cout << "avg insert = " << avg_insert << endl;
  // cout << "avg_throughput = " << avg_throughput << endl;
  cout << "avg delete = " << avg_delete << endl;

  // *** do sum ***

  // base set
  T target_sum = 0;
  // TODO: is this the right way to do batch inserts? 
  
  // put all data in set
  auto batch = parlay::sequence<T>(input.begin(), input.end());
  set s(batch);
  // s = set::multi_insert(s, batch);
  auto f = [&] (T e) {return e;};
  t.reset();

  double sum_time = 0.0;
  // run sum
  for(int i = 0; i < num_trials; i++) {
    // do sum 
    t.start();
    auto result = set::map_reduce(s, f, parlay::Add<T>());
    auto trial_time = t.stop();
    
    sum_time += trial_time;

    cout << "got sum = " << result << endl; 
  }
  auto avg_sum = sum_time / num_trials;
  cout << "avg sum time = " << avg_sum << endl;

  std::ofstream outfile;
  outfile.open(outfilename, std::ios_base::app);
  outfile << filename << "," << batch_size << "," << avg_insert << "," << avg_delete << "," << avg_sum << endl;
  outfile.close();

  // output space
  // set::GC::print_stats();
}


int main(int32_t argc, char *argv[]) {
  // first arg is max size
  assert(argc >= 3);
  // uint64_t max_size = atoi(argv[1]);
  uint64_t batch_size = atoi(argv[2]);
  int num_trials = atoi(argv[1]);
  /*
  if (argc == 3) {
    test_pam_ordered_insert<uint32_t>(batch_size, max_size);
  } else {
  */
  // second arg is input file if there is one
  char* filename = argv[3];
  char* outfilename = argv[4];
  auto data = get_data_from_file(filename);

  /*
  std::vector<uint32_t> data;
  std::ifstream input_file(filename);

  // read in data from file
  uint32_t temp;
  for( std::string line; getline( input_file, line ); ) {
    std::istringstream iss(line);
    if(!(iss >> temp)) { break; }
    data.push_back(temp);
  }
  */
  cout << "\n" << filename << endl;
  test_pam_from_data<uint32_t>(batch_size, data, num_trials, filename, outfilename);
}
