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
void test_pam_from_data( std::vector<T> input, int num_trials, char* filename, char* outfilename, int serial) {
  uint64_t start, end;

  struct set_entry {
    using key_t = T;
    static inline bool comp(key_t a, key_t b) { return a < b;}
  };
  using set = pam_set<set_entry>;

  timer t;
  double total_insert_time = 0.0, total_sum_time = 0.0;
  for(int i = 0; i < num_trials; i++) {
    auto e = parlay::sequence<T>();

    // base set
    set s(e);



    // base set
    T target_sum = 0;
    // TODO: is this the right way to do batch inserts? 
    
    // put all data in set
    
    t.start();
    auto batch = parlay::sequence<T>(input.begin(), input.end());
    // set s(batch);
    t.start();
    s = set::multi_insert(s, batch);
    total_insert_time += t.stop();

    auto f = [&] (T e) {return e;};

    double sum_time = 0.0;
    // run sum
      // do sum 
    t.start();
    auto result = set::map_reduce(s, f, parlay::Add<T>());
    total_sum_time += t.stop();
    
    cout << "got sum = " << result << endl; 
  }
  auto avg_insert = total_insert_time / num_trials;
  auto avg_sum = total_sum_time / num_trials;
 
  // write out to file
  std::ofstream outfile;
  outfile.open(outfilename, std::ios_base::app);
  if (serial) {
    outfile << "," << avg_insert << "," << avg_sum << std::endl;
  } else {
    outfile << filename << "," << avg_insert << "," << avg_sum;
  }
  outfile.close();
}


int main(int32_t argc, char *argv[]) {
  // first arg is max size
  assert(argc >= 3);
  int num_trials = atoi(argv[1]);
  char* filename = argv[2];
  char* outfilename = argv[3];
  int serial = atoi(argv[4]);  
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
  test_pam_from_data<uint32_t>(data, num_trials, filename, outfilename, serial);
}
