#include <parlay/hash_table.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include "../helpers.h"
#include <limits>
#include <random>
#include <sys/time.h>
#include <set>
// #include "../io_util.h"
// #define VERIFY

template <class T>
void test_ht_from_data(std::vector<T> input, char* filename, int num_uniq, char* outfilename, double expansion_factor = 4, int num_trials = 5, int serial = 0) {
  uint64_t max_size = num_uniq;

  uint64_t start, end;

  // first param is size, last param is load
  // HT size is = 100 + static_cast<size_t>(load * size),
  uint64_t insert_time = 0, sum_time = 0;;
  for(int i = 0; i < num_trials; i++) {
    parlay::hashtable<parlay::hash_numeric<T>>
      table(max_size, parlay::hash_numeric<T>{}, expansion_factor);

    // do insert
    start = get_usecs();
    parlay::parallel_for(0, input.size(), [&](T i) {
      table.insert(input[i]);
    });

    end = get_usecs();
    insert_time += end - start;

    start = get_usecs();
    auto sum = table.sum();
    end = get_usecs();
    printf("got sum %lu\n", sum);
    sum_time += end - start;
  }

  double avg_insert = ((double) insert_time / 1000000) / num_trials;
  double avg_sum = ((double) sum_time / 1000000) / num_trials;

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
  int num_trials = atoi(argv[1]);
  uint64_t batch_size = atoi(argv[2]);

  // double max_expansion = 8;
  // double expansion = 1;
  // test_ht_ordered_insert<int>(max_size, expansion);

  // second arg is input file if there is one
  char* filename = argv[3];
  int num_uniq = atoi(argv[4]);
  char* outfilename = argv[5];
  int serial = atoi(argv[6]);
  
  // auto data = get_data_from_file(filename);
  std::vector<uint32_t> data;
  std::ifstream input_file(filename);

  // read in data from file
  uint32_t temp;
  for( std::string line; getline( input_file, line ); ) {
    std::istringstream iss(line);
    if(!(iss >> temp)) { break; }
    data.push_back(temp);
  }
  std::cout << filename << std::endl;

  // almost full
  char small_outfilename[80];
  strcpy(small_outfilename, "ht_small_");
  strcat(small_outfilename, outfilename);
  printf("\n***expansion = 1***\n");
  test_ht_from_data<uint32_t>(data, filename, num_uniq, small_outfilename, 1, num_trials, serial);
  

  // half full
  printf("\n***expansion = 2***\n");
  char big_outfilename[80];
  strcpy(big_outfilename, "ht_big_");
  strcat(big_outfilename, outfilename);
  test_ht_from_data<uint32_t>(data, filename, num_uniq, big_outfilename, 2, num_trials, serial);
  
}
