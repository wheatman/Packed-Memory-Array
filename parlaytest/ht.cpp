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
void test_ht_from_data(std::vector<T> input, char* filename, uint64_t batch_size, int num_uniq, char* outfilename, double expansion_factor = 4, int num_trials = 5) {
  uint64_t max_size = num_uniq;

  std::cout << "inserting " << input.size() << " elts in batch size " << batch_size << std::endl;
  uint64_t start, end;

  // first param is size, last param is load
  // HT size is = 100 + static_cast<size_t>(load * size),
  uint64_t insert_time = 0, delete_time = 0;
  for(int i = 0; i < num_trials; i++) {
    parlay::hashtable<parlay::hash_numeric<T>>
      table(max_size, parlay::hash_numeric<T>{}, expansion_factor);

    // printf("inserting %lu elts, batch size %lu\n", max_size, batch_size);

    uint64_t start_idx = 0;
    
    // do insert
    start = get_usecs();
    while(start_idx < input.size()) {
      uint64_t end_idx = std::min(start_idx + batch_size, max_size - 1);
      
      /*
      parallel_for(uint64_t idx = start_idx; idx < end_idx; idx++) {
        table.insert(input[idx]);
      }
      */
      parlay::parallel_for(start_idx, end_idx, [&](T i) {
        table.insert(input[i]);
      });
      start_idx += batch_size;
    }
    end = get_usecs();
    insert_time += end - start;

    // do delete
    start_idx = 0;
    start = get_usecs();
    while(start_idx < input.size()) {
      uint64_t end_idx = std::min(start_idx + batch_size, max_size - 1);
      
      /*
      parallel_for(uint64_t idx = start_idx; idx < end_idx; idx++) {
        table.deleteVal(input[idx]);
      }
      */
      parlay::parallel_for(start_idx, end_idx, [&](T i) {
        table.deleteVal(input[i]);
      });
      start_idx += batch_size;
    }
    end = get_usecs();
    delete_time += end - start;
  }

  double avg_delete = ((double) delete_time / 1000000) / num_trials;
  double avg_insert = ((double) insert_time / 1000000) / num_trials;
  std::cout << "avg insert " << avg_insert << ", avg delete " << avg_delete << std::endl;

  // do sum
  // init table for sum
  parlay::hashtable<parlay::hash_numeric<T>>
    table(max_size, parlay::hash_numeric<T>{}, expansion_factor);

  /*
  parallel_for(uint64_t idx = 0; idx < input.size(); idx++) {
    table.insert(input[idx]);
  }
  */
  parlay::parallel_for(0, input.size(), [&](T i) {
    table.insert(input[i]);
  });

  uint64_t sum;
  uint64_t sum_time = 0;
  for(int i = 0; i < num_trials; i++) {
    start = get_usecs();
    sum = table.sum();
    end = get_usecs();
    // std::cout << "got sum " << sum << std::endl;
    sum_time += end - start;
  }
  double avg_sum_time = ((double) sum_time / 1000000) / num_trials;
  std::cout << "avg sum time: " << avg_sum_time << std::endl;

  // get correct sum
#ifdef VERIFY
  start = get_usecs();
  auto elts = table.entries();
  uint64_t correct_sum = parlay::reduce(elts);
  end = get_usecs();

  double correct_sum_time = ((double) (end - start) / 1000000);
  std::cout << "correct sum time: " << correct_sum_time << std::endl;
  printf("correct sum %lu, got sum %lu, as 32 %u\n", correct_sum, sum, (uint32_t) sum);

  /*
  std::set<T> correct;
  for(auto i : input) {
      correct.insert(i);
  }
  uint64_t target_sum = 0;
  for(auto e : correct) {
    target_sum += e;
  }
  printf("target sum from set: %lu\n", target_sum);
  */

  // verify that all the elemnts are actually in there 
  parlay::parallel_for(0, max_size - 1, [&](T i) {
    auto val = table.find(input[i]);
    if (val != input[i]) { printf("idx %d, should be %u, got %u\n", i, input[i], val); }
    assert(val == input[i]); 
  }); 

  assert(correct_sum == (uint32_t) sum);
#endif

  // write out to file
  std::ofstream outfile;
  outfile.open(outfilename, std::ios_base::app);
  outfile << filename << "," << batch_size << "," << avg_insert << "," << avg_delete << "," << avg_sum_time << std::endl;
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
  test_ht_from_data<uint32_t>(data, filename, batch_size, num_uniq, small_outfilename, 1, num_trials);
  

  // medium
  /*
  char med_outfilename[80];
  strcpy(med_outfilename, "ht_med_");
  strcat(med_outfilename, outfilename);
  printf("\n***expansion = 1.5***\n");
  test_ht_from_data<uint32_t>(data, filename, batch_size, num_uniq, med_outfilename, 1.5, num_trials);
  */

  // half full
  printf("\n***expansion = 2***\n");
  char big_outfilename[80];
  strcpy(big_outfilename, "ht_big_");
  strcat(big_outfilename, outfilename);
  test_ht_from_data<uint32_t>(data, filename, batch_size, num_uniq, big_outfilename, 2, num_trials);
  
  /*
  //almost empty
  printf("\n***expansion = 4***\n");
  test_ht_from_data<uint32_t>(data, batch_size, 4);

  //almost empty
  printf("\n***expansion = 8***\n");
  test_ht_from_data<uint32_t>(data, batch_size, 8);
  */

  //almost empty
  // printf("\n***expansion = 20***\n");
  // test_ht_from_data<uint32_t>(data, batch_size, 20, num_trials);
}
