#include "../zipf.hpp"

#include <limits>
#include <random>
#include <fstream>
#include <algorithm>  // generate
#include <functional> // bind
#include <iostream>   // cout
#include <iterator>   // begin, end, and ostream_iterator
#include <random>     // mt19937 and uniform_int_distribution
#include <vector>     // vector
#include <string>

template <class T>
void write_data_to_file(std::vector<T> data, char* filename) {
  std::ofstream outfile;
  outfile.open(filename);
  for(const auto &e : data) { outfile << e << "\n"; }
  outfile.close();
}

template <class T>
void gen_zipf(uint64_t max_size) {
  std::random_device r;
  std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};

  // zipf distributions
  // for (uint32_t max = 1 << 22U; max < 1 << 27; max *= 8) {
  uint32_t max = 1 << 26;
    // for (double alpha = 1; alpha < 2.1; alpha += 1) {
    for (double alpha = 0.4; alpha < 1.0; alpha += 0.2) {

      std::cout << "ZipF: max = " << max << " alpha = " << alpha << std::endl;
      zipf zip(max, alpha, seed);
      std::vector<T> data;
      for (uint32_t i = 0; i < max_size; i++) {
        data.push_back(zip.gen());        
      }
      char filename[50];
      sprintf(filename, "zipf_%u_%f_%lu", max, alpha, max_size);
      write_data_to_file(data, filename);
    }
  // }
}


template <class T>
std::vector<T> create_random_data(size_t n, size_t max_val,
                                  std::seed_seq &seed) {

  std::mt19937 eng(seed); // a source of random data

  std::uniform_int_distribution<T> dist(0, max_val);
  std::vector<T> v(n);

  generate(begin(v), end(v), bind(dist, eng));
  return v;
}



template <class T>
void gen_uniform_random(uint64_t& max_size, char* filename) {
  std::random_device r;
  std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};

  auto data = create_random_data<T>(max_size, std::numeric_limits<T>::max(), seed);

  write_data_to_file(data, filename);
}

int main(int32_t argc, char *argv[]) {
  assert(argc == 4);
  // argv[1] = type of data to create
    // 1 = uniform, 2 = zipf
  // argv[2] = number of points to make
  // argv[3] = name of output file to write to
  uint64_t type = atoi(argv[1]);
  uint64_t n = atoi(argv[2]);
  char* filename = argv[3];
  if (type == 1) {
    gen_uniform_random<uint32_t>(n, filename);    
  } else {
    gen_zipf<uint32_t>(n);
  }

}
