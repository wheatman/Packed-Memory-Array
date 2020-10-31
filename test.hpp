#pragma once

#include "CPMA.hpp"
#include "ParallelTools/parallel.h"
#include "ParallelTools/reducer.h"
#include <cstdint>

#include "helpers.h"
#include "io_util.h"
#include "leaf.hpp"
#include "parlay/sequence.h"
#include "rmat_util.h"
#include "zipf.hpp"

#include "EdgeMapVertexMap/algorithms/BC.h"
#include "EdgeMapVertexMap/algorithms/BFS.h"
#include "EdgeMapVertexMap/algorithms/Components.h"
#include "EdgeMapVertexMap/algorithms/PageRank.h"

#include <iomanip>
#include <limits>
#include <set>
#include <sys/time.h>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include <algorithm> // generate
#include <iostream>  // cout
#include <iterator>  // begin, end, and ostream_iterator
#include <random>    // mt19937 and uniform_int_distribution
#include <vector>    // vector

#include "parlay/random.h"

#include "tlx/container/btree_map.hpp"
#include "tlx/container/btree_set.hpp"
// just redefines some things in the tlx btree so I can get access to the size
// of the nodes
#include "btree_size_helpers.hpp"

template <class T>
std::vector<T> create_random_data_with_seed(size_t n, size_t max_val,
                                            std::seed_seq &seed) {

  if constexpr (std::is_same_v<T, uint24_t>) {
    std::mt19937_64 eng(seed); // a source of random data

    std::uniform_int_distribution<uint64_t> dist(0,
                                                 std::min(max_val, 1UL << 24));
    std::vector<uint64_t> v(n);
    for (auto &el : v) {
      el = dist(eng);
    }
    std::vector<T> v2(v.begin(), v.end());
    return v2;
  } else {
    std::mt19937_64 eng(seed); // a source of random data

    std::uniform_int_distribution<T> dist(0, max_val);
    std::vector<T> v(n);
    for (auto &el : v) {
      el = dist(eng);
    }
    return v;
  }
}

template <class T> std::vector<T> create_random_data(size_t n, size_t max_val) {

  std::random_device rd;
  auto seed = rd();
  std::seed_seq s{seed};
  return create_random_data_with_seed<T>(n, max_val, s);
}
template <class T>
parlay::sequence<T> create_random_data_in_parallel(size_t n, size_t max_val,
                                                   uint64_t seed_add = 0) {

  auto v = parlay::sequence<T>::uninitialized(n);
  uint64_t per_worker = (n / ParallelTools::getWorkers()) + 1;
  ParallelTools::parallel_for(0, ParallelTools::getWorkers(), [&](uint64_t i) {
    uint64_t start = i * per_worker;
    uint64_t end = (i + 1) * per_worker;
    if (end > n) {
      end = n;
    }
    std::random_device rd;
    std::mt19937_64 eng(i + seed_add); // a source of random data

    std::uniform_int_distribution<uint64_t> dist(0, max_val);
    for (size_t j = start; j < end; j++) {
      v[j] = dist(eng);
    }
  });
  return v;
}

template <class T> void test_set_ordered_insert(uint64_t max_size) {
  if (max_size > std::numeric_limits<T>::max()) {
    max_size = std::numeric_limits<T>::max();
  }
  uint64_t start = 0;
  uint64_t end = 0;
  std::set<T> s;

  start = get_usecs();
  for (uint32_t i = 1; i < max_size; i++) {
    s.insert(i);
  }
  end = get_usecs();
  printf("insertion,\t %lu,\t", end - start);
  start = get_usecs();
  uint64_t sum = 0;
  for (auto el : s) {
    sum += el;
  }
  end = get_usecs();
  printf("sum_time, \t%lu, \tsum_total, \t%lu\n", end - start, sum);
}
template <class T>
void test_set_unordered_insert(uint64_t max_size, std::seed_seq &seed) {
  if (max_size > std::numeric_limits<T>::max()) {
    max_size = std::numeric_limits<T>::max();
  }
  uint64_t start = 0;
  uint64_t end = 0;
  std::set<T> s;
  std::vector<T> data = create_random_data_with_seed<T>(
      max_size, std::numeric_limits<T>::max(), seed);

  start = get_usecs();
  for (uint32_t i = 1; i < max_size; i++) {
    s.insert(data[i]);
  }
  end = get_usecs();
  printf("insertion,\t %lu,\t", end - start);
  start = get_usecs();
  uint64_t sum = 0;
  for (auto el : s) {
    sum += el;
  }
  end = get_usecs();
  printf("sum_time, \t%lu, \tsum_total, \t%lu\n", end - start, sum);
}
template <class T> void test_unordered_set_ordered_insert(uint64_t max_size) {
  if (max_size > std::numeric_limits<T>::max()) {
    max_size = std::numeric_limits<T>::max();
  }
  uint64_t start = 0;
  uint64_t end = 0;
  std::unordered_set<T> s;
  start = get_usecs();
  for (uint32_t i = 1; i < max_size; i++) {
    s.insert(i);
  }
  end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  start = get_usecs();
  uint64_t sum = 0;
  for (auto el : s) {
    sum += el;
  }
  end = get_usecs();
  printf("\tsum_time, \t%lu, \tsum_total, \t%lu\n", end - start, sum);
}
template <class T>
void test_unordered_set_unordered_insert(uint64_t max_size,
                                         std::seed_seq &seed) {
  if (max_size > std::numeric_limits<T>::max()) {
    max_size = std::numeric_limits<T>::max();
  }
  uint64_t start = 0;
  uint64_t end = 0;
  std::unordered_set<T> s;
  std::vector<T> data = create_random_data_with_seed<T>(
      max_size, std::numeric_limits<T>::max(), seed);

  start = get_usecs();
  for (uint32_t i = 1; i < max_size; i++) {
    s.insert(data[i]);
  }
  end = get_usecs();
  printf("insertion,\t %lu,\t", end - start);
  start = get_usecs();
  uint64_t sum = 0;
  for (auto el : s) {
    sum += el;
  }
  end = get_usecs();
  printf("sum_time, \t%lu, \tsum_total, \t%lu\n", end - start, sum);
}

template <typename traits> void test_cpma_ordered_insert(uint64_t max_size) {
  if (max_size > std::numeric_limits<typename traits::key_type>::max()) {
    max_size = std::numeric_limits<typename traits::key_type>::max();
  }
  uint64_t start = 0;
  uint64_t end = 0;
  CPMA<traits> s;
  start = get_usecs();
  for (uint32_t i = 0; i < max_size; i++) {
    s.insert(i);
  }
  end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  uint64_t sum = 0;
  start = get_usecs();
  sum = s.sum();
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu\n", end - start, sum);
}

template <typename T> void test_tlx_btree_ordered_insert(uint64_t max_size) {
  if (max_size > std::numeric_limits<T>::max()) {
    max_size = std::numeric_limits<T>::max();
  }
  uint64_t start = 0;
  uint64_t end = 0;
  tlx::btree_set<T> s;
  start = get_usecs();
  for (uint32_t i = 0; i < max_size; i++) {
    s.insert(i);
  }
  end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  uint64_t sum = 0;
  start = get_usecs();
  for (const auto e : s) {
    sum += e;
  }
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu\n", end - start, sum);
}

template <typename traits>
void test_cpma_size(uint64_t max_size, std::seed_seq &seed) {
  if (max_size > std::numeric_limits<typename traits::key_type>::max()) {
    max_size = std::numeric_limits<typename traits::key_type>::max();
  }
  {
    std::cout << "numbers 1 to " << max_size << std::endl;
    std::vector<double> sizes;
    CPMA<traits> s;
    for (uint32_t i = 1; i < max_size; i++) {
      s.insert(i);
      if (i > 1000) {
        sizes.push_back(static_cast<double>(s.get_size()) / i);
      }
      if (i % (max_size / 8) == 0) {
        std::cout << "after " << i << " elements, size is " << s.get_size()
                  << ", " << s.get_size() / i << " per element" << std::endl;
      }
    }
    std::cout << "after " << max_size << " elements, size is " << s.get_size()
              << ", " << s.get_size() / max_size << " per element" << std::endl;
    std::cout << std::endl;
    if (!sizes.empty()) {
      std::cout << "statisics ignoring the first 1000" << std::endl;
      double min_num = std::numeric_limits<double>::max();
      double max_num = 0;
      double sum = 0;
      for (auto x : sizes) {
        min_num = std::min(x, min_num);
        max_num = std::max(x, max_num);
        sum += x;
      }
      std::cout << "min = " << min_num << ", max = " << max_num
                << ", average = " << sum / (max_size - 1000) << std::endl;
    }
    std::cout << std::endl;
  }
  {
    std::cout << max_size << " uniform random 32 bit numbers" << std::endl;
    std::vector<double> sizes;
    std::mt19937 eng(seed);
    std::uniform_int_distribution<uint64_t> distrib(
        1, std::numeric_limits<typename traits::key_type>::max());
    CPMA<traits> s;
    for (uint32_t i = 1; i < max_size; i++) {
      s.insert(distrib(eng));
      if (i > 1000) {
        sizes.push_back(static_cast<double>(s.get_size()) /
                        s.get_element_count());
      }
      if (i % (max_size / 8) == 0) {
        std::cout << "after " << i << " inserts: number of elements is "
                  << s.get_element_count() << ", size is " << s.get_size()
                  << ", " << s.get_size() / s.get_element_count()
                  << " per element" << std::endl;
      }
    }
    std::cout << "after " << max_size << " inserts: number of elements is "
              << s.get_element_count() << ", size is " << s.get_size() << ", "
              << s.get_size() / s.get_element_count() << " per element"
              << std::endl;
    std::cout << std::endl;
    if (!sizes.empty()) {
      std::cout << "statisics ignoring the first 1000" << std::endl;
      double min_num = std::numeric_limits<double>::max();
      double max_num = 0;
      double sum = 0;
      for (auto x : sizes) {
        min_num = std::min(x, min_num);
        max_num = std::max(x, max_num);
        sum += x;
      }
      std::cout << "min = " << min_num << ", max = " << max_num
                << ", average = " << sum / (max_size - 1000) << std::endl;
    }
    std::cout << std::endl;
  }

  // zipf distributions
  for (uint32_t max = 1U << 22U; max < 1U << 27U; max *= 8) {
    for (double alpha = 1; alpha < 2.1; alpha += 1) {
      std::cout << "ZipF: max = " << max << " alpha = " << alpha << std::endl;
      zipf zip(max, alpha, seed);

      std::vector<double> sizes;
      CPMA<traits> s;
      for (uint32_t i = 1; i < max_size; i++) {
        s.insert(zip.gen());
        if (i > 1000) {
          sizes.push_back(static_cast<double>(s.get_size()) /
                          s.get_element_count());
        }
        if (i % (max_size / 8) == 0) {
          std::cout << "after " << i << " inserts: number of elements is "
                    << s.get_element_count() << ", size is " << s.get_size()
                    << ", " << s.get_size() / s.get_element_count()
                    << " per element" << std::endl;
        }
      }
      std::cout << "after " << max_size << " inserts: number of elements is "
                << s.get_element_count() << ", size is " << s.get_size() << ", "
                << s.get_size() / s.get_element_count() << " per element"
                << std::endl;
      std::cout << std::endl;
      if (!sizes.empty()) {
        std::cout << "statisics ignoring the first 1000" << std::endl;
        double min_num = std::numeric_limits<double>::max();
        double max_num = 0;
        double sum = 0;
        for (auto x : sizes) {
          min_num = std::min(x, min_num);
          max_num = std::max(x, max_num);
          sum += x;
        }
        std::cout << "min = " << min_num << ", max = " << max_num
                  << ", average = " << sum / (max_size - 1000) << std::endl;
      }
      std::cout << std::endl;
    }
  }
  {
    std::cout << "binomial t=MAX_INT p = .5" << std::endl;
    std::vector<double> sizes;
    std::mt19937 eng(seed);
    std::binomial_distribution<uint64_t> distrib(max_size, .5);
    CPMA<traits> s;
    for (uint32_t i = 1; i < max_size; i++) {
      s.insert(distrib(eng));
      if (i > 1000) {
        sizes.push_back(static_cast<double>(s.get_size()) /
                        s.get_element_count());
      }
      if (i % (max_size / 8) == 0) {
        std::cout << "after " << i << " inserts: number of elements is "
                  << s.get_element_count() << ", size is " << s.get_size()
                  << ", " << s.get_size() / s.get_element_count()
                  << " per element" << std::endl;
      }
    }
    std::cout << "after " << max_size << " inserts: number of elements is "
              << s.get_element_count() << ", size is " << s.get_size() << ", "
              << s.get_size() / s.get_element_count() << " per element"
              << std::endl;
    std::cout << std::endl;
    if (!sizes.empty()) {
      std::cout << "statisics ignoring the first 1000" << std::endl;
      double min_num = std::numeric_limits<double>::max();
      double max_num = 0;
      double sum = 0;
      for (auto x : sizes) {
        min_num = std::min(x, min_num);
        max_num = std::max(x, max_num);
        sum += x;
      }
      std::cout << "min = " << min_num << ", max = " << max_num
                << ", average = " << sum / (max_size - 1000) << std::endl;
    }
    std::cout << std::endl;
  }
  {
    std::cout << "geometrc p = 1/" << max_size << std::endl;
    std::vector<double> sizes;
    std::mt19937 eng(seed);
    std::geometric_distribution<uint64_t> distrib(1.0 / max_size);
    CPMA<traits> s;
    for (uint32_t i = 1; i < max_size; i++) {
      s.insert(distrib(eng));
      if (i > 1000) {
        sizes.push_back(static_cast<double>(s.get_size()) /
                        s.get_element_count());
      }
      if (i % (max_size / 8) == 0) {
        std::cout << "after " << i << " inserts: number of elements is "
                  << s.get_element_count() << ", size is " << s.get_size()
                  << ", " << s.get_size() / s.get_element_count()
                  << " per element" << std::endl;
      }
    }
    std::cout << "after " << max_size << " inserts: number of elements is "
              << s.get_element_count() << ", size is " << s.get_size() << ", "
              << s.get_size() / s.get_element_count() << " per element"
              << std::endl;
    std::cout << std::endl;
    if (!sizes.empty()) {
      std::cout << "statisics ignoring the first 1000" << std::endl;
      double min_num = std::numeric_limits<double>::max();
      double max_num = 0;
      double sum = 0;
      for (auto x : sizes) {
        min_num = std::min(x, min_num);
        max_num = std::max(x, max_num);
        sum += x;
      }
      std::cout << "min = " << min_num << ", max = " << max_num
                << ", average = " << sum / (max_size - 1000) << std::endl;
    }
    std::cout << std::endl;
  }

  {
    std::cout << "poisson_distribution mean = " << max_size << std::endl;
    std::vector<double> sizes;
    std::mt19937 eng(seed);
    std::poisson_distribution<uint64_t> distrib(max_size);
    CPMA<traits> s;
    for (uint32_t i = 1; i < max_size; i++) {
      s.insert(distrib(eng));
      if (i > 1000) {
        sizes.push_back(static_cast<double>(s.get_size()) /
                        s.get_element_count());
      }
      if (i % (max_size / 8) == 0) {
        std::cout << "after " << i << " inserts: number of elements is "
                  << s.get_element_count() << ", size is " << s.get_size()
                  << ", " << s.get_size() / s.get_element_count()
                  << " per element" << std::endl;
      }
    }
    std::cout << "after " << max_size << " inserts: number of elements is "
              << s.get_element_count() << ", size is " << s.get_size() << ", "
              << s.get_size() / s.get_element_count() << " per element"
              << std::endl;
    std::cout << std::endl;
    if (!sizes.empty()) {
      std::cout << "statisics ignoring the first 1000" << std::endl;
      double min_num = std::numeric_limits<double>::max();
      double max_num = 0;
      double sum = 0;
      for (auto x : sizes) {
        min_num = std::min(x, min_num);
        max_num = std::max(x, max_num);
        sum += x;
      }
      std::cout << "min = " << min_num << ", max = " << max_num
                << ", average = " << sum / (max_size - 1000) << std::endl;
    }
    std::cout << std::endl;
  }
}

template <typename traits>
void test_cpma_size_file_out(uint64_t max_size, std::seed_seq &seed,
                             const std::string &filename) {
  if (max_size > std::numeric_limits<typename traits::key_type>::max()) {
    max_size = std::numeric_limits<typename traits::key_type>::max();
  }
  std::vector<std::string> header;
  std::vector<std::vector<double>> sizes(max_size + 1);
  {
    header.emplace_back("sequence");
    sizes[0].push_back(0);
    CPMA<traits> s;
    for (uint32_t i = 1; i <= max_size; i++) {
      s.insert(i);
      sizes[i].push_back(static_cast<double>(s.get_size()) / i);
    }
    std::cout << "the sum of the numbers from 1 to " << max_size << " is "
              << s.sum() << std::endl;
  }
  {
    header.emplace_back("uniform_random");
    std::vector<typename traits::key_type> elements =
        create_random_data<typename traits::key_type>(
            max_size, std::numeric_limits<typename traits::key_type>::max(),
            seed);
    sizes[0].push_back(0);
    CPMA<traits> s;
    for (uint32_t i = 1; i <= max_size; i++) {
      s.insert(elements[i - 1]);
      sizes[i].push_back(static_cast<double>(s.get_size()) /
                         s.get_element_count());
    }
  }

  // zipf distributions
  // for (uint32_t max = 1 << 25U; max < 1 << 26; max *= 32) {
  uint32_t max = 1UL << 25U;
  for (double alpha = 1; alpha < 2.1; alpha += 1) {
    zipf zip(max, alpha, seed);
    header.push_back("zipf_" + std::to_string(max) + "_" +
                     std::to_string(alpha));
    std::vector<uint64_t> elements = zip.gen_vector(max_size);
    sizes[0].push_back(0);
    CPMA<traits> s;
    for (uint32_t i = 1; i <= max_size; i++) {
      s.insert(elements[i - 1]);
      sizes[i].push_back(static_cast<double>(s.get_size()) /
                         s.get_element_count());
    }
  }
  // }
  std::ofstream myfile;
  myfile.open(filename);
  const char delim = ',';
  myfile << Join(header, delim) << std::endl;
  for (const auto &row : sizes) {
    std::vector<std::string> stringVec;
    stringVec.reserve(row.size());
    for (const auto &e : row) {
      stringVec.push_back(std::to_string(e));
    }
    myfile << Join(stringVec, delim) << std::endl;
  }
}

template <typename traits>
void test_cpma_size_file_out_simple(uint64_t max_size, std::seed_seq &seed,
                                    const std::string &filename) {
  if (max_size > std::numeric_limits<typename traits::key_type>::max()) {
    max_size = std::numeric_limits<typename traits::key_type>::max();
  }
  std::ofstream myfile;
  myfile.open(filename);

  std::vector<typename traits::key_type> elements =
      create_random_data_with_seed<typename traits::key_type>(
          max_size, std::min(1UL << 40, max_size), seed);
  CPMA<traits> s;
  uint64_t start = get_usecs();
  std::vector<double> sizes(max_size, 0);
  for (uint32_t i = 1; i <= max_size; i++) {
    s.insert(elements[i - 1]);
    sizes[i] = static_cast<double>(s.get_size()) / s.get_element_count();
  }
  uint64_t end = get_usecs();
  std::cout << "took " << end - start << " micros" << std::endl;
  for (const auto si : sizes) {
    myfile << si << std::endl;
  }
  myfile.close();
}

template <typename traits>
void test_cpma_unordered_insert(uint64_t max_size, std::seed_seq &seed,
                                uint64_t iters) {
  if (max_size > std::numeric_limits<typename traits::key_type>::max()) {
    max_size = std::numeric_limits<typename traits::key_type>::max();
  }
  std::vector<typename traits::key_type> data =
      create_random_data_with_seed<typename traits::key_type>(
          max_size * iters,
          std::min(
              1UL << 40,
              (uint64_t)std::numeric_limits<typename traits::key_type>::max()),
          seed);
  uint64_t start = 0;
  uint64_t end = 0;
  std::vector<uint64_t> times;
  for (uint64_t it = 0; it < iters; it++) {
    CPMA<traits> s;
    start = get_usecs();
    for (uint64_t i = 0; i < max_size; i++) {
      s.insert(data[it * max_size + i]);
    }
    end = get_usecs();
    printf("insertion, \t %lu,", end - start);
    times.push_back(end - start);
    uint64_t sum = 0;
    start = get_usecs();
    sum = s.sum();
    end = get_usecs();
    printf("sum, \t %lu, total = %lu, size = %lu, size per element = %f\n",
           end - start, sum, s.get_size(),
           ((double)s.get_size()) / s.get_element_count());
  }
  std::sort(times.begin(), times.end());
  printf("mean time %lu\n", times[iters / 2]);
}

template <typename key_type>
void test_tlx_btree_unordered_insert(uint64_t max_size, std::seed_seq &seed,
                                     uint64_t iters) {
  if (max_size > std::numeric_limits<key_type>::max()) {
    max_size = std::numeric_limits<key_type>::max();
  }
  std::vector<key_type> data = create_random_data_with_seed<key_type>(
      max_size * iters,
      std::min(1UL << 40, (uint64_t)std::numeric_limits<key_type>::max()),
      seed);
  uint64_t start = 0;
  uint64_t end = 0;
  std::vector<uint64_t> times;
  for (uint64_t it = 0; it < iters; it++) {
    tlx::btree_set<key_type> s;
    start = get_usecs();
    for (uint64_t i = 0; i < max_size; i++) {
      s.insert(data[it * max_size + i]);
    }
    end = get_usecs();
    printf("insertion, \t %lu,", end - start);
    times.push_back(end - start);
    uint64_t sum = 0;
    start = get_usecs();
    for (const auto e : s) {
      sum += e;
    }
    end = get_usecs();
    printf("sum, \t %lu, total = %lu\n", end - start, sum);
  }
  std::sort(times.begin(), times.end());
  printf("mean time %lu\n", times[iters / 2]);
}

template <typename T>
void test_btree_ordered_and_unordered_insert(uint64_t max_size,
                                             uint64_t percent_ordered,
                                             std::seed_seq &seed,
                                             uint64_t trials) {
  if (max_size > std::numeric_limits<T>::max()) {
    max_size = std::numeric_limits<T>::max();
  }

  std::vector<T> data = create_random_data_with_seed<T>(
      max_size * trials, std::min(1UL << 40, max_size), seed);
  uint64_t start = 0;
  uint64_t end = 0;
  std::vector<uint64_t> times;
  for (uint64_t it = 0; it < trials; it++) {
    tlx::btree_set<T> s;
    start = get_usecs();
    for (int64_t i = max_size - 1; i >= 0; i--) {
      if (data[it * max_size + i] % 100 < percent_ordered) {
        s.insert(i);
      } else {
        s.insert(data[it * max_size + i] + max_size);
      }
    }
    end = get_usecs();
    printf("insertion, \t %lu,", end - start);
    times.push_back(end - start);
    uint64_t sum = 0;
    start = get_usecs();
    for (const auto e : s) {
      sum += e;
    }
    end = get_usecs();
    printf("sum = %lu, time was %lu\n", sum, end - start);
  }
  std::sort(times.begin(), times.end());
  printf("mean time %lu\n", times[trials / 2]);
}

template <typename traits>
void test_cpma_ordered_and_unordered_insert(uint64_t max_size,
                                            uint64_t percent_ordered,
                                            std::seed_seq &seed,
                                            uint64_t trials) {
  if (max_size > std::numeric_limits<typename traits::key_type>::max()) {
    max_size = std::numeric_limits<typename traits::key_type>::max();
  }
  std::vector<typename traits::key_type> data =
      create_random_data_with_seed<typename traits::key_type>(
          max_size * trials, std::min(1UL << 40, max_size), seed);
  uint64_t start = 0;
  uint64_t end = 0;
  std::vector<uint64_t> times;
  for (uint64_t it = 0; it < trials; it++) {
    CPMA<traits> s;
    start = get_usecs();
    for (int64_t i = max_size - 1; i >= 0; i--) {
      if (data[it * max_size + i] % 100 < percent_ordered) {
        s.insert(i);
      } else {
        s.insert(data[it * max_size + i] + max_size);
      }
    }
    end = get_usecs();
    printf("insertion, \t %lu,", end - start);
    times.push_back(end - start);
    uint64_t sum = 0;
    start = get_usecs();
    sum = s.sum();
    end = get_usecs();
    printf("sum = %lu, time was %lu\n", sum, end - start);
  }
  std::sort(times.begin(), times.end());
  printf("mean time %lu\n", times[trials / 2]);
}

template <typename T>
void test_btree_multi_seq_insert(uint64_t max_size, uint64_t groups) {
  if (max_size > std::numeric_limits<T>::max()) {
    max_size = std::numeric_limits<T>::max();
  }
  uint64_t elements_per_group = max_size / groups;
  std::vector<uint64_t> group_position(groups);
  for (uint64_t i = 0; i < groups; i++) {
    group_position[i] = i;
  }
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::shuffle(group_position.begin(), group_position.end(), eng);
  uint64_t start = 0;
  uint64_t end = 0;
  tlx::btree_set<T> s;
  start = get_usecs();
  for (uint32_t i = 0; i < elements_per_group; i++) {
    for (uint64_t j = 0; j < groups; j++) {
      s.insert(i + (group_position[j] << 30U));
    }
  }
  end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  uint64_t sum = 0;
  start = get_usecs();
  for (const auto e : s) {
    sum += e;
  }
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu\n", end - start, sum);
}

template <typename traits>
void test_cpma_multi_seq_insert(uint64_t max_size, uint64_t groups) {
  if (max_size > std::numeric_limits<typename traits::key_type>::max()) {
    max_size = std::numeric_limits<typename traits::key_type>::max();
  }
  uint64_t elements_per_group = max_size / groups;
  std::vector<uint64_t> group_position(groups);
  for (uint64_t i = 0; i < groups; i++) {
    group_position[i] = i;
  }
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::shuffle(group_position.begin(), group_position.end(), eng);
  uint64_t start = 0;
  uint64_t end = 0;
  CPMA<traits> s;
  start = get_usecs();
  for (uint32_t i = 0; i < elements_per_group; i++) {
    for (uint64_t j = 0; j < groups; j++) {
      s.insert(i + (group_position[j] << 30U));
    }
  }
  end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  uint64_t sum = 0;
  start = get_usecs();
  sum = s.sum();
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu\n", end - start, sum);
}

template <typename T>
void test_btree_bulk_insert(uint64_t max_size, uint64_t num_per) {
  if (max_size > std::numeric_limits<T>::max()) {
    max_size = std::numeric_limits<T>::max();
  }
  uint64_t groups = max_size / num_per;
  std::vector<uint64_t> group_position(groups);
  for (uint64_t i = 0; i < groups; i++) {
    group_position[i] = i;
  }
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::shuffle(group_position.begin(), group_position.end(), eng);
  uint64_t start = 0;
  uint64_t end = 0;
  tlx::btree_set<T> s;
  start = get_usecs();
  for (uint32_t j = 0; j < groups; j++) {
    for (uint64_t i = 0; i < num_per; i++) {
      s.insert(i + (group_position[j] << 30U));
    }
  }
  end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  uint64_t sum = 0;
  start = get_usecs();
  for (const auto e : s) {
    sum += e;
  }
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu\n", end - start, sum);
}

template <typename traits>
void test_cpma_bulk_insert(uint64_t max_size, uint64_t num_per) {
  if (max_size > std::numeric_limits<typename traits::key_type>::max()) {
    max_size = std::numeric_limits<typename traits::key_type>::max();
  }
  uint64_t groups = max_size / num_per;
  std::vector<uint64_t> group_position(groups);
  for (uint64_t i = 0; i < groups; i++) {
    group_position[i] = i;
  }
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::shuffle(group_position.begin(), group_position.end(), eng);
  uint64_t start = 0;
  uint64_t end = 0;
  CPMA<traits> s;
  start = get_usecs();
  for (uint32_t j = 0; j < groups; j++) {
    for (uint64_t i = 0; i < num_per; i++) {
      s.insert(i + (group_position[j] << 30U));
    }
  }
  end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  uint64_t sum = 0;
  start = get_usecs();
  sum = s.sum();
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu\n", end - start, sum);
}

template <typename traits>
void test_cpma_unordered_insert_batches(uint64_t max_size,
                                        uint64_t batch_size) {
  if (max_size > std::numeric_limits<typename traits::key_type>::max()) {
    max_size = std::numeric_limits<typename traits::key_type>::max();
  }
  if (batch_size > max_size) {
    batch_size = max_size;
  }
  std::vector<typename traits::key_type> data =
      create_random_data<typename traits::key_type>(
          max_size, std::numeric_limits<typename traits::key_type>::max());
  CPMA<traits> s;
  uint64_t start = get_usecs();
  for (uint32_t i = 0; i < max_size; i += batch_size) {
    if (i + batch_size > max_size) {
      batch_size = max_size - i;
    }
    s.insert_batch(data.data() + i, batch_size);
    // std::cout << "inserting " << data[i] << std::endl;
    // s.print_pma();
  }
  uint64_t end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  uint64_t sum = 0;
  start = get_usecs();
  sum = s.sum();
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu, size per element was %f\n",
         end - start, sum, ((double)s.get_size()) / s.get_element_count());
}

// take data as a vector
template <typename traits>
void test_cpma_unordered_insert_batches_from_data(
    uint64_t batch_size, std::vector<typename traits::key_type> data,
    int num_trials, char *filename, char *outfilename) {
  uint64_t max_size = data.size();

  if (batch_size > max_size) {
    batch_size = max_size;
  }

#ifdef VERIFY
  std::unordered_set<typename traits::key_type> correct;
  for (auto i : data) {
    correct.insert(i);
  }
  uint64_t correct_sum = 0;
  for (auto e : correct) {
    correct_sum += e;
  }
#endif

  uint64_t temp_batch_size = 0;
  uint64_t total_insert_time = 0;
  uint64_t total_delete_time = 0;
  uint64_t start = 0;
  uint64_t end = 0;
  uint64_t batch_time = 0;
  for (int i = 0; i < num_trials; i++) {
    CPMA<traits> s;
    start = get_usecs();
    if (batch_size > 1) {
      temp_batch_size = batch_size;
      // do batch inserts
      for (uint64_t j = 0; j < max_size; j += batch_size) {
        if (j + batch_size > max_size) {
          temp_batch_size = max_size % batch_size;
        }
        s.insert_batch(data.data() + j, temp_batch_size);
        // std::cout << "the pma has " << s.get_element_count()
        //           << " unique elements so far" << endl;
        // std::cout << "inserting " << data[i] << std::endl;
        // s.print_pma();
      }
    } else {
      for (uint64_t j = 0; j < max_size; j++) {
        s.insert(data[j]);
      }
    }
    end = get_usecs();
    total_insert_time += end - start;

    start = get_usecs();
    if (batch_size > 1) {
      // do batch deletes
      temp_batch_size = batch_size;
      for (uint64_t j = 0; j < max_size; j += batch_size) {
        if (j + batch_size > max_size) {
          temp_batch_size = max_size % batch_size;
        }
        s.remove_batch(data.data() + j, temp_batch_size);
        // std::cout << "inserting " << data[i] << std::endl;
        // s.print_pma();
      }
    } else {
      for (uint64_t j = 0; j < max_size; j++) {
        s.remove(data[j]);
      }
    }
    end = get_usecs();
    batch_time = end - start;
    // cout << "delete batch time: " << batch_time << endl;
    total_delete_time += batch_time;
  }
  double avg_insert = ((double)total_insert_time / 1000000) / num_trials;
  double avg_delete = ((double)total_delete_time / 1000000) / num_trials;
  // cout << "avg insert: " << avg_insert << endl;
  //  cout << "avg delete: " << avg_delete << endl;

  // do sum
  CPMA<traits> s;
  s.insert_batch(data.data(), data.size());

  num_trials *= 5; // do more trials for sum
  uint64_t total_time = 0;

  for (int i = 0; i < num_trials; i++) {
    uint64_t sum = 0;
    start = get_usecs();
    sum = s.sum();
    end = get_usecs();
    total_time += end - start;
#ifdef VERIFY
    assert(correct_sum == sum);
    if (i == 0) {
      cout << "got sum: " << sum << " expected sum: " << correct_sum << endl;
    }
#else
    if (i == 0) {
      std::cout << "got sum: " << sum << std::endl;
    }
#endif
  }
  double avg_sum = ((double)total_time / 1000000) / num_trials;

  auto size = s.get_size();

  // write out to file
  if (filename != nullptr) {
    std::ofstream outfile;
    outfile.open(outfilename, std::ios_base::app);
    outfile << filename << "," << batch_size << "," << avg_insert << ","
            << avg_delete << "," << avg_sum << "," << size << ","
            << ParallelTools::getWorkers() << std::endl;
    outfile.close();
  } else {
    std::cout << std::setprecision(2) << std::setw(12) << std::setfill(' ')
              << max_size << ", " << std::setw(12) << std::setfill(' ')
              << batch_size << ", " << std::setw(12) << std::setfill(' ')
              << avg_insert << ", " << std::setw(12) << std::setfill(' ')
              << avg_delete << ", " << std::setw(12) << std::setfill(' ')
              << avg_sum << ", " << size << std::endl;
  }
  // printf("sum_time, %lu, correct sum %lu, sum_total, %lu\n", end - start,
  // correct_sum, sum);
}

template <class pma_type, class set_type>
bool pma_different_from_set(const pma_type &pma, const set_type &set) {
  // check that the right data is in the set
  uint64_t correct_sum = 0;
  for (auto e : set) {
    correct_sum += e;
    if (!pma.has(e)) {
      printf("pma missing %lu\n", (uint64_t)e);
      return true;
    }
  }
  bool have_something_wrong = pma.template map<false>([&set](uint64_t element) {
    if (set.find(element) == set.end()) {
      printf("have something (%lu) that the set doesn't have\n",
             (uint64_t)element);

      return true;
    }
    return false;
  });
  if (have_something_wrong) {
    printf("pma has something is shouldn't\n");
    return true;
  }
  if (correct_sum != pma.sum()) {
    printf("pma has bad sum\n");
    return true;
  }
  return false;
}

template <typename traits>
bool verify_cpma(uint64_t number_trials, bool fast = false) {
  using T = typename traits::key_type;
  uint64_t max_num = std::numeric_limits<T>::max() - 1;
  if (max_num > std::numeric_limits<T>::max()) {
    max_num = std::numeric_limits<T>::max() - 1;
  }
  {
    CPMA<traits> t;
    uint64_t sum = 0;
    for (uint64_t i = 1; i < number_trials; i += 1) {
      // t.print_pma();
      // std::cout << "trying to insert " << i << "\n";
      t.insert(i);
      sum += i;
      if (sum != t.sum()) {
        std::cout << "bad sum after inserting sequntial numbers" << std::endl;
        std::cout << "got " << t.sum() << " expected " << sum << std::endl;
        std::cout << "just inserted " << i << std::endl;
        t.print_pma();
        return true;
      }
    }
  }
  tlx::btree_set<T> correct;
  CPMA<traits> test;
  // test.print_pma();
  auto random_numbers = create_random_data<T>(number_trials, max_num);
  for (uint64_t i = 0; i < number_trials; i++) {
    T x = random_numbers[i] + 1;
    // test.print_pma();
    // std::cout << "inserting: " << x << "\n";
    correct.insert(x);
    test.insert(x);
    assert(test.check_nothing_full());
    if (!fast) {
      if (pma_different_from_set(test, correct)) {
        printf("issue during inserts\n");
        return true;
      }
    }
  }
  if (pma_different_from_set(test, correct)) {
    printf("issue after inserts\n");
    return true;
  }
  random_numbers = create_random_data<T>(number_trials, max_num);
  for (uint64_t i = 0; i < number_trials; i++) {
    T x = random_numbers[i] + 1;
    // test.print_pma();
    // std::cout << "removing: " << x << std::endl;
    correct.erase(x);
    test.remove(x);
    if (!fast) {
      if (pma_different_from_set(test, correct)) {
        printf("issue during deletes\n");
        return true;
      }
    }
  }

  if (pma_different_from_set(test, correct)) {
    printf("issue after deletes\n");
    return true;
  }
  // test.print_pma();
  uint64_t num_rounds = 10;
  for (uint64_t round = 0; round < num_rounds; round++) {
    // put stuff into the pma
    std::vector<T> batch;
    random_numbers = create_random_data<T>(number_trials / num_rounds, max_num);
    for (uint64_t i = 0; i < number_trials / num_rounds; i++) {
      T x = random_numbers[i] + 1;
      batch.push_back(x);
      correct.insert(x);
    }

    // printf("before insert\n");
    // test.print_pma();
    // // test.print_array();

    // std::sort(batch.begin(), batch.end());
    // printf("\n*** BATCH %lu ***\n", round);
    // for (auto elt : batch) {
    //   std::cout << elt << ", ";
    // }
    // std::cout << std::endl;

    // try inserting batch
    test.insert_batch(batch.data(), batch.size());

    // everything in batch has to be in test
    for (auto e : batch) {
      if (!test.has(e)) {
        std::cout << "missing something in batch " << e << std::endl;
        test.print_pma();

        printf("\n*** BATCH ***\n");
        std::sort(batch.begin(), batch.end());
        for (auto elt : batch) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;

        printf("\n*** CORRECT ***\n");
        for (auto elt : correct) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;
        return true;
      }
    }

    // everything in correct has to be in test
    uint64_t correct_sum = 0;
    for (auto e : correct) {
      correct_sum += e;
      if (!test.has(e)) {
        std::cout << "missing something not in batch " << e << std::endl;
        // test.print_array();
        test.print_pma();

        printf("\n*** BATCH ***\n");
        std::sort(batch.begin(), batch.end());
        for (auto elt : batch) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;

        printf("\n*** CORRECT ***\n");
        for (auto elt : correct) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;
        return true;
      }
    }

    bool have_something_wrong = test.template map<false>([&correct](T element) {
      if (correct.find(element) == correct.end()) {
        printf("have something (%lu) that the set doesn't have\n",
               (uint64_t)element);

        return true;
      }
      return false;
    });
    if (have_something_wrong) {
      test.print_pma();
      printf("\n*** CORRECT ***\n");
      for (auto elt : correct) {
        std::cout << elt << ", ";
      }
      std::cout << std::endl;
    }

    // sum
    if (test.sum() != correct_sum) {
      std::cout << "sum got " << test.sum() << ", should be " << correct_sum
                << std::endl;
      test.print_pma();
      printf("\n*** CORRECT ***\n");
      for (auto elt : correct) {
        std::cout << elt << ", ";
      }
      std::cout << std::endl;
      return true;
    }
  }

  // random batch
  for (uint64_t round = 0; round < num_rounds; round++) {
    // put stuff into the pma
    std::vector<T> batch;
    random_numbers = create_random_data<T>(number_trials / num_rounds, max_num);
    for (uint64_t i = 0; i < number_trials / num_rounds; i++) {
      T x = random_numbers[i] + 1;
      batch.push_back(x);
      correct.erase(x);
    }

    // printf("before remove\n");
    // test.print_pma();

    // std::sort(batch.begin(), batch.end());
    // printf("\n*** BATCH %lu ***\n", round);
    // for (auto elt : batch) {
    //   std::cout << elt << ", ";
    // }
    // std::cout << std::endl;

    // try removing batch
    test.remove_batch(batch.data(), batch.size());

    // everything in batch has to be in test
    for (auto e : batch) {
      if (test.has(e)) {
        std::cout << "has something in random batch " << e << std::endl;
        test.print_pma();

        printf("\n*** BATCH ***\n");
        std::sort(batch.begin(), batch.end());
        for (auto elt : batch) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;

        printf("\n*** CORRECT ***\n");
        for (auto elt : correct) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;
        return true;
      }
    }

    // everything in correct has to be in test
    uint64_t correct_sum = 0;
    for (auto e : correct) {
      correct_sum += e;
      if (!test.has(e)) {
        std::cout << "missing something not in random batch after deletes " << e
                  << std::endl;
        // test.print_array();
        test.print_pma();

        printf("\n*** BATCH ***\n");
        std::sort(batch.begin(), batch.end());
        for (auto elt : batch) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;

        printf("\n*** CORRECT ***\n");
        for (auto elt : correct) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;
        return true;
      }
    }
    bool have_something_wrong = test.template map<false>([&correct](T element) {
      if (correct.find(element) == correct.end()) {
        printf("have something (%lu) that the set doesn't have after random "
               "batch deletes\n",
               (uint64_t)element);

        return true;
      }
      return false;
    });
    if (have_something_wrong) {
      test.print_pma();
      printf("\n*** CORRECT ***\n");
      for (auto elt : correct) {
        std::cout << elt << ", ";
      }
      std::cout << std::endl;
    }

    // sum
    if (test.sum() != correct_sum) {
      std::cout << "random batch: sum got " << test.sum() << ", should be "
                << correct_sum << std::endl;
      test.print_pma();
      printf("\n*** CORRECT ***\n");
      for (auto elt : correct) {
        std::cout << elt << ", ";
      }
      std::cout << std::endl;
      return true;
    }
  }
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<uint64_t> dist(0, correct.size());
  // batch of elements
  for (uint64_t round = 0; round < num_rounds; round++) {
    // put stuff into the pma
    std::vector<T> batch;
    for (uint64_t i = 0; i < number_trials / num_rounds; i++) {
      if (correct.empty()) {
        break;
      }
      // this is quite slow, but will generate random elements from the set
      uint32_t steps = dist(eng);
      auto it = std::begin(correct);
      std::advance(it, steps);
      if (it == std::end(correct)) {
        continue;
      }
      batch.push_back(*it);
      correct.erase(*it);
    }

    // printf("before insert\n");
    // test.print_pma();
    // test.print_array();

    // std::sort(batch.begin(), batch.end());
    // printf("\n*** BATCH %lu ***\n", round);
    // for (auto elt : batch) {
    //   std::cout << elt << ", ";
    // }
    // std::cout << std::endl;

    // try inserting batch
    test.remove_batch(batch.data(), batch.size());

    // everything in batch has to be in test
    for (auto e : batch) {
      if (test.has(e)) {
        std::cout << "has something in batch " << e << std::endl;
        test.print_pma();

        printf("\n*** BATCH ***\n");
        std::sort(batch.begin(), batch.end());
        for (auto elt : batch) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;

        printf("\n*** CORRECT ***\n");
        for (auto elt : correct) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;
        return true;
      }
    }

    // everything in correct has to be in test
    uint64_t correct_sum = 0;
    for (auto e : correct) {
      correct_sum += e;
      if (!test.has(e)) {
        std::cout << "batch deletes: missing something not in batch " << e
                  << std::endl;
        // test.print_array();
        test.print_pma();

        printf("\n*** BATCH ***\n");
        std::sort(batch.begin(), batch.end());
        for (auto elt : batch) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;

        printf("\n*** CORRECT ***\n");
        for (auto elt : correct) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;
        return true;
      }
    }

    bool have_something_wrong = test.template map<false>([&correct](T element) {
      if (correct.find(element) == correct.end()) {
        printf("have something (%lu) that the set doesn't have after "
               "batch deletes\n",
               (uint64_t)element);

        return true;
      }
      return false;
    });
    if (have_something_wrong) {
      test.print_pma();
      printf("\n*** CORRECT ***\n");
      for (auto elt : correct) {
        std::cout << elt << ", ";
      }
      std::cout << std::endl;
    }

    // sum
    if (test.sum() != correct_sum) {
      std::cout << "after deletes: sum got " << test.sum() << ", should be "
                << correct_sum << std::endl;
      test.print_pma();
      test.template map<true>([](T element) { std::cout << element << ", "; });
      printf("\n*** CORRECT ***\n");
      for (auto elt : correct) {
        std::cout << elt << ", ";
      }
      std::cout << std::endl;
      return true;
    }
  }

  return false;
}

template <typename traits>
bool verify_cpma_different_sizes(
    const std::vector<std::pair<uint64_t, bool>> &args) {
  for (auto arg : args) {
    printf("testing size %lu, fast = %d\n", std::get<0>(arg), std::get<1>(arg));
    if (verify_cpma<traits>(std::get<0>(arg), std::get<1>(arg))) {
      return true;
    }
  }
  return false;
}

template <typename leaf_type>
bool verify_leaf(uint32_t size, uint32_t num_ops, uint32_t range_start,
                 uint32_t range_end, uint32_t print = 0,
                 uint32_t bit_mask = 0) {
  static constexpr bool head_in_place = false;
  assert(size % 32 == 0);
  std::cout << "testing leaf size=" << size << ", num_ops=" << num_ops
            << ", range_start=" << range_start << ", range_end=" << range_end
            << ", bit_mask = " << bit_mask << "\n";
  uint8_t *array = (uint8_t *)memalign(32, size);
  typename leaf_type::key_type head = 0;
  for (uint32_t i = 0; i < size; i++) {
    array[i] = 0;
  }
  leaf_type leaf(head, array, size);

  // uncompressed_leaf<uint32_t> leaf(array, size);
  std::set<typename leaf_type::key_type> correct;
  auto random_numbers = create_random_data<typename leaf_type::key_type>(
      num_ops, range_end - range_start);
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<uint32_t> dist(0, 1);
  for (uint32_t i = 0; i < num_ops; i++) {
    uint32_t random_number = random_numbers[i];
    if (bit_mask) {
      random_number &= ~bit_mask;
    }
    uint32_t num = range_start + random_number;
    if (dist(eng) == 0) {
      if (print) {
        std::cout << "inserting " << num << std::endl;
      }
      leaf.template insert<head_in_place>(num);
      correct.insert(num);
    } else {
      if (print) {
        std::cout << "removing " << num << std::endl;
      }
      leaf.template remove<head_in_place>(num);
      correct.erase(num);
    }
    if (print > 1) {
      leaf.print();
    }

    bool leaf_missing = false;
    for (auto e : correct) {
      if (leaf.template contains<head_in_place>(e) != true) {
        std::cout << "leaf is missing " << e << std::endl;
        leaf_missing = true;
      }
    }
    if (leaf_missing) {
      std::cout << "correct has: ";
      for (auto e : correct) {
        std::cout << e << ", ";
      }
      std::cout << std::endl;
      leaf.print();
      free(array);
      return true;
    }
    uint64_t correct_sum = 0;
    for (auto e : correct) {
      correct_sum += e;
    }
    uint64_t leaf_sum = leaf.template sum<head_in_place>();
    if (correct_sum != leaf_sum) {
      std::cout << "the sums do not match, got " << leaf_sum << ", expected "
                << correct_sum << std::endl;
      std::cout << "correct has: ";
      for (auto e : correct) {
        std::cout << e << ", ";
      }
      std::cout << std::endl;
      leaf.print();
      free(array);
      return true;
    }
  }

  free(array);
  int num_leaves = 16;
  array = (uint8_t *)memalign(32, size * num_leaves);
  for (uint32_t i = 0; i < size * num_leaves; i++) {
    array[i] = 0;
  }
  uint32_t elts_per_leaf = size / 10;
  if (elts_per_leaf > num_ops) {
    elts_per_leaf = num_ops;
  }

  std::set<typename leaf_type::key_type> inputs;
  std::uniform_int_distribution<uint64_t> dist_1000(0, 999);
  if (elts_per_leaf > 900) {
    elts_per_leaf = 900;
  }
  while (inputs.size() < elts_per_leaf) {
    inputs.insert(dist_1000(eng));
  }
  // std::sort(inputs.begin(), inputs.end());

  // printf("inputs = {");

  uint64_t sum = 0;
  std::vector<typename leaf_type::key_type> heads(num_leaves);
  for (int i = 0; i < num_leaves; i++) {
    leaf_type leaf2(heads[i], array + (i * size), size);

    for (uint32_t elt : inputs) {
      uint32_t e = 1000 * i + elt;
      sum += e;
      leaf2.template insert<head_in_place>(e);
    }
    if (print > 1) {
      leaf2.print();
    }
  }
  // printf("}\n");

  auto result = leaf_type::template merge<head_in_place, false>(
      (typename leaf_type::key_type *)array, num_leaves, size, 0,
      [&heads](size_t idx) ->
      typename leaf_type::element_ref_type { return heads[idx]; },
      nullptr);
  free(array);
  if (result.leaf.template sum<head_in_place>() != sum) {
    printf("MERGE got sum %lu, should be %lu\n",
           result.leaf.template sum<head_in_place>(), sum);
    result.leaf.print();
    result.free();
    return true;
  }
  // free(result.first.array);

  // testing split

  uint8_t *dest_array = (uint8_t *)malloc(size * num_leaves);
  if (print > 1) {
    result.leaf.print();
  }
  result.leaf.template split<head_in_place, false>(
      num_leaves, result.size, size, (typename leaf_type::key_type *)dest_array,
      0,
      [&heads](size_t idx) ->
      typename leaf_type::element_ref_type { return heads[idx]; },
      nullptr);
  uint64_t total_sum = 0;
  for (int i = 0; i < num_leaves; i++) {
    leaf_type leaf2(heads[i], dest_array + (i * size), size);
    if (print > 1) {
      leaf2.print();
    }
    total_sum += leaf2.template sum<head_in_place>();
  }
  free(dest_array);
  result.free();

  if (total_sum != sum) {
    printf("SPLIT got sum %lu, should be %lu\n", total_sum, sum);
    return true;
  }
  return false;
}

template <class leaf>
bool time_leaf(uint32_t num_added_per_leaf, uint32_t num_leafs,
               uint32_t range_start, uint32_t range_end) {
  num_leafs =
      (num_leafs / ParallelTools::getWorkers()) * ParallelTools::getWorkers();
  std::cout << "timing leaf num_added_per_leaf=" << num_added_per_leaf
            << ", num_leafs=" << num_leafs << ", range_start=" << range_start
            << ", range_end=" << range_end << std::endl;
  size_t size_per_leaf = 5 * num_added_per_leaf;
  size_t num_cells = size_per_leaf * num_leafs;

  uint8_t *array = (uint8_t *)malloc(num_cells);
  if (!array) {
    std::cerr << "bad alloc array" << std::endl;
    return true;
  }

  ParallelTools::parallel_for(0, num_cells, [&](uint32_t i) { array[i] = 0; });
  auto random_numbers = create_random_data<uint32_t>(
      num_added_per_leaf * num_leafs, range_end - range_start);
  for (uint32_t i = 0; i < num_added_per_leaf * num_leafs; i++) {
    random_numbers[i] += range_start;
  }
  {
    timer insert_timer("inserts");
    insert_timer.start();
    ParallelTools::parallel_for(0, ParallelTools::getWorkers(), [&](int j) {
      for (uint32_t i = 0; i < num_added_per_leaf; i++) {
        for (uint32_t k = 0; k < num_leafs / ParallelTools::getWorkers(); k++) {
          uint32_t which_leaf =
              j * (num_leafs / ParallelTools::getWorkers()) + k;
          leaf l(array + (which_leaf * size_per_leaf), size_per_leaf);
          l.insert(random_numbers[i * num_leafs + which_leaf]);
        }
      }
    });
    insert_timer.stop();
  }
  timer sum_timer("sum");
  sum_timer.start();

  ParallelTools::Reducer_sum<uint64_t> counts;
  ParallelTools::parallel_for(0, num_leafs, [&](uint32_t j) {
    leaf l(array + (j * size_per_leaf), size_per_leaf);
    counts.add(l.sum());
  });
  uint64_t total_sum = counts.get();
  sum_timer.stop();
  uint64_t total_size = 0;
  for (uint32_t j = 0; j < num_leafs; j++) {
    leaf l(array + (j * size_per_leaf), size_per_leaf);
    total_size += l.used_size();
  }

  std::cout << "the total leaf sum is " << total_sum << std::endl;
  std::cout << "the average leaf size is " << total_size / num_leafs
            << std::endl;
  free(array);
  return false;
}

template <class traits>
void timing_cpma_helper(uint64_t max_size, uint64_t start_batch_size,
                        uint64_t end_batch_size, uint64_t num_trials,
                        std::seed_seq &seed) {
  using T = typename traits::key_type;
  if (max_size > std::numeric_limits<T>::max()) {
    max_size = std::numeric_limits<T>::max();
  }
  std::vector<T> data = create_random_data_with_seed<T>(
      max_size, std::numeric_limits<T>::max(), seed);
  std::cout << "uncompressed:" << std::endl;
  std::cout << std::setw(12) << std::setfill(' ') << "max_size"
            << ", " << std::setw(12) << std::setfill(' ') << "batch_size"
            << ", " << std::setw(12) << std::setfill(' ') << "avg_insert"
            << ", " << std::setw(12) << std::setfill(' ') << "avg_delete"
            << ", " << std::setw(12) << std::setfill(' ') << "avg_sum"
            << ", "
            << "size" << std::endl;
  for (uint64_t batch_size = start_batch_size; batch_size < end_batch_size;
       batch_size *= 10) {
    test_cpma_unordered_insert_batches_from_data<traits>(
        batch_size, data, num_trials, nullptr, nullptr);
  }

  std::cout << "compressed:" << std::endl;
  std::cout << std::setw(12) << std::setfill(' ') << "max_size"
            << ", " << std::setw(12) << std::setfill(' ') << "batch_size"
            << ", " << std::setw(12) << std::setfill(' ') << "avg_insert"
            << ", " << std::setw(12) << std::setfill(' ') << "avg_delete"
            << ", " << std::setw(12) << std::setfill(' ') << "avg_sum"
            << ", "
            << "size" << std::endl;
  for (uint64_t batch_size = start_batch_size; batch_size < end_batch_size;
       batch_size *= 10) {
    test_cpma_unordered_insert_batches_from_data<traits>(
        batch_size, data, num_trials, nullptr, nullptr);
  }
}

template <class traits>
bool real_graph(const std::string &filename, int iters = 20,
                uint32_t start_node = 0, uint32_t max_batch_size = 100000) {
  uint32_t num_nodes = 0;
  uint64_t num_edges = 0;
  auto edges = get_edges_from_file_adj_sym(filename, &num_edges, &num_nodes);

  printf("done reading in the file, n = %u, m = %lu\n", num_nodes, num_edges);
  CPMA<traits> g;

  auto start = get_usecs();

  g.insert_batch(edges.data(), edges.size());
  auto end = get_usecs();
  printf("inserting the edges took %lums\n", (end - start) / 1000);
  num_nodes = g.num_nodes();
  int64_t size = g.get_size();
  printf("size = %lu bytes, num_edges = %lu, num_nodes = %u\n", size,
         g.get_element_count(), num_nodes);

  int32_t parallel_bfs_result2_ = 0;
  uint64_t parallel_bfs_time2 = 0;

  for (int i = 0; i < iters; i++) {
    start = get_usecs();
    int32_t *parallel_bfs_result = EdgeMapVertexMap::BFS(g, start_node);
    end = get_usecs();
    parallel_bfs_result2_ += parallel_bfs_result[0];
    if (i == 0 && parallel_bfs_result != nullptr) {
      uint64_t reached = 0;
      for (uint32_t j = 0; j < num_nodes; j++) {
        reached += parallel_bfs_result[j] != -1;
      }
      printf("the bfs from source %u, reached %lu vertices\n", start_node,
             reached);
      std::vector<uint32_t> depths(num_nodes, UINT32_MAX);
      ParallelTools::parallel_for(0, num_nodes, [&](uint32_t j) {
        uint32_t current_depth = 0;
        int32_t current_parent = j;
        if (parallel_bfs_result[j] < 0) {
          return;
        }
        while (current_parent != parallel_bfs_result[current_parent]) {
          current_depth += 1;
          current_parent = parallel_bfs_result[current_parent];
        }
        depths[j] = current_depth;
      });
      std::ofstream myfile;
      myfile.open("bfs.out");
      for (unsigned int j = 0; j < num_nodes; j++) {
        myfile << depths[j] << "\n";
      }
      myfile.close();
    }

    free(parallel_bfs_result);
    parallel_bfs_time2 += (end - start);
  }

  printf("parallel_bfs with edge_map took %lums, parent of 0 = %d\n",
         parallel_bfs_time2 / (1000 * iters), parallel_bfs_result2_ / iters);
  printf("F-Graph, %d, BFS, %u, %s, ##, %f\n", iters, start_node,
         filename.c_str(), (double)parallel_bfs_time2 / (iters * 1000000));
  double pagerank_value = 0;
  uint64_t pagerank_time = 0;
  double *values3 = nullptr;
  for (int i = 0; i < iters; i++) {
    if (values3 != nullptr) {
      free(values3);
    }
    start = get_usecs();
    values3 = EdgeMapVertexMap::PR_S<double>(g, 10);
    end = get_usecs();
    pagerank_value += values3[0];
    pagerank_time += end - start;
  }
  printf("pagerank with MAPS took %f microsecond, value of 0 = %f, for %d "
         "iters, trash=%f\n",
         (double)pagerank_time / iters, values3[0], iters, pagerank_value);
  printf("F-Graph, %d, PageRank, %u, %s, ##, %f\n", iters, start_node,
         filename.c_str(), (double)pagerank_time / (iters * 1000000));
  {
    std::ofstream myfile;
    myfile.open("pr.out");
    for (unsigned int i = 0; i < num_nodes; i++) {
      myfile << values3[i] << "\n";
    }
    myfile.close();
  }
  free(values3);

  double *values4 = nullptr;
  double dep_0 = 0;
  uint64_t bc_time = 0;
  for (int i = 0; i < iters; i++) {
    if (values4 != nullptr) {
      free(values4);
    }
    start = get_usecs();
    values4 = EdgeMapVertexMap::BC(g, start_node);
    end = get_usecs();
    bc_time += end - start;
    dep_0 += values4[0];
  }

  printf("BC took %lums, value of 0 = %f\n", bc_time / (1000 * iters),
         dep_0 / iters);

  printf("F-Graph, %d, BC, %u, %s, ##, %f\n", iters, start_node,
         filename.c_str(), (double)bc_time / (iters * 1000000));
  if (values4 != nullptr) {
    std::ofstream myfile;
    myfile.open("bc.out");
    for (uint32_t i = 0; i < num_nodes; i++) {
      myfile << values4[i] << "\n";
    }
    myfile.close();
    free(values4);
  }

  uint32_t *values5 = nullptr;
  uint32_t id_0 = 0;
  uint64_t cc_time = 0;
  for (int i = 0; i < iters; i++) {
    if (values5) {
      free(values5);
    }
    start = get_usecs();
    values5 = EdgeMapVertexMap::CC(g);
    end = get_usecs();
    cc_time += end - start;
    id_0 += values5[0];
  }

  printf("CC took %lums, value of 0 = %u\n", cc_time / (1000 * iters),
         id_0 / iters);
  printf("F-Graph, %d, Components, %u, %s, ##, %f\n", iters, start_node,
         filename.c_str(), (double)cc_time / (iters * 1000000));
  if (values5 != nullptr) {
    std::unordered_map<uint32_t, uint32_t> components;
    for (uint32_t i = 0; i < num_nodes; i++) {
      components[values5[i]] += 1;
    }
    printf("there are %zu components\n", components.size());
    uint32_t curent_max = 0;
    uint32_t curent_max_key = 0;
    for (auto p : components) {
      if (p.second > curent_max) {
        curent_max = p.second;
        curent_max_key = p.first;
      }
    }
    printf("the element with the biggest component is %u, it has %u members "
           "to its component\n",
           curent_max_key, curent_max);
    std::ofstream myfile;
    myfile.open("cc.out");
    for (uint32_t i = 0; i < num_nodes; i++) {
      myfile << values5[i] << "\n";
    }
    myfile.close();
  }

  free(values5);

  if (true) {
    for (uint32_t b_size = 10; b_size <= max_batch_size; b_size *= 10) {
      auto r = random_aspen(b_size);
      double batch_insert_time = 0;
      double batch_remove_time = 0;
      for (int it = 0; it < iters + 1; it++) {

        double a = 0.5;
        double b = 0.1;
        double c = 0.1;
        size_t nn = 1UL << (log2_up(num_nodes) - 1);
        auto rmat = rMat<uint32_t>(nn, r.ith_rand(it), a, b, c);
        std::vector<uint64_t> es(b_size);
        ParallelTools::parallel_for(0, b_size, [&](uint32_t i) {
          std::pair<uint32_t, uint32_t> edge = rmat(i);
          es[i] = (static_cast<uint64_t>(edge.first) << 32U) | edge.second;
        });

        start = get_usecs();
        g.insert_batch(es.data(), b_size);
        end = get_usecs();
        if (it > 0) {
          batch_insert_time += end - start;
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(es.begin(), es.end(), gen);
        // size = g.get_memory_size();
        // printf("size end = %lu\n", size);
        start = get_usecs();
        g.remove_batch(es.data(), b_size);
        end = get_usecs();
        if (it > 0) {
          batch_remove_time += end - start;
        }
      }
      batch_insert_time /= (1000000 * iters);
      batch_remove_time /= (1000000 * iters);
      // printf("batch_size = %d, time to insert = %f seconds, throughput =
      // %4.2e "
      //        "updates/second\n",
      //        b_size, batch_insert_time, b_size / (batch_insert_time));
      // printf("batch_size = %d, time to remove = %f seconds, throughput =
      // %4.2e "
      //        "updates/second\n",
      //        b_size, batch_remove_time, b_size / (batch_remove_time));
      printf("%u, %f, %f\n", b_size, batch_insert_time, batch_remove_time);
    }
  }
  return true;
}

template <typename traits>
bool scan_bench(size_t num_elements, size_t num_bits, size_t iters = 5) {
  std::vector<uint64_t> times;
  for (size_t i = 0; i < iters; i++) {
    uint64_t start = 0;
    uint64_t end = 0;
    uint64_t sum = 0;
    auto data = create_random_data_in_parallel<typename traits::key_type>(
        num_elements, 1UL << num_bits);

    {
      CPMA<traits> uncompressed_pma;
      uncompressed_pma.insert_batch(data.data(), data.size());
      start = get_usecs();
      sum = uncompressed_pma.sum();
      end = get_usecs();
      printf("uncompressed sum took %lu microseconds was %lu: size = %lu "
             "head_size = %lu "
             "num_unique = %lu\n",
             end - start, sum, uncompressed_pma.get_size(),
             uncompressed_pma.get_head_structure_size(),
             uncompressed_pma.get_element_count());
      uint64_t sum2 = 0;
      uint64_t start2 = get_usecs();
      uncompressed_pma.template map<true>([&sum2](auto el) { sum2 += el; });
      uint64_t end2 = get_usecs();
      printf("sum2 = %lu, took %lu\n", sum2, end2 - start2);
      times.push_back((end - start));
    }
  }
  std::sort(times.begin(), times.end());
  printf("median after %zu iters was: %zu\n", iters, times[iters / 2]);

  return true;
}

bool scan_bench_btree(size_t num_elements, size_t num_bits, size_t iters = 5) {
  std::vector<uint64_t> btree_times;
  for (size_t i = 0; i < iters; i++) {
    std::seed_seq seed;
    uint64_t start = 0;
    uint64_t end = 0;
    uint64_t sum = 0;
    auto data = create_random_data<uint64_t>(num_elements, 1UL << num_bits);

    {

      tlx::btree_set<uint64_t> btree_set(data.begin(), data.end());

      uint64_t num_partitions = ParallelTools::getWorkers() * 10;
      uint64_t per_partition = num_elements / num_partitions;
      std::vector<tlx::btree_set<uint64_t>::const_iterator> iterators(
          num_partitions + 1);
      uint64_t position = 0;
      uint64_t partion_num = 0;
      for (auto it = btree_set.begin(); it != btree_set.end(); it++) {
        if (position % per_partition == 0) {
          iterators[partion_num] = it;
          partion_num += 1;
          if (partion_num == num_partitions) {
            break;
          }
        }
        position += 1;
      }
      uint64_t correct_sum = 0;
      uint64_t serial_start = get_usecs();
      for (auto it = btree_set.begin(); it != btree_set.end(); it++) {
        correct_sum += *it;
      }
      uint64_t serial_end = get_usecs();
      iterators[num_partitions] = btree_set.end();
      ParallelTools::Reducer_sum<uint64_t> partial_sums;
      start = get_usecs();
      ParallelTools::parallel_for(0, num_partitions, [&](uint64_t j) {
        uint64_t local_sum = 0;
        auto start_it = iterators[j];
        auto end_it = iterators[j + 1];
        for (auto it = start_it; it != end_it; it++) {
          local_sum += *it;
        }
        partial_sums.add(local_sum);
      });
      end = get_usecs();
      sum = partial_sums.get();
      struct key_of_value {
        static const uint64_t &get(const uint64_t &v) { return v; }
      };
      using btree_type = BTree_size_helper<uint64_t, uint64_t, key_of_value>;
      size_t btree_size =
          btree_set.get_stats().inner_nodes * sizeof(btree_type::InnerNode) +
          btree_set.get_stats().leaves * sizeof(btree_type::LeafNode);
      printf("parallel sum took %lu microseconds was %lu: serial sum took "
             "%lu, "
             "got "
             "%lu size was %lu\n",
             end - start, sum, serial_end - serial_start, correct_sum,
             btree_size);
      btree_times.push_back(std::min((end - start), serial_end - serial_start));
    }
  }
  std::sort(btree_times.begin(), btree_times.end());
  printf("average after %zu iters was: \n %zu\n", iters,
         btree_times[iters / 2]);

  return true;
}

template <typename traits>
std::tuple<uint64_t, uint64_t>
batch_test(size_t num_elements_start, size_t batch_size, size_t num_bits = 40,
           size_t iters = 5, bool verify = false) {
  uint64_t insert_total = 0;
  uint64_t delete_total = 0;
  for (size_t i = 0; i < iters + 1; i++) {

    uint64_t start = 0;
    uint64_t end = 0;
    auto data_start = create_random_data_in_parallel<typename traits::key_type>(
        num_elements_start, 1UL << num_bits);

    tlx::btree_set<typename traits::key_type> correct;
    if (verify) {
      correct.insert(data_start.begin(), data_start.end());
    }

    CPMA<traits> pma;
    pma.insert_batch(data_start.data(), data_start.size());
    data_start.clear();
    split_cnt.reset();
    size_cnt.reset();
    search_cnt.reset();
    search_steps_cnt.reset();
    auto data_batches =
        create_random_data_in_parallel<typename traits::key_type>(
            num_elements_start, 1UL << num_bits, 1000);
    if (verify) {
      correct.insert(data_batches.begin(), data_batches.end());
    }
    start = get_usecs();
    for (auto *batch_start = data_batches.data();
         batch_start < data_batches.data() + data_batches.size();
         batch_start += batch_size) {
      pma.insert_batch(batch_start, batch_size);
    }
    end = get_usecs();
    split_cnt.report();
    size_cnt.report();
    search_cnt.report();
    search_steps_cnt.report();
    if (i > 0 || iters == 0) {
      insert_total += end - start;
    }

    if (!verify) {
      printf("batch_size = %lu, total sum = %lu, time = %lu\n", batch_size,
             pma.sum(), end - start);
    }
    if (verify) {
      if (pma_different_from_set(pma, correct)) {
        printf("bad pma\n");
        return {-1UL, -1UL};
      }
    }
    std::random_device rd;
    std::mt19937 g(rd());
    auto delete_batch = parlay::random_shuffle(data_batches, rd());

    start = get_usecs();
    for (auto *batch_start = delete_batch.data();
         batch_start < delete_batch.data() + delete_batch.size();
         batch_start += batch_size) {
      pma.remove_batch(batch_start, batch_size);
    }
    end = get_usecs();
    if (i > 0) {
      delete_total += end - start;
    }
  }
  if (iters == 0) {
    iters = 1;
  }
  return {insert_total / iters, delete_total / iters};
}

// make a structure by adding elements in batches
// random numbers are not generated before hand to make it nicer to the cache
// do not time this function, only use it for cache numbers
template <typename traits>
uint64_t build_by_batch_for_cache_count(size_t total_elements,
                                        size_t batch_size,
                                        size_t num_bits = 40) {

  CPMA<traits> pma;
  for (size_t i = 0; i < total_elements; i += batch_size) {
    auto data = create_random_data<typename traits::key_type>(batch_size,
                                                              1UL << num_bits);
    pma.insert_batch(data.data(), data.size());
  }
  return pma.get_size();
}

template <typename traits>
double build_by_batch_for_time(size_t total_elements, size_t batch_size,
                               size_t trials, size_t num_bits = 40) {

  uint64_t total_time = 0;

  for (uint64_t trial = 0; trial <= trials; trial++) {
    auto data = create_random_data<typename traits::key_type>(total_elements,
                                                              1UL << num_bits);
    CPMA<traits> pma;
    uint64_t start = get_usecs();
    for (size_t i = 0; i < total_elements; i += batch_size) {

      pma.insert_batch(data.data() + i, batch_size);
    }
    uint64_t end = get_usecs();
    if (trial > 0) {
      total_time += (end - start);
    }
    std::cout << "the size is " << pma.get_size() << " took " << end - start
              << "\n";
  }
  double seconds = ((double)total_time) / (trials * 1000000);
  return seconds;
}

template <typename traits>
bool batch_bench(size_t num_elements_start, size_t num_bits = 40,
                 size_t iters = 5, bool verify = false) {
  std::map<size_t, uint64_t> insert_total;
  std::map<size_t, uint64_t> delete_total;

  for (size_t batch_size = 1; batch_size < num_elements_start;
       batch_size *= 10) {
    auto results = batch_test<traits>(num_elements_start, batch_size, num_bits,
                                      iters, verify);
    insert_total[batch_size] = std::get<0>(results);
    delete_total[batch_size] = std::get<1>(results);
  }

  for (size_t batch_size = 1; batch_size < num_elements_start;
       batch_size *= 10) {
    printf("%lu, %lu, %lu\n", batch_size, insert_total[batch_size] / iters,
           delete_total[batch_size] / iters);
  }
  return false;
}

template <class Set, class F, class key_type>
void map_range(const Set &set, F f, key_type s, key_type t) {
  constexpr bool has_map_range = requires(const Set &g) {
    g.map_range(f, s, t);
  };
  if constexpr (has_map_range) {
    set.map_range(f, s, t);
  } else {

    auto it = set.lower_bound(s);
    while (it != set.end() && *it < t) {
      f(*it);
      it++;
    }
  }
}

template <class Set>
void map_range_test(size_t num_elements_start, uint64_t num_ranges,
                    uint64_t size_of_range, size_t num_bits,
                    std::seed_seq &seed1, std::seed_seq &seed2) {
  using T = typename Set::key_type;
  auto data_to_insert = create_random_data_with_seed<T>(num_elements_start,
                                                        1UL << num_bits, seed1);
  Set s(data_to_insert.data(), data_to_insert.data() + data_to_insert.size());
  ParallelTools::Reducer_sum<uint64_t> sum_of_counts;
  ParallelTools::Reducer_sum<uint64_t> sum_of_vals;

  auto range_starts = create_random_data_with_seed<uint64_t>(
      num_ranges, (1UL << num_bits) - size_of_range, seed2);
  uint64_t start = get_usecs();
  ParallelTools::parallel_for(0, num_ranges, [&](uint64_t i) {
    uint64_t num_in_range = 0;
    uint64_t sum_in_range = 0;
    map_range(
        s,
        [&num_in_range, &sum_in_range]([[maybe_unused]] auto el) {
          num_in_range += 1;
          sum_in_range += el;
        },
        range_starts[i], range_starts[i] + size_of_range);
    sum_of_counts.add(num_in_range);
    sum_of_vals.add(sum_in_range);
  });
  uint64_t end = get_usecs();
  printf("the sum of the counts from the ranges was %lu took %lu "
         "range_size was %lu, sum of elements was %lu\n",
         sum_of_counts.get(), end - start, size_of_range, sum_of_vals.get());
}

template <class Set>
bool map_range_single(size_t num_elements_start, uint64_t num_ranges,
                      uint64_t size_of_range, size_t num_bits, size_t iters,
                      std::seed_seq &seed1, std::seed_seq &seed2) {
  using T = typename Set::key_type;
  std::vector<uint64_t> times;
  auto data_to_insert = create_random_data_with_seed<T>(
      num_elements_start * (iters + 1), 1UL << num_bits, seed1);

  for (size_t it = 0; it < iters + 1; it++) {

    uint64_t start = 0;
    uint64_t end = 0;
    Set s(data_to_insert.data() + (it * num_elements_start),
          data_to_insert.data() + ((it + 1) * num_elements_start));
    auto range_starts = create_random_data_with_seed<uint64_t>(
        num_ranges, (1UL << num_bits) - size_of_range, seed2);
    ParallelTools::Reducer_sum<uint64_t> sum_of_counts;
    ParallelTools::Reducer_sum<uint64_t> sum_of_vals;
    start = get_usecs();
    ParallelTools::parallel_for(0, num_ranges, [&](uint64_t i) {
      uint64_t num_in_range = 0;
      uint64_t sum_in_range = 0;
      map_range(
          s,
          [&num_in_range, &sum_in_range]([[maybe_unused]] auto el) {
            num_in_range += 1;
            sum_in_range += el;
          },
          range_starts[i], range_starts[i] + size_of_range);
      sum_of_counts.add(num_in_range);
      sum_of_vals.add(sum_in_range);
    });
    end = get_usecs();
    printf("the sum of the counts from the ranges was %lu took %lu "
           "range_size was %lu, sum of elements was %lu\n",
           sum_of_counts.get(), end - start, size_of_range, sum_of_vals.get());
    times.push_back(end - start);
  }
  uint64_t total_time = 0;
  for (uint64_t j = 1; j <= iters; j++) {
    total_time += times[j];
  }
  printf("for size %lu, mean time was %f\n", size_of_range,
         ((double)total_time) / (iters * 1000000));
  return true;
}

template <class Set>
bool map_range_bench(size_t num_elements_start, uint64_t num_ranges,
                     uint64_t num_sizes, size_t num_bits, size_t iters,
                     std::seed_seq &seed1, std::seed_seq &seed2) {
  using T = typename Set::key_type;
  std::vector<std::vector<uint64_t>> times(num_sizes);
  auto data_to_insert = create_random_data_with_seed<T>(
      num_elements_start * (iters + 1), 1UL << num_bits, seed1);

  for (size_t it = 0; it < iters + 1; it++) {

    uint64_t start = 0;
    uint64_t end = 0;
    Set s(data_to_insert.data() + (it * num_elements_start),
          data_to_insert.data() + ((it + 1) * num_elements_start));
    uint64_t log_size_of_range = 0;
    while (log_size_of_range < num_sizes) {
      uint64_t size_of_range = 1UL << log_size_of_range;
      auto range_starts = create_random_data_with_seed<uint64_t>(
          num_ranges, (1UL << num_bits) - size_of_range, seed2);
      ParallelTools::Reducer_sum<uint64_t> sum_of_counts;
      ParallelTools::Reducer_sum<uint64_t> sum_of_vals;
      start = get_usecs();
      ParallelTools::parallel_for(0, num_ranges, [&](uint64_t i) {
        uint64_t num_in_range = 0;
        uint64_t sum_in_range = 0;
        map_range(
            s,
            [&num_in_range, &sum_in_range]([[maybe_unused]] auto el) {
              num_in_range += 1;
              sum_in_range += el;
            },
            range_starts[i], range_starts[i] + size_of_range);
        sum_of_counts.add(num_in_range);
        sum_of_vals.add(sum_in_range);
      });
      end = get_usecs();
      printf("the sum of the counts from the ranges was %lu took %lu "
             "range_size was %lu, sum of elements was %lu\n",
             sum_of_counts.get(), end - start, size_of_range,
             sum_of_vals.get());
      times[log_size_of_range].push_back(end - start);
      log_size_of_range += 1UL;
    }
  }
  for (uint64_t i = 0; i < num_sizes; i++) {
    uint64_t total_time = 0;
    for (uint64_t j = 1; j <= iters; j++) {
      total_time += times[i][j];
    }
    printf("for size %lu, mean time was %f\n", 1UL << i,
           ((double)total_time) / (iters * 1000000));
  }
  return true;
}

template <typename traits>
bool find_bench(size_t num_elements_start, uint64_t num_searches,
                size_t num_bits = 40, size_t iters = 5, bool verify = false) {
  std::vector<size_t> find_times;
  for (size_t i = 0; i < iters; i++) {

    uint64_t start = 0;
    uint64_t end = 0;
    std::random_device rd1;
    std::seed_seq seed1{rd1()};

    auto data_to_insert =
        create_random_data_in_parallel<typename traits::key_type>(
            num_elements_start, 1UL << num_bits);

    auto data_to_search =
        create_random_data_in_parallel<typename traits::key_type>(
            num_searches, 1UL << num_bits, num_elements_start);
    std::unordered_set<uint64_t> correct;
    uint64_t correct_num_contains = 0;
    if (verify) {
      correct.insert(data_to_insert.begin(), data_to_insert.end());
      ParallelTools::Reducer_sum<uint64_t> number_contains;
      ParallelTools::parallel_for(0, data_to_search.size(), [&](uint64_t j) {
        number_contains.add(correct.contains(data_to_search[j]));
      });
      correct_num_contains = number_contains.get();
    }
    CPMA<traits> pma;
    pma.insert_batch(data_to_insert.data(), data_to_insert.size());
    ParallelTools::Reducer_sum<uint64_t> number_contains;
    start = get_usecs();
    ParallelTools::parallel_for(0, data_to_search.size(), [&](uint64_t j) {
      number_contains.add(pma.has(data_to_search[j]));
    });

    end = get_usecs();
    uint64_t num_contains = number_contains.get();
    find_times.push_back(end - start);

    if (!verify) {
      printf("found %lu elements, pma had %lu elements, the heads took %lu "
             "bytes, total took %lu bytes, %lu\n",
             num_contains, pma.get_element_count(),
             pma.get_head_structure_size(), pma.get_size(), end - start);
    }
    if (verify) {
      if (correct_num_contains != num_contains) {
        printf("something wrong with the finds, we found %lu, while the "
               "correct found %lu\n",
               num_contains, correct_num_contains);
      }
    }
  }
  std::sort(find_times.begin(), find_times.end());
  size_t find_time = 0;
  find_time = find_times[iters / 2];
  printf("median_find_time = %lu\n", find_time);
  return false;
}

bool find_bench_tlx_btree(size_t num_elements_start, uint64_t num_searches,
                          size_t num_bits = 40, size_t iters = 5,
                          bool verify = false) {
  std::vector<size_t> uncompressed_find_times;
  for (size_t i = 0; i < iters; i++) {

    uint64_t start;
    uint64_t end;
    std::random_device rd1;
    std::seed_seq seed1{rd1()};

    auto data_to_insert = create_random_data_in_parallel<uint64_t>(
        num_elements_start, 1UL << num_bits);

    auto data_to_search = create_random_data_in_parallel<uint64_t>(
        num_searches, 1UL << num_bits, num_elements_start);
    std::unordered_set<uint64_t> correct;
    uint64_t correct_num_contains = 0;
    if (verify) {
      correct.insert(data_to_insert.begin(), data_to_insert.end());
      ParallelTools::Reducer_sum<uint64_t> number_contains;
      ParallelTools::parallel_for(0, data_to_search.size(), [&](uint64_t j) {
        number_contains.add(correct.contains(data_to_search[j]));
      });
      correct_num_contains = number_contains.get();
    }
    {

      tlx::btree_set<uint64_t> btree_set;

      for (const auto &el : data_to_insert) {
        btree_set.insert(el);
      }

      ParallelTools::Reducer_sum<uint64_t> number_contains;
      start = get_usecs();
      ParallelTools::parallel_for(0, data_to_search.size(), [&](uint64_t j) {
        number_contains.add(btree_set.exists(data_to_search[j]));
      });

      end = get_usecs();
      uint64_t num_contains = number_contains.get();
      uncompressed_find_times.push_back(end - start);

      if (!verify) {
        struct key_of_value {
          static const uint64_t &get(const uint64_t &v) { return v; }
        };
        using btree_type = BTree_size_helper<uint64_t, uint64_t, key_of_value>;
        size_t btree_size =
            btree_set.get_stats().inner_nodes * sizeof(btree_type::InnerNode) +
            btree_set.get_stats().leaves * sizeof(btree_type::LeafNode);
        printf("found %lu elements, memory size was %lu, %lu\n", num_contains,
               btree_size, end - start);
      }
      if (verify) {
        if (correct_num_contains != num_contains) {
          printf("something wrong with the finds, we found %lu, while the "
                 "correct found %lu\n",
                 num_contains, correct_num_contains);
        }
      }
    }
  }
  std::sort(uncompressed_find_times.begin(), uncompressed_find_times.end());
  size_t uncompressed_time = uncompressed_find_times[iters / 2];
  printf("mean_uncompressed_find_time = %lu, compressed_find_time = 0\n",
         uncompressed_time);
  return false;
}

template <class Set, class F, class key_type>
void map_range_length(const Set &set, F f, key_type s, uint64_t length) {
  constexpr bool has_map_range_length = requires(const Set &g) {
    g.map_range_length(f, s, length);
  };
  if constexpr (has_map_range_length) {
    set.map_range_length(f, s, length);
  } else {
    auto it = set.lower_bound(s);
    uint64_t count = 0;
    while (it != set.end() && count < length) {
      f(*it);
      it++;
      count += 1;
    }
  }
}

template <class Set>
void ycsb_insert_bench(std::vector<bool> &operations,
                       std::vector<typename Set::key_type> &values) {

  Set s(values.data(), values.data() + values.size() / 2);
  uint64_t count_found = 0;
  uint64_t start = get_usecs();
  for (uint64_t i = values.size() / 2; i < values.size(); i++) {
    if (operations[i]) {
      s.insert(values[i]);
    } else {
      count_found += s.exists(values[i]);
    }
  }
  uint64_t end = get_usecs();
  printf("found %lu elemnts in %lu microseconds\n", count_found, end - start);
}

template <class Set>
void ycsb_scan_bench(std::vector<typename Set::key_type> values,
                     std::vector<typename Set::key_type> ranges) {

  Set s(values.data(), values.data() + values.size() / 2);
  uint64_t total_sum = 0;
  uint64_t start = get_usecs();
  for (uint64_t i = values.size() / 2; i < values.size(); i++) {
    if (ranges[i] == 0) {
      s.insert(values[i]);
    } else {
      map_range_length(
          s, [&total_sum](auto el) { total_sum += el; }, values[i], ranges[i]);
    }
  }
  uint64_t end = get_usecs();
  printf("total sum was %lu in %lu microseconds\n", total_sum, end - start);
}
