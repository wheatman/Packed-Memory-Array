#pragma once

#include <cassert>
#include <cstdio>
#include <limits>
#include <random> // mt19937 and uniform_int_distribution
#include <type_traits>
#include <vector>

#include <parlay/primitives.h>

class zipf {
  double c = 0; // Normalization constant
  double alpha;
  uint64_t max;
  std::vector<double> sum_probs; // Pre-calculated sum of probabilities
  std::uniform_real_distribution<double> dist;
  std::mt19937 eng;

public:
  zipf(uint64_t m, double a, std::seed_seq &seed) : alpha(a), max(m) {
    dist = std::uniform_real_distribution<double>(0, 1);
    eng.seed(seed);

    for (uint64_t i = 1; i <= max; i++) {
      c = c + (1.0 / pow((double)i, alpha));
    }
    c = 1.0 / c;
    sum_probs.reserve(max + 1);
    sum_probs.push_back(0);
    for (uint64_t i = 1; i <= max; i++) {
      sum_probs.push_back(sum_probs[i - 1] + c / pow((double)i, alpha));
    }
  }

  uint64_t gen() {
    double z;
    // Pull a uniform random number (0 < z < 1)
    do {
      z = dist(eng);
    } while (z == 0);

    // Map z to the value
    uint64_t low = 1;
    uint64_t high = max;
    do {
      uint64_t mid = (low + high) / 2;
      if (sum_probs[mid] >= z && sum_probs[mid - 1] < z) {
        // Assert that zipf_value is between 1 and N
        assert((mid >= 1) && (mid <= max));
        return mid;
      }
      if (sum_probs[mid] >= z) {
        high = mid - 1;
      } else {
        low = mid + 1;
      }
    } while (low <= high);
    assert(false);
    return 1;
  }
  std::vector<uint64_t> gen_vector(uint64_t count) {
    std::vector<uint64_t> res;
    res.reserve(count);
    for (uint64_t i = 0; i < count; i++) {
      res.push_back(gen());
    }
    return res;
  }
};

template <class T> class parallel_zipf {
  double c = 0; // Normalization constant
  double alpha;
  uint64_t max;
  parlay::sequence<double> sum_probs; // Pre-calculated sum of probabilities
  std::uniform_real_distribution<double> dist;
  parlay::random_generator eng;

public:
  parallel_zipf(uint64_t m, double a, uint64_t seed) : alpha(a), max(m) {
    assert(m < std::numeric_limits<T>::max());
    dist = std::uniform_real_distribution<double>(0, 1);
    eng.seed(seed);
    // double c_test = 0;
    // for (uint64_t i = 1; i <= max; i++) {
    //   c_test = c_test + (1.0 / pow((double)i, alpha));
    // }
    // c_test = 1.0 / c_test;

    auto c_vec = parlay::delayed_tabulate(
        max, [&](size_t i) { return (1.0 / pow((double)i + 1, alpha)); });
    c = parlay::reduce(c_vec);
    c = 1.0 / c;
    // std::vector<double> sum_probs_test;
    // sum_probs_test.reserve(max + 1);
    // sum_probs_test.push_back(0);
    // for (uint64_t i = 1; i <= max; i++) {
    //   sum_probs_test.push_back(sum_probs_test[i - 1] +
    //                            c / pow((double)i, alpha));
    // }
    sum_probs = parlay::tabulate(max + 1, [&](size_t i) {
      if (i == 0) {
        return 0.0;
      } else {
        return c / pow((double)i, alpha);
      }
    });
    parlay::scan_inclusive_inplace(sum_probs);
  }

  T gen() {
    double z;
    // Pull a uniform random number (0 < z < 1)
    do {
      z = dist(eng);
    } while (z == 0);

    // Map z to the value
    T low = 1;
    T high = max;
    do {
      T mid = (low + high) / 2;
      if (sum_probs[mid] >= z && sum_probs[mid - 1] < z) {
        // Assert that zipf_value is between 1 and N
        assert((mid >= 1) && (mid <= max));
        return mid;
      }
      if (sum_probs[mid] >= z) {
        high = mid - 1;
      } else {
        low = mid + 1;
      }
    } while (low <= high);
    assert(false);
    return 1;
  }
  parlay::sequence<T> gen_vector(uint64_t count) {
    return parlay::tabulate(count, [&](size_t i) {
      double z;
      // Pull a uniform random number (0 < z < 1)
      do {
        auto r = eng[i];
        z = dist(r);
      } while (z == 0);
      T low = 1;
      T high = max;
      do {
        T mid = (low + high) / 2;
        if (sum_probs[mid] >= z && sum_probs[mid - 1] < z) {
          // Assert that zipf_value is between 1 and N
          assert((mid >= 1) && (mid <= max));
          return mid;
        }
        if (sum_probs[mid] >= z) {
          high = mid - 1;
        } else {
          low = mid + 1;
        }
      } while (low <= high);
      assert(false);
      return (T)1;
    });
  }
};
