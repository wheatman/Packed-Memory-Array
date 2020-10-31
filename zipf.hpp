#pragma once

#include <cassert>
#include <random> // mt19937 and uniform_int_distribution
#include <vector>

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
