#pragma once

// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2016 Guy Blelloch, Daniel Ferizovic, and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <climits>
// from numerical recipes
inline uint64_t hash64(uint64_t u) {
  uint64_t v = u * 3935559000370003845ul + 2691343689449507681ul;
  v ^= v >> 21U;
  v ^= v << 37U;
  v ^= v >> 4U;
  v *= 4768777513237032717ul;
  v ^= v << 20U;
  v ^= v >> 41U;
  v ^= v << 5U;
  return v;
}

// a 32-bit hash function
inline uint32_t hash32(uint32_t a) {
  a = (a + 0x7ed55d16u) + (a << 12u);
  a = (a ^ 0xc761c23cu) ^ (a >> 19u);
  a = (a + 0x165667b1u) + (a << 5u);
  a = (a + 0xd3a2646cu) ^ (a << 9u);
  a = (a + 0xfd7046c5u) + (a << 3u);
  a = (a ^ 0xb55a4f09u) ^ (a >> 16u);
  return a;
}

// A cheap version of an inteface that should be improved
// Allows forking a state into multiple states
struct random_aspen {
public:
  explicit random_aspen(size_t seed) : state(seed) {}
  random_aspen() {}
  random_aspen fork(uint64_t i) const {
    return random_aspen(hash64(hash64(i + state)));
  }
  random_aspen next() const { return fork(0); }
  size_t ith_rand(uint64_t i) const { return hash64(i + state); }
  size_t operator[](size_t i) const { return ith_rand(i); }
  size_t rand() { return ith_rand(0); }

private:
  uint64_t state = 0;
};

// returns the log base 2 rounded up (works on ints or longs or unsigned
// versions)
template <class T> size_t log2_up(T i) {
  size_t a = 0;
  T b = i - 1;
  while (b > 0) {
    b = b >> 1U;
    a++;
  }
  return a;
}

template <class intT> struct rMat {
  double a, ab, abc;
  intT n;
  intT h;
  rMat(intT _n, intT _seed, double _a, double _b, double _c) {
    n = _n;
    a = _a;
    ab = _a + _b;
    abc = _a + _b + _c;
    h = hash32((intT)_seed);
    if (abc > 1) {
      std::cout << "in rMat: a + b + c add to more than 1\n";
      abort();
    }
    if ((1UL << log2_up(n)) != n) {
      std::cout << "in rMat: n not a power of 2";
      abort();
    }
  }

  double hashDouble(intT i) {
    return ((double)(hash32((intT)i)) /
            ((double)std::numeric_limits<int32_t>::max()));
  }

  std::pair<intT, intT> rMatRec(intT nn, intT randStart, intT randStride) {
    if (nn == 1) {
      return std::make_pair<intT, intT>(0, 0);
    } else {
      std::pair<intT, intT> x =
          rMatRec(nn / 2, randStart + randStride, randStride);
      double r = hashDouble(randStart);
      if (r < a) {
        return x;
      } else if (r < ab) {
        return std::make_pair(x.first, x.second + nn / 2);
      } else if (r < abc) {
        return std::make_pair(x.first + nn / 2, x.second);
      } else {
        return std::make_pair(x.first + nn / 2, x.second + nn / 2);
      }
    }
  }

  std::pair<intT, intT> operator()(intT i) {
    intT randStart = hash32((intT)(2 * i) * h);
    intT randStride = hash32((intT)(2 * i + 1) * h);
    return rMatRec(n, randStart, randStride);
  }
};
