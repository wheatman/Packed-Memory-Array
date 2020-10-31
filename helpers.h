#ifndef HELPERS_H
#define HELPERS_H

#include "ParallelTools/parallel.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdint.h>
#include <string>
#include <sys/time.h>
#include <vector>
#include <x86intrin.h>
#ifndef NDEBUG
#define ASSERT(PREDICATE, ...)                                                 \
  do {                                                                         \
    if (!(PREDICATE)) {                                                        \
      fprintf(stderr,                                                          \
              "%s:%d (%s) Assertion " #PREDICATE " failed: ", __FILE__,        \
              __LINE__, __PRETTY_FUNCTION__);                                  \
      fprintf(stderr, __VA_ARGS__);                                            \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#else
#define ASSERT(...) // Nothing.
#endif

class __attribute__((__packed__)) uint24_t {
  std::array<uint8_t, 3> data = {0};

public:
  constexpr operator uint32_t() const {
    uint32_t x =
        static_cast<uint32_t>(data[0] | (data[1] << 8U) | (data[2] << 16U));
    return x;
  }
  constexpr uint24_t &operator+=(uint32_t y) {
    uint32_t x =
        static_cast<uint32_t>(data[0] | (data[1] << 8U) | (data[2] << 16U));
    x += y;
    data[0] = x & 0xFF;
    data[1] = (x >> 8U) & 0xFF;
    data[2] = (x >> 16U) & 0xFF;
    return *this;
  }
  constexpr uint24_t &operator-=(uint32_t y) {
    uint32_t x =
        static_cast<uint32_t>(data[0] | (data[1] << 8U) | (data[2] << 16U));
    x -= y;
    data[0] = x & 0xFF;
    data[1] = (x >> 8U) & 0xFF;
    data[2] = (x >> 16U) & 0xFF;
    return *this;
  }
  constexpr uint24_t(uint64_t x) {
    data[0] = x & 0xFF;
    data[1] = (x >> 8U) & 0xFF;
    data[2] = (x >> 16U) & 0xFF;
  }
  constexpr uint24_t() = default;
};

namespace std {
template <> class numeric_limits<uint24_t> {
public:
  static constexpr uint24_t max() { return uint24_t((1UL << 24U) - 1); }
  // One can implement other methods if needed
};
} // namespace std

template <typename T> T *newA(size_t n) { return (T *)malloc(n * sizeof(T)); }

#define watch(x) (#x) << "=" << (x) << std::endl;

#define REPORT2(a, b) watch(a) << ", " << watch(b) << std::endl;
#define REPORT3(a, b, c) REPORT2(a, b) << ", " << watch(c) << std::endl;
#define REPORT4(a, b, c, d) REPORT3(a, b, c) << ", " << watch(d) << std::endl;
#define REPORT5(a, b, c, d, e)                                                 \
  REPORT4(a, b, c, d) << ", " << watch(e) << std::endl;

#define intT int32_t
#define uintT uint32_t

// find index of first 1-bit (least significant bit)
static inline uint32_t bsf_word(uint32_t word) {
  uint32_t result;
  __asm__("bsf %1, %0" : "=r"(result) : "r"(word));
  return result;
}

static inline long bsf_long(long word) {
  long result;
  __asm__("bsfq %1, %0" : "=r"(result) : "r"(word));
  return result;
}

static inline int bsr_word(int word) {
  int result;
  __asm__("bsr %1, %0" : "=r"(result) : "r"(word));
  return result;
}

static inline uint64_t bsr_long(uint64_t word) {
  long result;
  __asm__("bsrq %1, %0" : "=r"(result) : "r"(word));
  return static_cast<uint64_t>(result);
}

static constexpr uint64_t bsr_long_constexpr(uint64_t word) {
  if (word == 0) {
    return 0;
  }
  if (word & (1UL << 63U)) {
    return 63;
  } else {
    return bsr_long_constexpr(word << 1UL) - 1;
  }
}

static inline bool power_of_2(uint64_t word) {
  return __builtin_popcountll(word) == 1;
}
static constexpr inline uint64_t nextPowerOf2(uint64_t n) {
  n--;
  n |= n >> 1UL;
  n |= n >> 2UL;
  n |= n >> 4UL;
  n |= n >> 8UL;
  n |= n >> 16UL;
  n |= n >> 32UL;
  n++;
  return n;
}

//#define ENABLE_TRACE_TIMER

static inline uint64_t get_usecs() {
  struct timeval st {};
  gettimeofday(&st, nullptr);
  return static_cast<uint64_t>(st.tv_sec * 1000000 + st.tv_usec);
}

template <typename T> static inline T unaligned_load(const void *loc) {
  static_assert(sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                "Size of T must be either 2, 4, or 8");
  T data;
  std::memcpy(&data, loc, sizeof(T));
  return data;
}

template <typename T> static inline void unaligned_store(void *loc, T value) {
  static_assert(sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                "Size of T must be either 2, 4, or 8");
  std::memcpy(loc, &value, sizeof(T));
}

inline std::string Join(std::vector<std::string> const &elements,
                        const char delimiter) {
  std::string out;
  for (size_t i = 0; i < elements.size() - 1; i++) {
    out += elements[i] + delimiter;
  }
  out += elements[elements.size() - 1];
  return out;
}

template <class T>
void wrapArrayInVector(T *sourceArray, size_t arraySize,
                       std::vector<T, std::allocator<T>> &targetVector) {
  typename std::_Vector_base<T, std::allocator<T>>::_Vector_impl *vectorPtr =
      (typename std::_Vector_base<T, std::allocator<T>>::_Vector_impl *)((
          void *)&targetVector);
  vectorPtr->_M_start = sourceArray;
  vectorPtr->_M_finish = vectorPtr->_M_end_of_storage =
      vectorPtr->_M_start + arraySize;
}

template <class T>
void releaseVectorWrapper(std::vector<T, std::allocator<T>> &targetVector) {
  typename std::_Vector_base<T, std::allocator<T>>::_Vector_impl *vectorPtr =
      (typename std::_Vector_base<T, std::allocator<T>>::_Vector_impl *)((
          void *)&targetVector);
  vectorPtr->_M_start = vectorPtr->_M_finish = vectorPtr->_M_end_of_storage =
      NULL;
}

template <class T> T prefix_sum_inclusive(std::vector<T> &data) {
  for (size_t i = 1; i < data.size(); i++) {
    data[i] += data[i - 1];
  }
  return data[data.size() - 1];
}

#if __AVX2__
template <class T> inline void Log(const __m256i &value) {
  const size_t n = sizeof(__m256i) / sizeof(T);
  T buffer[n];
  _mm256_storeu_si256((__m256i *)buffer, value);
  for (size_t i = 0; i < n; i++) {
    std::cout << +buffer[i] << " ";
  }
  std::cout << std::endl;
}

template <class T> inline void Log(const __m128i &value) {
  const size_t n = sizeof(__m128i) / sizeof(T);
  T buffer[n];
  _mm_storeu_si128((__m128i *)buffer, value);
  for (size_t i = 0; i < n; i++) {
    std::cout << +buffer[i] << " ";
  }
  std::cout << std::endl;
}
#endif

static uint64_t tzcnt(uint64_t num) {
#ifdef __BMI__
  return _tzcnt_u64(num);
#endif
  uint64_t count = 0;
  while ((num & 1) == 0) {
    count += 1;
    num >>= 1;
  }
  return count;
}

[[nodiscard]] inline uint64_t e_index(uint64_t index, uint64_t length) {
  uint64_t pos_0 = tzcnt(~index) + 1;
  uint64_t num_bits = bsr_long(length) + 1;
  return (1UL << (num_bits - pos_0)) + (index >> pos_0) - 1;
}

[[nodiscard]] inline uint64_t e_first_left_parent_eindex(uint64_t my_e_index) {
  if (my_e_index <= 1) {
    return std::numeric_limits<uint64_t>::max();
  }
  uint64_t parent_e_index = (my_e_index - 1) / 2;
  while ((parent_e_index * 2 + 2) != my_e_index && parent_e_index != 0) {
    my_e_index = parent_e_index;
    parent_e_index = (my_e_index - 1) / 2;
  }
  // if we are on the left spine
  if (parent_e_index == 0 && my_e_index == 1) {
    return std::numeric_limits<uint64_t>::max();
  }
  return parent_e_index;
}

[[nodiscard]] inline uint64_t e_first_left_parent_index(uint64_t index,
                                                        uint64_t length) {
  uint64_t my_e_index = e_index(index, length);
  return e_first_left_parent_eindex(my_e_index);
}

template <uint64_t B>
[[nodiscard]] inline uint64_t bnary_index(uint64_t index,
                                          uint64_t length_rounded) {
  static_assert(B != 0, "B can't be zero\n");
  uint64_t start = length_rounded / B;
  uint64_t tester = B;
  while ((index + 1) % tester == 0) {
    tester *= B;
    start /= B;
  }
  if (start == 0) {
    start = 1;
  }
#if DEBUG == 1
  {
    uint64_t start_test = 1;
    uint64_t test_size = length_rounded / B;
    while ((index + 1) % test_size != 0) {
      test_size /= B;
      start_test *= B;
    }
    ASSERT(start == start_test,
           "bad start, got %lu, expected %lu, index = %lu, length_rounded = "
           "%lu\n",
           start, start_test, index, length_rounded);
  }
#endif
  uint64_t size_to_consider = length_rounded / start;
  return start + (B - 1) * (index / (size_to_consider)) +
         (index % size_to_consider) / (size_to_consider / B) - 1;
}

template <typename... Args, std::enable_if_t<sizeof...(Args) == 0, int> = 0>
std::ostream &operator<<(std::ostream &os, const std::tuple<Args...> &t);
template <typename... Args, std::enable_if_t<sizeof...(Args) != 0, int> = 0>
std::ostream &operator<<(std::ostream &os, const std::tuple<Args...> &t);

// helper function to print a tuple of any size
template <class Tuple, std::size_t N> struct TuplePrinter {
  static void print(std::ostream &os, const Tuple &t) {
    using e_type = decltype(std::get<N - 1>(t));
    TuplePrinter<Tuple, N - 1>::print(os, t);
    if constexpr (sizeof(e_type) == 1) {
      os << static_cast<int64_t>(std::get<N - 1>(t));
    } else {
      os << ", " << std::get<N - 1>(t);
    }
  }
};

template <class Tuple> struct TuplePrinter<Tuple, 1> {
  static void print(std::ostream &os, const Tuple &t) {
    using e_type = decltype(std::get<0>(t));
    if constexpr (sizeof(e_type) == 1) {
      os << static_cast<int64_t>(std::get<0>(t));
    } else {
      os << std::get<0>(t);
    }
  }
};

template <typename... Args, std::enable_if_t<sizeof...(Args) == 0, int>>
std::ostream &operator<<(std::ostream &os,
                         [[maybe_unused]] const std::tuple<Args...> &t) {
  os << "()";
  return os;
}

template <typename... Args, std::enable_if_t<sizeof...(Args) != 0, int>>
std::ostream &operator<<(std::ostream &os, const std::tuple<Args...> &t) {
  os << "(";
  TuplePrinter<decltype(t), sizeof...(Args)>::print(os, t);
  os << ")";
  return os;
}

template <typename T1, typename T2>
std::ostream &operator<<(std::ostream &os, const std::pair<T1, T2> &t) {
  os << "( ";
  os << t.first << ", " << t.second;
  os << " )";
  return os;
}

template <typename T1, typename... rest>
static std::tuple<rest...>
leftshift_tuple(const std::tuple<T1, rest...> &tuple) {
  return std::apply([](auto &&, auto &...args) { return std::tie(args...); },
                    tuple);
}

struct free_delete {
  void operator()(void *x) { free(x); }
};

#endif
