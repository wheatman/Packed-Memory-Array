#ifndef HELPERS_H
#define HELPERS_H

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
#include <tuple>
#include <type_traits>
#include <utility>
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
  [[nodiscard]] constexpr uint32_t convert_to_32bit() const {
    return static_cast<uint32_t>(static_cast<uint32_t>(data[0]) |
                                 (static_cast<uint32_t>(data[1]) << 8U) |
                                 (static_cast<uint32_t>(data[2]) << 16U));
  }
  constexpr operator uint32_t() const { return convert_to_32bit(); }
  constexpr uint24_t &operator+=(uint32_t y) {
    uint32_t x = convert_to_32bit();
    x += y;
    data[0] = x & 0xFFU;
    data[1] = (x >> 8U) & 0xFFU;
    data[2] = (x >> 16U) & 0xFFU;
    return *this;
  }
  constexpr uint24_t &operator-=(uint32_t y) {
    uint32_t x = convert_to_32bit();
    x -= y;
    data[0] = x & 0xFFU;
    data[1] = (x >> 8U) & 0xFFU;
    data[2] = (x >> 16U) & 0xFFU;
    return *this;
  }
  constexpr uint24_t &operator++() {
    uint32_t x = convert_to_32bit();
    x++;
    data[0] = x & 0xFFU;
    data[1] = (x >> 8U) & 0xFFU;
    data[2] = (x >> 16U) & 0xFFU;
    return *this;
  }
  constexpr uint24_t &operator--() {
    uint32_t x = convert_to_32bit();
    x--;
    data[0] = x & 0xFFU;
    data[1] = (x >> 8U) & 0xFFU;
    data[2] = (x >> 16U) & 0xFFU;
    return *this;
  }
  constexpr uint24_t(uint64_t x) {
    data[0] = x & 0xFFU;
    data[1] = (x >> 8U) & 0xFFU;
    data[2] = (x >> 16U) & 0xFFU;
  }
  constexpr uint24_t() = default;
};

namespace std {
template <> class numeric_limits<uint24_t> {
public:
  static constexpr uint24_t max() { return uint24_t((1UL << 24U) - 1); }
  // One can implement other methods if needed
};
template <> class is_unsigned<uint24_t> {
public:
  static constexpr bool value() { return true; }
  // One can implement other methods if needed
};
} // namespace std


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
      nullptr;
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
  while ((num & 1U) == 0) {
    count += 1;
    num >>= 1U;
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

[[nodiscard]] inline uint64_t
rank_tree_array_get_prior_in_range(uint64_t index, uint64_t length,
                                   uint64_t min_e_index,
                                   uint64_t *rank_tree_array) {
  uint64_t left_parent_e_index = e_first_left_parent_index(index, length);
  uint64_t total = 0;
  while (left_parent_e_index != std::numeric_limits<uint64_t>::max() &&
         left_parent_e_index >= min_e_index) {
    total += rank_tree_array[left_parent_e_index];
    left_parent_e_index = e_first_left_parent_eindex(left_parent_e_index);
  }
  return total;
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

template <typename... T>
constexpr void add_to_tuple(const std::tuple<T &...> &t1,
                            const std::tuple<T...> &t2) {
  if constexpr (sizeof...(T) > 0) {
    std::get<0>(t1) += std::get<0>(t2);
  }
  if constexpr (sizeof...(T) > 1) {
    add_to_tuple(leftshift_tuple(t1), leftshift_tuple(t2));
  }
}

template <typename... Args, std::size_t... I>
std::tuple<Args &...> MakeTupleRef(std::tuple<Args...> &tuple,
                                   std::index_sequence<I...>) {
  return std::tie(std::get<I>(tuple)...);
}

template <typename... Args>
std::tuple<Args &...> MakeTupleRef(std::tuple<Args...> &tuple) {
  return MakeTupleRef(tuple, std::make_index_sequence<sizeof...(Args)>{});
}

template <typename T1, typename T2>
constexpr bool approx_equal_tuple(const T1 &t1, const T2 &t2) {
  static_assert(std::tuple_size_v<T1> == std::tuple_size_v<T2>);
  // clang-format off
  if constexpr (std::tuple_size_v<T1> == 0) {
    return true;
  }
  bool close_enough = true;
  if constexpr (std::tuple_size_v<T1> > 1) {
    close_enough = approx_equal_tuple(leftshift_tuple(t1), leftshift_tuple(t2));
  }
  // clang-format on
  if constexpr (std::is_integral_v<std::tuple_element<0, T1>> &&
                std::is_integral_v<std::tuple_element<0, T2>>) {
    close_enough &= (std::get<0>(t1) == std::get<0>(t2));
  } else {
    auto a = std::get<0>(t1);
    auto b = std::get<0>(t2);
    if (a == 0 || b == 0) {
      close_enough |= (std::abs((double)a + (double)b) < .000000001);
    } else {
      auto bigger = (a > b) ? a : b;
      auto smaller = (a > b) ? b : a;
      close_enough |= (bigger / smaller < 1.00000001);
    }
  }
  return close_enough;
}

struct free_delete {
  void operator()(void *x) { free(x); }
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

#endif
