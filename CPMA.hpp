#ifndef CPMA_HPP
#define CPMA_HPP
#include "ParallelTools/concurrent_hash_map.hpp"
#include "ParallelTools/flat_hash_map.hpp"
#include "ParallelTools/parallel.h"
#include "ParallelTools/reducer.h"
#include "ParallelTools/sort.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include "StructOfArrays/soa.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
#pragma clang diagnostic ignored "-Wpadded"
#include "tlx/container/btree_map.hpp"
#pragma clang diagnostic pop

#if VQSORT == 1
#include <hwy/contrib/sort/vqsort.h>
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wextra-semi-stmt"
#pragma clang diagnostic ignored "-Wextra-semi"
#pragma clang diagnostic ignored "-Wcomma"
#pragma clang diagnostic ignored "-Wpadded"
#pragma clang diagnostic ignored "-Wunused-template"
#pragma clang diagnostic ignored "-Wpacked"
#pragma clang diagnostic ignored "-Wdeprecated-copy-with-dtor"
#pragma clang diagnostic ignored "-Wimplicit-int-float-conversion"
#pragma clang diagnostic ignored "-Wreserved-identifier"
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#pragma clang diagnostic ignored "-Wdouble-promotion"
#pragma clang diagnostic ignored "-Wnewline-eof"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wshadow-uncaptured-local"
#pragma clang diagnostic ignored "-Wshadow-field-in-constructor"
#include "parlaylib/include/parlay/primitives.h"
#pragma clang diagnostic pop

// #include "AlignedAllocator2.hpp"
#include "helpers.h"
#include "leaf.hpp"
#include "timers.hpp"

enum HeadForm { InPlace, Linear, Eytzinger, BNary };
// BNAry has B pointers, and B-1 elements in each block

template <typename l, HeadForm h, uint64_t b = 0, bool density = false>
class PMA_traits {
public:
  using leaf = l;
  using key_type = typename leaf::key_type;
  static constexpr HeadForm head_form = h;
  static constexpr uint64_t B_size = b;
  static constexpr bool store_density = density;

  static constexpr bool binary = leaf::binary;
  using element_type = typename leaf::element_type;
  using element_ref_type = typename leaf::element_ref_type;
  using element_ptr_type = typename leaf::element_ptr_type;
  using SOA_type = typename leaf::SOA_type;
  using value_type = typename leaf::value_type;
};
template <typename T = uint64_t>
using pma_settings = PMA_traits<uncompressed_leaf<T>, InPlace, 0, false>;
template <typename T = uint64_t>
using spmal_settings = PMA_traits<uncompressed_leaf<T>, Linear, 0, false>;
template <typename T = uint64_t>
using spmae_settings = PMA_traits<uncompressed_leaf<T>, Eytzinger, 0, false>;
template <typename T = uint64_t>
using spmab5_settings = PMA_traits<uncompressed_leaf<T>, BNary, 5, false>;
template <typename T = uint64_t>
using spmab9_settings = PMA_traits<uncompressed_leaf<T>, BNary, 9, false>;
template <typename T = uint64_t>
using spmab17_settings = PMA_traits<uncompressed_leaf<T>, BNary, 17, false>;
template <typename T = uint64_t> using spmab_settings = spmab17_settings<T>;

template <typename T = uint64_t>
using cpma_settings = PMA_traits<delta_compressed_leaf<T>, InPlace, 0, false>;
template <typename T = uint64_t>
using scpmal_settings = PMA_traits<delta_compressed_leaf<T>, Linear, 0, false>;
template <typename T = uint64_t>
using scpmae_settings =
    PMA_traits<delta_compressed_leaf<T>, Eytzinger, 0, false>;
template <typename T = uint64_t>
using scpmab5_settings = PMA_traits<delta_compressed_leaf<T>, BNary, 5, false>;
template <typename T = uint64_t>
using scpmab9_settings = PMA_traits<delta_compressed_leaf<T>, BNary, 9, false>;
template <typename T = uint64_t>
using scpmab17_settings =
    PMA_traits<delta_compressed_leaf<T>, BNary, 17, false>;
template <typename T = uint64_t> using scpmab_settings = scpmab17_settings<T>;

class empty_type {};

template <typename traits> class CPMA {
public:
  using leaf = typename traits::leaf;
  using key_type = typename traits::key_type;
  static constexpr uint64_t B_size = traits::B_size;
  static constexpr HeadForm head_form = traits::head_form;
  static constexpr bool store_density = traits::store_density;

  static_assert(std::is_trivially_copyable_v<key_type>,
                "T must be trivially copyable");
  static_assert(B_size == 0 || head_form == BNary,
                "B_size should only be used if we are using head_form = BNary");

  static constexpr bool binary = traits::binary;
  using element_type = typename traits::element_type;
  using element_ref_type = typename traits::element_ref_type;
  using element_ptr_type = typename traits::element_ptr_type;
  using SOA_type = typename traits::SOA_type;
  using value_type = typename traits::value_type;

private:
  static constexpr double growing_factor = 1.2; // 1.5;

  static constexpr int leaf_blow_up_factor = (sizeof(key_type) == 3) ? 18 : 16;
  static constexpr uint64_t min_leaf_size = 64;
  static_assert(min_leaf_size >= 64, "min_leaf_size must be at least 64 bytes");

  // This is technically only enough to grow to about 20TB, if you want it to be
  // bigger you can't keep the index as a uint8_t, but it is probably plenty for
  // now

  struct meta_data_t {
    uint64_t n;
    uint64_t logn;
    uint64_t loglogn;
    uint64_t total_leaves;
    uint64_t H;
    uint64_t total_leaves_rounded_up;
    uint64_t elts_per_leaf;
    void print() const {
      printf(
          "N = %lu, logN = %lu, loglogN = %lu, total_leaves = %lu, H = %lu\n",
          n, logn, loglogn, total_leaves, H);
    }
  };
  using meta_data_t = struct meta_data_t;

  static constexpr std::array<meta_data_t, 256> get_metadata_table() {

    std::array<meta_data_t, 256> res = {{}};
    uint64_t n = min_leaf_size;
    uint64_t lln = bsr_long_constexpr(bsr_long_constexpr(n));
    uint64_t ln = leaf_blow_up_factor * (1UL << lln);
    if (n < ln) {
      n = ln;
      lln = bsr_long_constexpr(bsr_long_constexpr(n));
    }

    while (n % ln != 0) {
      n += ln - (n % ln);
      lln = bsr_long_constexpr(bsr_long_constexpr(n));
      ln = leaf_blow_up_factor * (1UL << lln);
    }
    uint64_t total_leaves = n / ln;
    uint64_t total_leaves_rounded_up = 0;
    if constexpr (head_form == Eytzinger || head_form == BNary) {
      total_leaves_rounded_up = calculate_num_leaves_rounded_up(total_leaves);
    }

    uint64_t H = bsr_long_constexpr(total_leaves);

    uint64_t i = 0;
    res[i] = {.n = n,
              .logn = ln,
              .loglogn = lln,
              .total_leaves = total_leaves,
              .H = H,
              .total_leaves_rounded_up = total_leaves_rounded_up,
              .elts_per_leaf = ln / sizeof(key_type)};
    i += 1;
    for (; i < 256; i++) {
      uint64_t min_new_size = n + ln;
      uint64_t desired_new_size = std::numeric_limits<uint64_t>::max();
      // max sure that it fits in the amount of bits
      if (n < desired_new_size / growing_factor) {
        desired_new_size = n * growing_factor;
      }
      if (desired_new_size < min_new_size) {
        n = min_new_size;
      } else {
        n = desired_new_size;
      }
      lln = bsr_long_constexpr(bsr_long_constexpr(n));
      ln = leaf_blow_up_factor * (1UL << lln);
      while (n % ln != 0) {
        n += ln - (n % ln);
        lln = bsr_long_constexpr(bsr_long_constexpr(n));
        ln = leaf_blow_up_factor * (1U << lln);
      }
      total_leaves = n / ln;
      H = bsr_long_constexpr(total_leaves);
      if constexpr (head_form == Eytzinger || head_form == BNary) {
        total_leaves_rounded_up = calculate_num_leaves_rounded_up(total_leaves);
      }
      res[i] = {.n = n,
                .logn = ln,
                .loglogn = lln,
                .total_leaves = total_leaves,
                .H = H,
                .total_leaves_rounded_up = total_leaves_rounded_up,
                .elts_per_leaf = ln / sizeof(key_type)};
    }

    return res;
  }

  static constexpr std::array<meta_data_t, 256> meta_data =
      get_metadata_table();

  static constexpr std::array<std::array<float, sizeof(key_type) * 8>, 256>
  get_upper_density_bound_table() {
    std::array<std::array<float, sizeof(key_type) * 8>, 256> res;
    for (uint64_t i = 0; i < 256; i++) {
      auto m = meta_data[i];
      for (uint64_t j = 0; j < sizeof(key_type) * 8; j++) {
        float val = 1.0 / 2.0;

        if (m.H != 0) {
          val = 1.0 / 2.0 + (((1.0 / 2.0) * j) / m.H);
        }
        if (val >= static_cast<double>(m.logn - (3 * leaf::max_element_size)) /
                       m.logn) {
          val = static_cast<double>(m.logn - (3 * leaf::max_element_size)) /
                    m.logn -
                .001;
        }
        res[i][j] = val;
      }
    }
    return res;
  }

  static constexpr std::array<std::array<float, sizeof(key_type) * 8>, 256>
  get_lower_density_bound_table() {
    std::array<std::array<float, sizeof(key_type) * 8>, 256> res;
    for (uint64_t i = 0; i < 256; i++) {
      for (uint64_t j = 0; j < sizeof(key_type) * 8; j++) {
        auto m = meta_data[i];
        float val = std::max(((double)sizeof(key_type)) / m.logn, 1.0 / 4.0);
        if (m.H != 0) {
          val = std::max(((double)sizeof(key_type)) / m.logn,
                         1.0 / 4.0 - ((.125 * j) / m.H));
        }
        res[i][j] = val;
      }
    }
    return res;
  }

  static constexpr std::array<std::array<float, sizeof(key_type) * 8>, 256>
      upper_density_bound_table = get_upper_density_bound_table();

  static constexpr std::array<std::array<float, sizeof(key_type) * 8>, 256>
      lower_density_bound_table = get_lower_density_bound_table();

  key_type *data_array;

  [[no_unique_address]]
  typename std::conditional<head_form == InPlace, empty_type, key_type *>::type
      head_array;

  key_type count_elements_ = 0;

  uint8_t meta_data_index = 0;

  bool has_0 = false;

public:
  [[nodiscard]] uint64_t N() const { return meta_data[meta_data_index].n; }

private:
  uint64_t soa_num_spots() const { return N() / sizeof(key_type); }

  template <size_t... Is> element_type get_data(size_t i) const {
    return SOA_type::get_static(data_array, soa_num_spots(), i);
  }
  template <size_t... Is> element_ref_type get_data_ref(size_t i) const {
    return SOA_type::get_static(data_array, soa_num_spots(), i);
  }
  template <size_t... Is> element_ptr_type get_data_ptr(size_t i) const {
    return SOA_type::get_static_ptr(data_array, soa_num_spots(), i);
  }

  template <size_t... Is> element_type get_head(size_t i) const {
    return SOA_type::get_static(head_array, num_heads(), i);
  }
  template <size_t... Is> element_ref_type get_head_ref(size_t i) const {
    return SOA_type::get_static(head_array, num_heads(), i);
  }
  template <size_t... Is> element_ptr_type get_head_ptr(size_t i) const {
    return SOA_type::get_static_ptr(head_array, num_heads(), i);
  }

  // stored the density of each leaf to speed up merges and get_density_count
  // only use a uint16_t to save on space
  // this might not be enough to store out of place leaves after batch merge, so
  // in the event of an overflow just store max which we then just go count the
  // density as usual, which is cheap since it is stored out of place and the
  // size is just written
  [[no_unique_address]]
  typename std::conditional<store_density, uint16_t *, empty_type>::type
      density_array = {};

  [[nodiscard]] uint8_t *byte_array() const { return (uint8_t *)data_array; }

#if VQSORT == 1
  hwy::Sorter sorter;
#endif

  element_ref_type index_to_head(uint64_t index) const {
    if constexpr (head_form == InPlace) {
      return get_data_ref(index * elts_per_leaf());
    }
    // linear order
    if constexpr (head_form == Linear) {
      return get_head_ref(index);
    }
    //
    // Eytzinger order
    if constexpr (head_form == Eytzinger) {
      return get_head_ref(e_index(index, total_leaves()));
    }
    // BNary order
    if constexpr (head_form == BNary) {
      uint64_t in = bnary_index<B_size>(index, total_leaves_rounded_up());
      return get_head_ref(in);
    }
  }
  key_type &index_to_head_key(uint64_t index) const {
    return std::get<0>(index_to_head(index));
  }

  element_ptr_type index_to_data(uint64_t index) const {
    if constexpr (head_form == InPlace) {
      return get_data_ptr(index * elts_per_leaf()) + 1;
    } else {
      return get_data_ptr(index * elts_per_leaf());
    }
  }

  leaf get_leaf(uint64_t leaf_number) const {
    return leaf(index_to_head(leaf_number), index_to_data(leaf_number),
                leaf_size_in_bytes());
  }

  // how big will the leaf be not counting the head
  [[nodiscard]] uint64_t leaf_size_in_bytes() const {
    if constexpr (head_form == InPlace) {
      return logN() - sizeof(key_type);
    } else {
      return logN();
    }
  }

  [[nodiscard]] uint64_t num_heads() const {
    if constexpr (head_form == InPlace) {
      return 0;
    }
    // linear order
    if constexpr (head_form == Linear) {
      return total_leaves();
    }
    // make next power of 2
    if constexpr (head_form == Eytzinger) {
      if (nextPowerOf2(total_leaves()) > total_leaves()) {
        uint64_t space =
            ((nextPowerOf2(total_leaves()) - 1) + total_leaves() + 1) / 2;
        return space;
      }
      return ((total_leaves() * 2) - 1);
    }
    // BNary order
    if constexpr (head_form == BNary) {
      uint64_t size = B_size;
      while (size <= total_leaves()) {
        size *= B_size;
      }
      uint64_t check_size =
          ((size / B_size + total_leaves() + B_size) / B_size) * B_size;

      return std::min(size, check_size);
    }
  }

  [[nodiscard]] uint64_t head_array_size() const;

  [[nodiscard]] static constexpr uint64_t
  calculate_num_leaves_rounded_up(uint64_t total_leaves) {
    static_assert(head_form == Eytzinger || head_form == BNary,
                  "you should only be rounding the head array size of you are "
                  "in either Eytzinger or BNary form");
    // Eytzinger and Bnary sometimes need to know the rounded number of leaves
    // linear order
    // make next power of 2
    if constexpr (head_form == Eytzinger) {
      if (nextPowerOf2(total_leaves) > total_leaves) {
        return (nextPowerOf2(total_leaves) - 1);
      }
      return ((total_leaves * 2) - 1);
    }
    // BNary order
    if constexpr (head_form == BNary) {
      uint64_t size = B_size;
      while (size <= total_leaves) {
        size *= B_size;
      }
      return size;
    }
  }

  [[nodiscard]] uint64_t calculate_num_leaves_rounded_up() const {
    static_assert(head_form == Eytzinger || head_form == BNary,
                  "you should only be rounding the head array size of you are "
                  "in either Eytzinger or BNary form");
    return calculate_num_leaves_rounded_up(total_leaves());
  }

  [[nodiscard]] std::pair<float, float> density_bound(uint64_t depth) const;

  [[nodiscard]] uint64_t loglogN() const {
    return meta_data[meta_data_index].loglogn;
  }
  [[nodiscard]] uint64_t logN() const {
    return meta_data[meta_data_index].logn;
  }
  [[nodiscard]] uint64_t H() const { return meta_data[meta_data_index].H; }
  [[nodiscard]] uint64_t total_leaves() const {
    return meta_data[meta_data_index].total_leaves;
  }

  [[nodiscard]] uint64_t total_leaves_rounded_up() const {
    static_assert(head_form == Eytzinger || head_form == BNary,
                  "you should only be rounding the head array size of you are "
                  "in either Eytzinger or BNary form");
    return meta_data[meta_data_index].total_leaves_rounded_up;
  }
  [[nodiscard]] uint64_t elts_per_leaf() const {
    return meta_data[meta_data_index].elts_per_leaf;
  }

  [[nodiscard]] float lower_density_bound(uint64_t depth) const {
    ASSERT(depth < 10000000,
           "depth shouldn't be higher than log(n) it is %lu\n", depth);
    assert(((double)sizeof(key_type)) / logN() <=
           lower_density_bound_table[meta_data_index][depth]);

    return lower_density_bound_table[meta_data_index][depth];
  }
  [[nodiscard]] float upper_density_bound(uint64_t depth) const {
    ASSERT(upper_density_bound_table[meta_data_index][depth] <= density_limit(),
           "density_bound_[%lu].second = %f > density_limit() = %f\n", depth,
           upper_density_bound_table[meta_data_index][depth], density_limit());
    // making sure we don't pass in a negative number
    ASSERT(depth < 100000000UL, "depth = %lu\n", depth);
    return upper_density_bound_table[meta_data_index][depth];
  }

  [[nodiscard]] double density_limit() const {
    // we need enough space on both sides regardless of how elements are split
    return static_cast<double>(logN() - (3 * leaf::max_element_size)) / logN();
  }

  void grow_list(uint64_t times);
  void shrink_list(uint64_t times);

  [[nodiscard]] uint64_t get_density_count(uint64_t index, uint64_t len) const;
  [[nodiscard]] uint64_t get_density_count_no_overflow(uint64_t index,
                                                       uint64_t len) const;
  bool
  check_leaf_heads(uint64_t start_idx = 0,
                   uint64_t end_idx = std::numeric_limits<uint64_t>::max());
  [[nodiscard]] uint64_t get_depth(uint64_t len) const {
    return bsr_long(N() / len);
  }

  [[nodiscard]] uint64_t find_node(uint64_t index, uint64_t len) const {
    return (index / len) * len;
  }

  [[nodiscard]] uint64_t find_containing_leaf_index(
      key_type e, uint64_t start = 0,
      uint64_t end = std::numeric_limits<uint64_t>::max()) const;

  [[nodiscard]] uint64_t find_containing_leaf_number(
      key_type e, uint64_t start = 0,
      uint64_t end = std::numeric_limits<uint64_t>::max()) const {
    return find_containing_leaf_index(e, start, end) / elts_per_leaf();
  }

  [[nodiscard]] uint64_t find_containing_leaf_index_debug(
      key_type e, uint64_t start = 0,
      uint64_t end = std::numeric_limits<uint64_t>::max()) const;

  template <class F>
  [[nodiscard]] std::pair<std::vector<std::tuple<uint64_t, uint64_t>>,
                          std::optional<uint64_t>>
  get_ranges_to_redistibute(
      const ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>>
          &leaves_to_check,
      uint64_t num_elts_merged, F bounds_check) const;

  void redistribute_ranges(
      const std::vector<std::tuple<uint64_t, uint64_t>> &ranges);

  using leaf_bound_t = struct {
    key_type start_elt;
    key_type end_elt;
    uint64_t start_leaf_index;
    uint64_t end_leaf_index;
  };
  std::vector<leaf_bound_t> get_leaf_bounds(uint64_t split_points) const;

  template <class F>
  [[nodiscard]] std::pair<std::vector<std::tuple<uint64_t, uint64_t>>,
                          std::optional<uint64_t>>
  get_ranges_to_redistibute_serial(
      const ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>>
          &leaves_to_check,
      uint64_t num_elts_merged, F bounds_check) const;

  [[nodiscard]] uint64_t get_ranges_to_redistibute_lookup_sibling_count(
      const std::vector<ParallelTools::concurrent_hash_map<uint64_t, uint64_t>>
          &ranges_check,
      uint64_t start, uint64_t length, uint64_t level,
      uint64_t depth = 0) const;
  [[nodiscard]] uint64_t get_ranges_to_redistibute_lookup_sibling_count_serial(
      const std::vector<ska::flat_hash_map<uint64_t, uint64_t>> &ranges_check,
      uint64_t start, uint64_t length, uint64_t level) const;

  [[nodiscard]] std::pair<std::vector<std::tuple<uint64_t, uint64_t>>,
                          std::optional<uint64_t>>
  get_ranges_to_redistibute_debug(
      const ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>>
          &leaves_to_check,
      uint64_t num_elts_merged) const;

  [[nodiscard]] std::map<uint64_t, std::pair<uint64_t, uint64_t>>
  get_ranges_to_redistibute_internal(std::pair<uint64_t, uint64_t> *begin,
                                     std::pair<uint64_t, uint64_t> *end) const;

  [[nodiscard]] uint64_t sum_parallel() const;
  [[nodiscard]] uint64_t sum_parallel2(uint64_t start, uint64_t end,
                                       int depth) const;
  void print_array_region(uint64_t start_leaf, uint64_t end_leaf) const;

  void insert_post_place(uint64_t leaf_number, uint64_t byte_count);
  void remove_post_place(uint64_t leaf_number, uint64_t byte_count);

  template <typename it> static void sort_batch(it e, uint64_t batch_size) {
    // if this isn't just a set
    if constexpr (!binary) {
      ParallelTools::sort(e, e + batch_size);
      return;
    } else {
      key_type *batch;
      if constexpr (std::is_same_v<it, key_type *>) {
        batch = e;
      } else {
        batch = e.get_pointer();
      }
#if PARALLEL == 0
      if constexpr (!std::is_integral_v<key_type>) {
        ParallelTools::sort(batch, batch + batch_size);
        return;
      } else {
        if (batch_size > 100000) {

          std::vector<key_type> data_vector;
          wrapArrayInVector(batch, batch_size, data_vector);
          parlay::integer_sort_inplace(data_vector);
          releaseVectorWrapper(data_vector);

        } else {
#if VQSORT == 0
          std::sort(batch, batch + batch_size);
#else
          if (batch_size * sizeof(key_type) < 8UL * 1024) {
            std::sort(batch, batch + batch_size);
          } else {
            sorter(batch, batch_size, hwy::SortAscending());
          }
#endif
        }
      }
#else
      if constexpr (!std::is_integral_v<key_type>) {
        ParallelTools::sort(batch, batch + batch_size);
        return;
      } else {
        if (batch_size > 1000) {
          // TODO find out why this doesn't work
          std::vector<key_type> data_vector;
          wrapArrayInVector(batch, batch_size, data_vector);
          parlay::integer_sort_inplace(data_vector);
          releaseVectorWrapper(data_vector);
        } else {
          ParallelTools::sort(batch, batch + batch_size);
        }
      }
#endif
    }
  }

public:
  // TODO(wheatman) make private
  bool check_nothing_full();
  static constexpr bool compressed = leaf::compressed;
  explicit CPMA();
  CPMA(const CPMA &source);
  CPMA(key_type *start, key_type *end);
  ~CPMA() {
    if constexpr (head_form != InPlace) {
      free(head_array);
    }
    free(data_array);
    if constexpr (store_density) {
      free(density_array);
    }
  }
  void print_pma() const;
  void print_array() const;
  bool has(key_type e) const;
  value_type value(key_type e) const;
  bool exists(key_type e) const { return has(e); }
  bool insert(element_type e);

  uint64_t insert_batch(element_ptr_type e, uint64_t batch_size);
  uint64_t remove_batch(key_type *e, uint64_t batch_size);
  // split num is the index of which partition you are
  uint64_t insert_batch_internal(
      element_ptr_type e, uint64_t batch_size,
      ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>>
          &leaves_to_check,
      uint64_t start_leaf_idx, uint64_t end_leaf_idx);

  uint64_t insert_batch_internal_small_batch(
      element_ptr_type e, uint64_t batch_size,
      ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>>
          &leaves_to_check,
      uint64_t start_leaf_idx, uint64_t end_leaf_idx);
  uint64_t remove_batch_internal(
      key_type *e, uint64_t batch_size,
      ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>>
          &leaves_to_check,
      uint64_t start_leaf_idx, uint64_t end_leaf_idx);

  bool remove(key_type e);
  [[nodiscard]] uint64_t get_size() const;
  [[nodiscard]] uint64_t get_size_no_allocator() const;

  [[nodiscard]] uint64_t get_element_count() const { return count_elements_; }

  [[nodiscard]] uint64_t sum() const;
  [[nodiscard]] uint64_t sum_serial(int64_t start, int64_t end) const;
  [[nodiscard]] key_type max() const;
  [[nodiscard]] uint32_t num_nodes() const;
  [[nodiscard]] uint64_t num_edges() const { return get_element_count(); }

  uint64_t get_degree(uint64_t i, const auto &extra_data) const {
    return extra_data.second.get()[i];
  }
  [[nodiscard]] uint64_t get_head_structure_size() const {
    return head_array_size();
  }

  template <bool no_early_exit, class F> bool map(F f) const;
  template <bool no_early_exit, class F> bool map_values(F f) const;

  template <bool no_early_exit, class F>
  void serial_map_with_hint(F f, key_type end_key,
                            const typename leaf::iterator &hint) const;

  template <bool no_early_exit, class F>
  void serial_map_with_hint_par(F f, key_type end_key,
                                const typename leaf::iterator &hint,
                                const typename leaf::iterator &end_hint) const;

  std::unique_ptr<uint64_t, free_delete>
  getDegreeVector(typename leaf::iterator *hints) const;

  std::unique_ptr<uint64_t, free_delete>
  getApproximateDegreeVector(typename leaf::iterator *hints) const;

  // returns a vector
  // for each node in the graph has a pointer to the start and the most
  // recent element so we know what the difference is from if we are
  // starting from a head, returns nullptr and the number leaf we are
  // starting with
  std::pair<std::unique_ptr<typename leaf::iterator, free_delete>,
            std::unique_ptr<uint64_t, free_delete>>
  getExtraData(bool skip = false) const {
    if (skip) {
      return {};
    }

    auto hints = (typename leaf::iterator *)malloc(
        sizeof(typename leaf::iterator) * (num_nodes() + 1));
    ParallelTools::parallel_for(0, num_nodes(), 1024, [&](uint64_t i_) {
      uint64_t end = std::min(i_ + 1024, (uint64_t)num_nodes());
      uint64_t i = i_;
      uint64_t start_hint = find_containing_leaf_index(i << 32U);
      uint64_t end_hint = find_containing_leaf_index(end << 32U, start_hint) +
                          (elts_per_leaf());
      if (i < end) {
        uint64_t leaf_idx =
            find_containing_leaf_index(i << 32U, start_hint, end_hint);
        hints[i] = get_leaf(leaf_idx / elts_per_leaf()).lower_bound(i << 32UL);
        if (hints[i] == typename leaf::iterator_end()) {
          leaf_idx += elts_per_leaf();
          hints[i] = get_leaf(leaf_idx / elts_per_leaf()).begin();
        }
        assert(*(hints[i]) >= i << 32UL);
        i += 1;
        for (; i < end; i++) {
          if (leaf_idx > 0 &&
              leaf_idx + elts_per_leaf() < N() / sizeof(key_type) &&
              index_to_head_key((leaf_idx + elts_per_leaf()) /
                                elts_per_leaf()) > (i << 32U)) {

            // in the same leaf as the previous one so use the hint to speed
            // up the search
            hints[i] = get_leaf(leaf_idx / elts_per_leaf())
                           .lower_bound(i << 32UL, hints[i - 1]);
            if (hints[i] == typename leaf::iterator_end()) {
              leaf_idx += elts_per_leaf();
              hints[i] = get_leaf(leaf_idx / elts_per_leaf()).begin();
            }
            assert(*(hints[i]) >= i << 32UL);

          } else {
            leaf_idx = find_containing_leaf_index(i << 32U, leaf_idx, end_hint);
            hints[i] =
                get_leaf(leaf_idx / elts_per_leaf()).lower_bound(i << 32UL);
            if (hints[i] == typename leaf::iterator_end()) {
              leaf_idx += elts_per_leaf();
              hints[i] = get_leaf(leaf_idx / elts_per_leaf()).begin();
            }
            assert(*(hints[i]) >= i << 32UL);
          }
        }
      }
    });
    // passing in the data for the first head, but that will never be read so it
    // doesn't matter, just need something valid
    hints[num_nodes()] = typename leaf::iterator(index_to_head(0),
                                                 index_to_data(total_leaves()));
    return {std::unique_ptr<typename leaf::iterator, free_delete>(hints),
            getApproximateDegreeVector(hints)};
  }

  template <class F>
  void map_neighbors(uint64_t i, F f,
                     [[maybe_unused]] const std::pair<
                         std::unique_ptr<typename leaf::iterator, free_delete>,
                         std::unique_ptr<uint64_t, free_delete>> &extra_data,
                     bool parallel) const {
    auto hints = extra_data.first.get();
    assert(*(hints[i]) >= i << 32UL);

    if (parallel) {
      serial_map_with_hint_par<F::no_early_exit>(
          [&](uint64_t el) { return f(el >> 32UL, el & 0xFFFFFFFFUL); },
          (i + 1) << 32U, hints[i], hints[i + 1]);
    } else {
      serial_map_with_hint<F::no_early_exit>(
          [&](uint64_t el) { return f(el >> 32UL, el & 0xFFFFFFFFUL); },
          (i + 1) << 32U, hints[i]);
    }
  }
  template <bool no_early_exit = true, class F>
  void map_range(F f, key_type start_key, key_type end_key) const;

  template <class F>
  void map_range_length(F f, key_type start, uint64_t length) const;

  // used for the graph world
  template <class F>
  void
  map_range(F f, uint64_t start_node, uint64_t end_node,
            [[maybe_unused]] const std::pair<
                std::unique_ptr<typename leaf::iterator, free_delete>,
                std::unique_ptr<uint64_t, free_delete>> &extra_data) const {
    uint64_t start = start_node << 32UL;
    uint64_t end = end_node << 32UL;
    auto f2 = [&](uint64_t el) { return f(el >> 32UL, el & 0xFFFFFFFFUL); };
    map_range<true>(f2, start, end);
  }

  // just for an optimized compare to end
  class iterator_end {};

  // to scan over the data when it is in valid state
  // does not deal with out of place data
  class iterator {

    typename leaf::iterator it;
    uint64_t next_leaf_number;
    const CPMA &pma;

  public:
    iterator(typename leaf::iterator it_, uint64_t next_leaf_number_,
             const CPMA &pma_)
        : it(it_), next_leaf_number(next_leaf_number_), pma(pma_) {}

    // only for use comparing to end
    bool operator==([[maybe_unused]] const iterator_end &end) const {
      return next_leaf_number == pma.total_leaves() + 1;
    }
    bool operator!=([[maybe_unused]] const iterator_end &end) const {
      return !(*this == end);
    }

    iterator &operator++() {
      if (it.inc_and_check_end()) [[unlikely]] {
        if (next_leaf_number < pma.total_leaves()) {
          it = typename leaf::iterator(pma.index_to_head(next_leaf_number),
                                       pma.index_to_data(next_leaf_number));
        }
        next_leaf_number++;
      }
      return *this;
    }

    auto operator*() const { return *it; }
  };
  iterator begin() const {
    return iterator(typename leaf::iterator(index_to_head(0), index_to_data(0)),
                    1, *this);
  }
  iterator_end end() const { return {}; }

  iterator lower_bound(key_type key) const {
    if (key > max()) {
      return iterator(
          typename leaf::iterator(index_to_head(0), index_to_data(0)),
          total_leaves() + 1, *this);
    }
    uint64_t leaf_idx = find_containing_leaf_number(key);

    auto leaf_it = get_leaf(leaf_idx).lower_bound(key);
    if (leaf_it == typename leaf::iterator_end()) {
      return iterator(get_leaf(leaf_idx + 1).begin(), leaf_idx + 2, *this);
    }
    return iterator(leaf_it, leaf_idx + 1, *this);
  }

  uint64_t
  get_leaf_number_from_leaf_iterator(const typename leaf::iterator &it) const {
    uint8_t *ptr = (uint8_t *)it.get_pointer();
    uintptr_t bytes_from_start = ptr - byte_array();
    return bytes_from_start / logN();
  }
};

// when adjusting the list size, make sure you're still in the
// density bound

template <typename traits>
std::pair<float, float> CPMA<traits>::density_bound(uint64_t depth) const {
  std::pair<double, double> pair;

  // between 1/4 and 1/2
  // pair.x = 1.0/2.0 - (( .25*depth)/list->H);
  // between 1/8 and 1/4
  if (H() == 0) {
    pair.first = std::max(((double)sizeof(key_type)) / logN(), 1.0 / 4.0);
    pair.second = 1.0 / 2.0;
    if (pair.second >= density_limit()) {
      pair.second = density_limit() - .001;
    }
    assert(pair.first < pair.second);
    return pair;
  }
  pair.first = std::max(((double)sizeof(key_type)) / logN(),
                        1.0 / 4.0 - ((.125 * depth) / H()));

  pair.second = 1.0 / 2.0 + (((1.0 / 2.0) * depth) / H());

  // // TODO(wheatman) not sure why I need this
  // if (H() < 12) {
  //   pair.second = 3.0 / 4.0 + (((1.0 / 4.0) * depth) / H());
  // } else {
  //   pair.second = 15.0 / 16.0 + (((1.0 / 16.0) * depth) / H());
  // }

  if (pair.second >= density_limit()) {
    pair.second = density_limit() - .001;
  }
  return pair;
}

// doubles the size of the base array
// assumes we already have all the locks from the lock array and the big lock
template <typename traits> void CPMA<traits>::grow_list(uint64_t times) {
  // static_timer merge_timer("merge_in_double");
  // merge_timer.start();
  auto merged_data = leaf::template merge<head_form == InPlace, store_density>(
      get_data_ptr(0), total_leaves(), logN(), 0,
      [this](uint64_t index) -> element_ref_type {
        return index_to_head(index);
      },
      density_array);
  assert(((uint64_t)meta_data_index) + times <= 255);
  meta_data_index += times;
  // merge_timer.stop();

  free(data_array);
  // steal an extra few bytes to ensure we never read off the end
  uint64_t allocated_size = N() + 32;
  if (allocated_size % 32 != 0) {
    allocated_size += 32 - (allocated_size % 32);
  }
  if constexpr (binary) {
    data_array = (key_type *)aligned_alloc(32, allocated_size);
  } else {
    data_array = (key_type *)std::malloc(
        SOA_type::get_size_static(allocated_size / sizeof(key_type)));
  }
  if constexpr (head_form != InPlace) {
    free(head_array);
    head_array = (key_type *)malloc(head_array_size());
    std::fill(head_array, head_array + (head_array_size() / sizeof(key_type)),
              0);
  }

  if constexpr (store_density) {
    free(density_array);
    density_array = (uint16_t *)malloc(total_leaves() * sizeof(uint16_t));
  }

  // static_timer split_timer("split_in_double");
  // split_timer.start();
  merged_data.leaf.template split<head_form == InPlace, store_density>(
      total_leaves(), merged_data.size, logN(), get_data_ptr(0), 0,
      [this](uint64_t index) -> element_ref_type {
        return index_to_head(index);
      },
      density_array);
  // split_timer.stop();
  // -1 is since in this case the head is just being stored before the data
  merged_data.free();
  for (uint64_t i = 0; i < N(); i += logN()) {
    ASSERT(get_density_count(i, logN()) <= logN() - leaf::max_element_size,
           "%lu > %lu\n tried to split %lu bytes into %lu leaves\n i = %lu\n",
           get_density_count(i, logN()), logN() - leaf::max_element_size,
           merged_data.size, total_leaves(), i / logN());
  }
#if DEBUG == 1
  if constexpr (!compressed) {
    for (uint64_t i = 0; i < total_leaves(); i++) {
      assert(get_leaf(i).check_increasing_or_zero());
    }
  }
#endif
}

// halves the size of the base array
// assumes we already have all the locks from the lock array and the big lock
template <typename traits> void CPMA<traits>::shrink_list(uint64_t times) {
  if (meta_data_index == 0) {
    return;
  }
  auto merged_data = leaf::template merge<head_form == InPlace, store_density>(
      get_data_ptr(0), total_leaves(), logN(), 0,
      [this](uint64_t index) -> element_ref_type {
        return index_to_head(index);
      },
      density_array);
  meta_data_index -= times;

  free(data_array);
  uint64_t allocated_size = N() + 32;
  if (allocated_size % 32 != 0) {
    allocated_size += 32 - (allocated_size % 32);
  }
  if constexpr (binary) {
    data_array = (key_type *)aligned_alloc(32, allocated_size);
  } else {
    data_array = (key_type *)std::malloc(
        SOA_type::get_size_static(allocated_size / sizeof(key_type)));
  }
  if constexpr (head_form != InPlace) {
    free(head_array);
    head_array = (key_type *)malloc(head_array_size());
    std::fill(head_array, head_array + (head_array_size() / sizeof(key_type)),
              0);
  }
  if constexpr (store_density) {
    free(density_array);
    density_array = (uint16_t *)malloc(total_leaves() * sizeof(uint16_t));
  }

  merged_data.leaf.template split<head_form == InPlace, store_density>(
      total_leaves(), merged_data.size, logN(), get_data_ptr(0), 0,
      [this](uint64_t index) -> element_ref_type {
        return index_to_head(index);
      },
      density_array);
  // -1 is since in this case the head is just being stored before the data
  merged_data.free();
}

template <typename traits>
uint64_t CPMA<traits>::get_density_count(uint64_t byte_index,
                                         uint64_t len) const {
  ASSERT((byte_index / sizeof(key_type)) * sizeof(key_type) == byte_index,
         "byte_index = %lu, sizeof(key_type) == %lu, byte_index mod "
         "sizeof(key_type) = %lu\n",
         byte_index, sizeof(key_type), byte_index % sizeof(key_type));
  ASSERT((byte_index / logN()) * logN() == byte_index,
         "byte_index = %lu, logN() == %lu, byte_index mod logN() = %lu\n",
         byte_index, logN(), byte_index % logN());

  uint64_t total = 0;
  int64_t num_leaves = len / logN();
  if constexpr (PARALLEL == 1) {
    if (num_leaves > ParallelTools::getWorkers() * 1024) {
      ParallelTools::Reducer_sum<uint64_t> total_red;
      ParallelTools::parallel_for(0, num_leaves, [&](int64_t i) {
        if constexpr (store_density) {
          uint64_t val = density_array[byte_index / logN() + i];
          if (val == std::numeric_limits<uint16_t>::max()) {
            val = get_leaf(byte_index / logN() + i)
                      .template used_size<head_form == InPlace>();
          }
          total_red.add(val);
        } else {
          total_red.add(get_leaf(byte_index / logN() + i)
                            .template used_size<head_form == InPlace>());
        }
      });
      return total_red.get();
    }
  }
  for (int64_t i = 0; i < num_leaves; i++) {
    if constexpr (store_density) {
      uint64_t val = density_array[byte_index / logN() + i];
      if (val == std::numeric_limits<uint16_t>::max()) {
        val = get_leaf(byte_index / logN() + i)
                  .template used_size<head_form == InPlace>();
      }
#if DEBUG == 1
      leaf l(index_to_head(byte_index / logN() + i),
             index_to_data(byte_index / logN() + i), leaf_size_in_bytes());
      ASSERT(val == l.template used_size<head_form == InPlace>(),
             "got %lu, expected %lu, leaf_number = %lu\n", val,
             l.template used_size<head_form == InPlace>(),
             byte_index / logN() + i);
#endif
      total += val;
    } else {
      total += get_leaf(byte_index / logN() + i)
                   .template used_size<head_form == InPlace>();
    }
  }
  return total;
}

template <typename traits>
uint64_t CPMA<traits>::get_density_count_no_overflow(uint64_t byte_index,
                                                     uint64_t len) const {

  uint64_t total = 0;
  int64_t num_leaves = len / logN();
  if constexpr (PARALLEL == 1) {
    if (num_leaves > ParallelTools::getWorkers() * 1024) {
      ParallelTools::Reducer_sum<uint64_t> total_red;
      ParallelTools::parallel_for(0, num_leaves, [&](int64_t i) {
        if constexpr (store_density) {
          total_red.add(density_array[byte_index / logN() + i]);
        } else {
          total_red.add(
              get_leaf(byte_index / logN() + i)
                  .template used_size_no_overflow<head_form == InPlace>());
        }
      });
      return total_red.get();
    }
  }

  for (int64_t i = 0; i < num_leaves; i++) {
    if constexpr (store_density) {
#if DEBUG == 1
      leaf l(index_to_head(byte_index / logN() + i),
             index_to_data(byte_index / logN() + i), leaf_size_in_bytes());
      ASSERT(density_array[byte_index / logN() + i] ==
                 l.template used_size_no_overflow<head_form == InPlace>(),
             "got %d, expected %lu\n", +density_array[byte_index / logN() + i],
             l.template used_size_no_overflow<head_form == InPlace>());
#endif
      total += density_array[byte_index / logN() + i];
    } else {
      total += get_leaf(byte_index / logN() + i)
                   .template used_size_no_overflow<head_form == InPlace>();
    }
  }
  return total;
}

template <typename traits>
uint64_t CPMA<traits>::find_containing_leaf_index_debug(key_type e,
                                                        uint64_t start,
                                                        uint64_t end) const {

  if (N() == logN()) {
    return 0;
  }
  if (end > N() / sizeof(key_type)) {
    end = N() / sizeof(key_type);
  }
  assert((start * sizeof(key_type)) % logN() == 0);
  uint64_t size = (end - start) / elts_per_leaf();
  uint64_t logstep = bsr_long(size);
  uint64_t first_step = (size - (1UL << logstep));

  uint64_t step = (1UL << logstep);
  uint64_t idx = start / elts_per_leaf();
  assert(index_to_head_key(idx + first_step) != 0);
  idx = (index_to_head_key(idx + first_step) <= e) ? idx + first_step : idx;
  static constexpr uint64_t linear_cutoff = 128;
  while (step > linear_cutoff) {
    step >>= 1U;
    idx = (index_to_head_key(idx + step) <= e) ? idx + step : idx;
  }
  uint64_t end_linear = std::min(linear_cutoff, step);
  for (uint64_t i = 1; i < end_linear; i++) {
    if (index_to_head_key(idx + i) > e) {
      return (idx + i - 1) * (elts_per_leaf());
    }
  }
  return (idx + end_linear - 1) * (elts_per_leaf());
}

static_counter search_cnt("total searches");
static_counter search_steps_cnt("total_search_steps");

// searches in the unlocked array and only looks at leaf heads
// start is in PMA index, start the binary search after that point
template <typename traits>
uint64_t CPMA<traits>::find_containing_leaf_index(key_type e, uint64_t start,
                                                  uint64_t end) const {

  if (N() == logN()) {
    return 0;
  }
  search_cnt.add(1);
  if constexpr (head_form == Eytzinger) {
    if (start == 0 && end == std::numeric_limits<uint64_t>::max()) {
      uint64_t length = total_leaves_rounded_up();
      uint64_t value_to_check = length / 2;
      uint64_t length_to_add = length / 4 + 1;
      uint64_t e_index = 0;
      while (length_to_add > 0) {
        if (head_array[e_index] == e) {
          ASSERT(value_to_check * elts_per_leaf() ==
                     find_containing_leaf_index_debug(e, start, end),
                 "got %lu, expected %lu", value_to_check * elts_per_leaf(),
                 find_containing_leaf_index_debug(e, start, end));
          __builtin_prefetch(&data_array[value_to_check * (elts_per_leaf())]);
          return value_to_check * elts_per_leaf();
        }
        if (e <= static_cast<key_type>(head_array[e_index] - 1)) {
          e_index = 2 * e_index + 1;
          value_to_check -= length_to_add;
        } else {
          e_index = 2 * e_index + 2;
          value_to_check += length_to_add;
        }
        // __builtin_prefetch(&head_array[e_index * 8 + 8]);
        length_to_add /= 2;
      }
      if (value_to_check >= total_leaves()) {
        value_to_check = total_leaves() - 1;
      }
      if (e < head_array[e_index] && value_to_check > 0) {
        value_to_check -= 1;
      }
      ASSERT(value_to_check * elts_per_leaf() ==
                 find_containing_leaf_index_debug(e, start, end),
             "got %lu, expected %lu\n", value_to_check * elts_per_leaf(),
             find_containing_leaf_index_debug(e, start, end));
      __builtin_prefetch(&data_array[value_to_check * (elts_per_leaf())]);
      return value_to_check * elts_per_leaf();
    } else {
      uint64_t length = total_leaves_rounded_up();
      uint64_t value_to_check = length / 2;
      uint64_t length_to_add = length / 4 + 1;
      uint64_t e_index = 0;
      while (length_to_add > 0) {

        // if we are outside the searching range before start then we are a
        // right child
        if (value_to_check * elts_per_leaf() < start) {
          e_index = 2 * e_index + 2;
          value_to_check += length_to_add;
          length_to_add /= 2;
          continue;
        }
        if (value_to_check * elts_per_leaf() >= end) {
          e_index = 2 * e_index + 1;
          value_to_check -= length_to_add;
          length_to_add /= 2;
          continue;
        }

        if (head_array[e_index] == e) {
          ASSERT(value_to_check * elts_per_leaf() ==
                     find_containing_leaf_index_debug(e, start, end),
                 "got %lu, expected %lu\n", value_to_check * elts_per_leaf(),
                 find_containing_leaf_index_debug(e, start, end));
          __builtin_prefetch(&data_array[value_to_check * (elts_per_leaf())]);
          return value_to_check * elts_per_leaf();
        }
        if (e <= static_cast<key_type>(head_array[e_index] - 1)) {
          e_index = 2 * e_index + 1;
          value_to_check -= length_to_add;
        } else {
          e_index = 2 * e_index + 2;
          value_to_check += length_to_add;
        }
        length_to_add /= 2;
      }
      if (value_to_check >= total_leaves()) {
        value_to_check = total_leaves() - 1;
      }
      if (value_to_check * elts_per_leaf() >= end) {
        value_to_check = (end / elts_per_leaf()) - 1;
      } else if (value_to_check * elts_per_leaf() <= start) {
        value_to_check = start / elts_per_leaf();
      } else if (e < head_array[e_index] && value_to_check > 0) {
        value_to_check -= 1;
      }
      ASSERT(value_to_check * elts_per_leaf() ==
                 find_containing_leaf_index_debug(e, start, end),
             "got %lu, expected %lu, elts_per_leaf = %lu, start = %lu, end = "
             "%lu\n",
             value_to_check * elts_per_leaf(),
             find_containing_leaf_index_debug(e, start, end), elts_per_leaf(),
             start, end);
      __builtin_prefetch(&data_array[value_to_check * (elts_per_leaf())]);
      return value_to_check * elts_per_leaf();
    }
  }
  if constexpr (head_form == BNary) {
    if (start == 0 && end == std::numeric_limits<uint64_t>::max()) {

      uint64_t block_number = 0;
      uint64_t leaf_index = 0;
      uint64_t amount_to_add = total_leaves_rounded_up() / B_size;
      while (amount_to_add >= 1) {
        uint64_t number_in_block_greater_item = 0;
#ifdef __AVX2NO__
        if constexpr ((B_size - 1) % 4 == 0 && sizeof(key_type) == 8) {
          const __m256i e_vec = _mm256_set1_epi64x(e);
          const __m256i zero_vec = _mm256_setzero_si256();
          if constexpr (B_size - 1 == 4) {
            __m256i data = _mm256_loadu_si256(
                (__m256i *)&head_array[block_number * (B_size - 1)]);
            __m256i equal_zero = _mm256_cmpeq_epi64(data, zero_vec);
            __m256i greater = _mm256_cmpgt_epi64(data, e_vec);
            __m256i cells_to_count = _mm256_or_si256(equal_zero, greater);
            number_in_block_greater_item +=
                __builtin_popcount(_mm256_movemask_epi8(cells_to_count)) / 8;
          }
          if constexpr (B_size - 1 == 8) {
            __m256i data1 = _mm256_loadu_si256(
                (__m256i *)&head_array[block_number * (B_size - 1)]);
            __m256i data2 = _mm256_loadu_si256(
                (__m256i *)&head_array[block_number * (B_size - 1) + 4]);
            __m256i equal_zero1 = _mm256_cmpeq_epi64(data1, zero_vec);
            __m256i equal_zero2 = _mm256_cmpeq_epi64(data2, zero_vec);
            __m256i greater1 = _mm256_cmpgt_epi64(data1, e_vec);
            __m256i greater2 = _mm256_cmpgt_epi64(data2, e_vec);
            __m256i cells_to_count1 = _mm256_or_si256(equal_zero1, greater1);
            __m256i cells_to_count2 = _mm256_or_si256(equal_zero2, greater2);
            __m256i cells_to_count =
                _mm256_blend_epi32(cells_to_count1, cells_to_count2, 0x55);
            number_in_block_greater_item +=
                __builtin_popcount(_mm256_movemask_epi8(cells_to_count)) / 4;
          }
          if constexpr (B_size - 1 == 16) {
            __m256i data1 = _mm256_loadu_si256(
                (__m256i *)&head_array[block_number * (B_size - 1)]);
            __m256i data2 = _mm256_loadu_si256(
                (__m256i *)&head_array[block_number * (B_size - 1) + 4]);
            __m256i data3 = _mm256_loadu_si256(
                (__m256i *)&head_array[block_number * (B_size - 1) + 8]);
            __m256i data4 = _mm256_loadu_si256(
                (__m256i *)&head_array[block_number * (B_size - 1) + 12]);
            __m256i equal_zero1 = _mm256_cmpeq_epi64(data1, zero_vec);
            __m256i equal_zero2 = _mm256_cmpeq_epi64(data2, zero_vec);
            __m256i equal_zero3 = _mm256_cmpeq_epi64(data3, zero_vec);
            __m256i equal_zero4 = _mm256_cmpeq_epi64(data4, zero_vec);
            __m256i greater1 = _mm256_cmpgt_epi64(data1, e_vec);
            __m256i greater2 = _mm256_cmpgt_epi64(data2, e_vec);
            __m256i greater3 = _mm256_cmpgt_epi64(data3, e_vec);
            __m256i greater4 = _mm256_cmpgt_epi64(data4, e_vec);
            __m256i cells_to_count1 = _mm256_or_si256(equal_zero1, greater1);
            __m256i cells_to_count2 = _mm256_or_si256(equal_zero2, greater2);
            __m256i cells_to_count3 = _mm256_or_si256(equal_zero3, greater3);
            __m256i cells_to_count4 = _mm256_or_si256(equal_zero4, greater4);
            __m256i cells_to_counta =
                _mm256_blend_epi32(cells_to_count1, cells_to_count2, 0x55);
            __m256i cells_to_countb =
                _mm256_blend_epi32(cells_to_count3, cells_to_count4, 0x55);
            __m256i cells_to_count =
                _mm256_blend_epi16(cells_to_counta, cells_to_countb, 0x55);
            number_in_block_greater_item +=
                __builtin_popcount(_mm256_movemask_epi8(cells_to_count)) / 2;
          }
        } else
#endif
        {
          for (uint64_t i = 0; i < B_size - 1; i++) {
            number_in_block_greater_item +=
                head_array[block_number * (B_size - 1) + i] > e;
          }

          for (uint64_t i = 0; i < B_size - 1; i++) {
            number_in_block_greater_item +=
                head_array[block_number * (B_size - 1) + i] == 0;
          }
        }
        uint64_t child_number = B_size - number_in_block_greater_item - 1;
        leaf_index += amount_to_add * child_number;
        amount_to_add /= B_size;
        block_number = block_number * B_size + child_number + 1;
      }

      if (leaf_index > 0 && index_to_head_key(leaf_index) > e) {
        leaf_index -= 1;
      }

      if (leaf_index >= total_leaves()) {
        leaf_index = total_leaves() - 1;
      }

      ASSERT(leaf_index * elts_per_leaf() ==
                 find_containing_leaf_index_debug(e, start, end),
             "got %lu, expected %lu\n", leaf_index * elts_per_leaf(),
             find_containing_leaf_index_debug(e, start, end));
      __builtin_prefetch(&data_array[leaf_index * (elts_per_leaf())]);
      return leaf_index * elts_per_leaf();
    }

    uint64_t block_number = 0;
    uint64_t leaf_index = 0;
    uint64_t amount_to_add = total_leaves_rounded_up() / B_size;
    while (amount_to_add >= 1) {
      uint64_t number_in_block_greater_item = 0;
      for (uint64_t i = 0; i < B_size - 1; i++) {
        if (leaf_index + (amount_to_add * (i + 1)) - 1 <
            start / elts_per_leaf()) {
          continue;
        } else if (leaf_index + (amount_to_add * (i + 1)) - 1 >=
                   end / elts_per_leaf()) {
          number_in_block_greater_item += 1;
        } else {
          ASSERT(head_array[block_number * (B_size - 1) + i] != 0,
                 "looked at a zero entry, shouldn't have happened, "
                 "block_number = %lu, i = %lu, amount_to_add = %lu, start = "
                 "%lu, end = %lu, leaf_index = %lu\n",
                 block_number, i, amount_to_add, start, end, leaf_index);
          number_in_block_greater_item +=
              head_array[block_number * (B_size - 1) + i] > e;
        }
      }
      uint64_t child_number = B_size - number_in_block_greater_item - 1;
      leaf_index += amount_to_add * child_number;
      amount_to_add /= B_size;
      block_number = block_number * B_size + child_number + 1;
    }
    bool in_range = true;
    if (leaf_index < start / elts_per_leaf()) {
      leaf_index = start / elts_per_leaf();
      in_range = false;
    }

    if (leaf_index >= end / elts_per_leaf()) {
      leaf_index = (end / elts_per_leaf()) - 1;
      in_range = false;
    }

    if (in_range && leaf_index > 0 && leaf_index > start / elts_per_leaf() &&
        index_to_head_key(leaf_index) > e) {
      leaf_index -= 1;
    }

    if (leaf_index >= total_leaves()) {
      leaf_index = total_leaves() - 1;
    }

    ASSERT(leaf_index * elts_per_leaf() ==
               find_containing_leaf_index_debug(e, start, end),
           "got %lu, expected %lu, start = "
           "%lu, end = %lu, leaf_index = %lu, elts_per_leaf = %lu, "
           "total_leaves = %lu\n",
           leaf_index * elts_per_leaf(),
           find_containing_leaf_index_debug(e, start, end), start, end,
           leaf_index, elts_per_leaf(), total_leaves());
    __builtin_prefetch(&data_array[leaf_index * (elts_per_leaf())]);
    return leaf_index * elts_per_leaf();
  }
  ASSERT(end > start, "end = %lu, start = %lu\n", end, start);
  if (end > N() / sizeof(key_type)) {
    end = N() / sizeof(key_type);
  }
  assert((start * sizeof(key_type)) % logN() == 0);
  uint64_t size = (end - start) / elts_per_leaf();
  uint64_t logstep = bsr_long(size);
  uint64_t first_step = (size - (1UL << logstep));

  uint64_t step = (1UL << logstep);
  uint64_t idx = start / elts_per_leaf();
  assert(idx < total_leaves());
  assert(idx + first_step < total_leaves());
  assert(index_to_head_key(idx + first_step) != 0);
  idx = (index_to_head_key(idx + first_step) <= e) ? idx + first_step : idx;
  static constexpr uint64_t linear_cutoff = (head_form == InPlace) ? 1 : 128;
  while (step > linear_cutoff) {
    search_steps_cnt.add(1);
    step >>= 1U;
    assert(idx < total_leaves());
    assert(index_to_head_key(idx + step) != 0);
    idx = (index_to_head_key(idx + step) <= e) ? idx + step : idx;
  }
  uint64_t end_linear = std::min(linear_cutoff, step);
  for (uint64_t i = 1; i < end_linear; i++) {
    if (index_to_head_key(idx + i) > e) {
      __builtin_prefetch(&data_array[(idx + i - 1) * (elts_per_leaf())]);
      return (idx + i - 1) * (elts_per_leaf());
    }
  }
  __builtin_prefetch(&data_array[(idx + end_linear - 1) * (elts_per_leaf())]);
  return (idx + end_linear - 1) * (elts_per_leaf());
}

template <typename traits>
void CPMA<traits>::print_array_region(uint64_t start_leaf,
                                      uint64_t end_leaf) const {
  for (uint64_t i = start_leaf; i < end_leaf; i++) {
    printf("LEAF NUMBER %lu, STARTING IDX %lu, BYTE IDX %lu", i,
           i * elts_per_leaf(), i * logN());
    if constexpr (store_density) {
      printf(", density_array says: %d", +density_array[i]);
    }
    printf("\n");
    get_leaf(i).print();
  }
}

template <typename traits> void CPMA<traits>::print_array() const {
  print_array_region(0, total_leaves());
}

template <typename traits> void CPMA<traits>::print_pma() const {
  printf("N = %lu, logN = %lu, loglogN = %lu, H = %lu\n", N(), logN(),
         loglogN(), H());
  printf("count_elements %lu\n", get_element_count());
  if (has_0) {
    printf("has 0\n");
  }
  if (get_element_count()) {
    print_array();
  } else {
    printf("The PMA is empty\n");
  }
}

template <typename traits> CPMA<traits>::CPMA() {

  uint64_t allocated_size = N() + 32;
  if (allocated_size % 32 != 0) {
    allocated_size += 32 - (allocated_size % 32);
  }
  if constexpr (binary) {
    data_array = (key_type *)aligned_alloc(32, allocated_size);
  } else {
    data_array = (key_type *)std::malloc(
        SOA_type::get_size_static(allocated_size / sizeof(key_type)));
  }
  std::fill(byte_array(), byte_array() + N(), 0);
  if constexpr (head_form != InPlace) {
    head_array = (key_type *)malloc(head_array_size());
    std::fill(head_array, head_array + (head_array_size() / sizeof(key_type)),
              0);
  }
  if constexpr (store_density) {
    density_array = (uint16_t *)malloc(total_leaves() * sizeof(uint16_t));
  }
}

template <typename traits> CPMA<traits>::CPMA(key_type *start, key_type *end) {

  uint64_t allocated_size = N() + 32;
  if (allocated_size % 32 != 0) {
    allocated_size += 32 - (allocated_size % 32);
  }
  if constexpr (binary) {
    data_array = (key_type *)aligned_alloc(32, allocated_size);
  } else {
    data_array = (key_type *)std::malloc(
        SOA_type::get_size_static(allocated_size / sizeof(key_type)));
  }
  std::fill(byte_array(), byte_array() + N(), 0);
  if constexpr (head_form != InPlace) {
    head_array = (key_type *)malloc(head_array_size());
    std::fill(head_array, head_array + (head_array_size() / sizeof(key_type)),
              0);
  }
  if constexpr (store_density) {
    density_array = (uint16_t *)malloc(total_leaves() * sizeof(uint16_t));
  }
  insert_batch(start, end - start);
}

template <typename traits>
CPMA<traits>::CPMA(const CPMA<traits> &source)
    : meta_data_index(source.meta_data_index), has_0(source.has_0) {
  uint64_t allocated_size = N() + 32;
  if (allocated_size % 32 != 0) {
    allocated_size += 32 - (allocated_size % 32);
  }
  if constexpr (binary) {
    data_array = (key_type *)aligned_alloc(32, allocated_size);
  } else {
    data_array = (key_type *)std::malloc(
        SOA_type::get_size_static(allocated_size / sizeof(key_type)));
  }
  std::copy(source.data_array, &source.data_array[N() / sizeof(key_type)],
            data_array);
  if constexpr (head_form != InPlace) {
    head_array = (key_type *)malloc(head_array_size());
    std::copy(source.head_array,
              &source.head_array[head_array_size() / sizeof(key_type)],
              head_array);
  }
  if constexpr (store_density) {
    density_array = (uint16_t *)malloc(total_leaves() * sizeof(uint16_t));
    std::copy(source.density_array, &source.density_array[total_leaves()],
              density_array);
  }
}

template <typename traits> bool CPMA<traits>::has(key_type e) const {
  if (get_element_count() == 0) {
    return false;
  }
  if (e == 0) {
    return has_0;
  }
  uint64_t leaf_number = find_containing_leaf_number(e);
  // std::cout << "has(" << e << ") leaf_start = " << leaf_start << std::endl;
  return get_leaf(leaf_number).template contains<head_form == InPlace>(e);
}

template <typename traits>
typename traits::value_type CPMA<traits>::value(key_type e) const {
  static_assert(!binary);
  if (get_element_count() == 0) {
    return {};
  }
  // TODO(wheatman) deal with the value for zero
  if (e == 0) {
    if (has_0) {
      assert(false);
      // TODO(wheatman) deal with the value of 0
      return {};
    } else {
      return {};
    }
  }
  uint64_t leaf_number = find_containing_leaf_number(e);
  return get_leaf(leaf_number).template value<head_form == InPlace>(e);
}

// input: ***sorted*** batch, number of elts in a batch
// return true if the element was inserted, false if it was already there
// return number of things inserted (not already there)
// can only merge at least one leaf at a time (not multiple threads on the
// same leaf) - can merge multiple leaves at a time optional arguments in
// terms of leaf idx
template <typename traits>
bool CPMA<traits>::check_leaf_heads(uint64_t start_idx, uint64_t end_idx) {
  if (get_element_count() == 0) {
    return true;
  }
  uint64_t end = std::min(end_idx * sizeof(key_type), N());
  for (uint64_t idx = start_idx * sizeof(key_type); idx < end; idx += logN()) {
    if (index_to_head_key(idx / logN()) == 0) {
      printf("\n\nLEAF %lu HEAD IS 0\n", idx / logN());
      printf(
          "idx is %lu, byte_idx = %lu, logN = %lu, N = %lu, count_elements = "
          "%lu\n",
          idx / sizeof(key_type), idx, logN(), N(), get_element_count());
      return false;
    }
  }
  return true;
}

template <typename traits> bool CPMA<traits>::check_nothing_full() {
  for (uint64_t i = 0; i < N(); i += logN()) {
    if constexpr (compressed) {
      if (get_density_count(i, logN()) >= logN() - leaf::max_element_size) {
        printf("%lu >= %lu, i = %lu\n", get_density_count(i, logN()),
               logN() - leaf::max_element_size, i / logN());
        return false;
      }
    } else {
      if (get_density_count(i, logN()) > logN() - leaf::max_element_size) {
        printf("%lu >= %lu, i = %lu\n", get_density_count(i, logN()),
               logN() - leaf::max_element_size, i / logN());
        return false;
      }
    }
  }
  return true;
}

template <bool head_in_place, typename leaf, typename key_type>
bool everything_from_batch_added(leaf l, key_type *start, key_type *end) {
  bool have_everything = true;
  if (end - start < 10000000) {
    ParallelTools::parallel_for(0, end - start, [&](size_t i) {
      if (!l.template debug_contains<head_in_place>(start[i])) {
        std::cout << "missing " << start[i] << std::endl;
        have_everything = false;
      }
    });
  }
  return have_everything;
}

// mark all leaves that exceed their density bound
template <typename traits>
uint64_t CPMA<traits>::insert_batch_internal(
    element_ptr_type e, uint64_t batch_size,
    ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>>
        &leaves_to_check,
    uint64_t start_leaf_idx, uint64_t end_leaf_idx) {
  // current_elt_ptr is ptr into the batch
  element_ptr_type current_elt_ptr = e;
  uint64_t num_elts_merged = 0;

  uint64_t prev_leaf_start = start_leaf_idx;
  while (current_elt_ptr < e + batch_size) {
    assert(check_leaf_heads(prev_leaf_start, end_leaf_idx));
    // find the leaf that the next batch element goes into
    // NOTE: so we are guaranteed that a merge will always take at least one
    // thing from the batch

#if DEBUG == 1
    uint64_t leaf_idx_debug = find_containing_leaf_index(
        current_elt_ptr.get(), prev_leaf_start, end_leaf_idx);
#endif

    uint64_t leaf_idx = prev_leaf_start;
    if (leaf_idx < end_leaf_idx - elts_per_leaf()) {
      if (index_to_head_key((leaf_idx / elts_per_leaf()) + 1) <=
          current_elt_ptr.get()) {
        leaf_idx = find_containing_leaf_index(
            current_elt_ptr.get(), leaf_idx + elts_per_leaf(), end_leaf_idx);
      }
    }

#if DEBUG == 1
    if (leaf_idx != leaf_idx_debug) {
      printf("got %lu, expected %lu, prev_leaf_start = %lu\n", leaf_idx,
             leaf_idx_debug, prev_leaf_start);
      assert(false);
    }
#endif

    assert(leaf_idx % (elts_per_leaf()) == 0);
    prev_leaf_start = leaf_idx + elts_per_leaf();
    assert(prev_leaf_start % (elts_per_leaf()) == 0);

    // merge as much of this batch as you can
    // N(), logN() are in terms of bytes
    // find_containing_leaf_index gives you in terms of elt idx
    uint64_t next_head = std::numeric_limits<uint64_t>::max(); // max int
    if (leaf_idx + elts_per_leaf() < end_leaf_idx) {
      next_head = index_to_head_key((leaf_idx / elts_per_leaf()) + 1);
    }

    // merge into leaf returns the pointer in the batch
    // takes in pointer to start of merge in batch, remaining size in batch,
    // idx of start leaf in merge, and head of the next leaf

    // returns pointer to new start in batch, number of (distinct) elements
    // merged in
    auto result =
        get_leaf(leaf_idx / elts_per_leaf())
            .template merge_into_leaf<head_form == InPlace>(
                current_elt_ptr, e.get_pointer() + batch_size, next_head);

    assert(everything_from_batch_added<head_form == InPlace>(
        get_leaf(leaf_idx / elts_per_leaf()), current_elt_ptr.get_pointer(),
        std::get<0>(result).get_pointer()));
    current_elt_ptr = std::get<0>(result);
    // number of elements merged is the number of distinct elts merged into
    // this leaf
    num_elts_merged += std::get<1>(result);

    // number of bytes used in this leaf (if exceeds logN(), merge_into_leaf
    // will have written some auxiliary memory
    auto bytes_used = std::get<2>(result);
    if (std::get<1>(result)) {
      if constexpr (store_density) {
        density_array[leaf_idx / elts_per_leaf()] = std::min(
            bytes_used,
            static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()));
      }
      ASSERT(bytes_used ==
                 get_density_count(leaf_idx * sizeof(key_type), logN()),
             "got %lu, expected %lu\n", bytes_used,
             get_density_count(leaf_idx * sizeof(key_type), logN()));
      // if exceeded leaf density bound, add self to per-worker queue for
      // leaves to rebalance
      if (bytes_used > logN() * upper_density_bound(H())) {
        leaves_to_check.push_back({leaf_idx, bytes_used});
      }
    }
  }
  return num_elts_merged;
}

template <typename traits>
uint64_t CPMA<traits>::insert_batch_internal_small_batch(
    element_ptr_type e, uint64_t batch_size,
    ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>>
        &leaves_to_check,
    uint64_t start_leaf_idx, uint64_t end_leaf_idx) {
  if (batch_size == 0) {
    return 0;
  }
  ASSERT(start_leaf_idx < end_leaf_idx,
         "start_leaf_idx = %lu, end_leaf_idx = %lu, total_size = %lu\n",
         start_leaf_idx, end_leaf_idx, N() / sizeof(key_type));
  if (batch_size == 1) {
    uint64_t leaf_number =
        find_containing_leaf_number(e.get(), start_leaf_idx, end_leaf_idx);
    auto [inserted, bytes_used] =
        get_leaf(leaf_number).template insert<head_form == InPlace>(*e);
    if (!inserted) {
      return 0;
    }

    if constexpr (store_density) {
      density_array[leaf_number] =
          std::min(bytes_used,
                   static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()));
    }
    ASSERT(bytes_used == get_density_count(leaf_number * logN(), logN()),
           "got %lu, expected %lu\n", bytes_used,
           get_density_count(leaf_number * logN(), logN()));
    if (bytes_used > logN() * upper_density_bound(H())) {
      leaves_to_check.push_back({leaf_number * elts_per_leaf(), bytes_used});
    }
    return 1;
  }

  uint64_t num_elts_merged = 0;

  if (batch_size * 10 >= (end_leaf_idx - start_leaf_idx) / elts_per_leaf()) {
    while (start_leaf_idx + elts_per_leaf() < end_leaf_idx &&
           e.get() <
               index_to_head_key((start_leaf_idx / elts_per_leaf()) + 1)) {
      auto result =
          get_leaf(start_leaf_idx / elts_per_leaf())
              .template merge_into_leaf<head_form == InPlace>(
                  e, e.get_pointer() + batch_size,
                  index_to_head_key((start_leaf_idx / elts_per_leaf()) + 1));
      num_elts_merged += std::get<1>(result);

      if (std::get<1>(result)) {
        auto bytes_used = std::get<2>(result);
        if constexpr (store_density) {
          density_array[start_leaf_idx / elts_per_leaf()] = std::min(
              bytes_used,
              static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()));
        }
        ASSERT(bytes_used ==
                   get_density_count(start_leaf_idx * sizeof(key_type), logN()),
               "got %lu, expected %lu\n", bytes_used,
               get_density_count(start_leaf_idx * sizeof(key_type), logN()));
        // if exceeded leaf density bound, add self to per-worker queue for
        // leaves to rebalance
        if (bytes_used > logN() * upper_density_bound(H())) {
          leaves_to_check.push_back({start_leaf_idx, bytes_used});
        }
      }
      uint64_t num_elements_in =
          (std::get<0>(result).get_pointer() - e.get_pointer());
      batch_size = batch_size - num_elements_in;
      if (batch_size == 0) {
        return num_elts_merged;
      }
      e = std::get<0>(result);
      start_leaf_idx += elts_per_leaf();
    }
  }

  // else we want to start with the middle element
  element_ptr_type middle = e + (batch_size / 2);
  uint64_t leaf_idx =
      find_containing_leaf_index(middle.get(), start_leaf_idx, end_leaf_idx);
  // then we want to find the first element in the batch which is in the same
  // leaf as the middle element
  assert(leaf_idx / elts_per_leaf() < total_leaves());
  key_type head = index_to_head_key(leaf_idx / elts_per_leaf());
  if (leaf_idx == 0) {
    middle = e;
  }
  while ((middle - 1 >= e) && ((middle.get(-1)) >= head)) {
    middle = middle - 1;
  }

  uint64_t next_head = std::numeric_limits<uint64_t>::max(); // max int
  if (leaf_idx + elts_per_leaf() < end_leaf_idx) {
    next_head = index_to_head_key((leaf_idx / elts_per_leaf()) + 1);
  }
  assert(middle.get() < next_head);
  auto result = get_leaf(leaf_idx / elts_per_leaf())
                    .template merge_into_leaf<head_form == InPlace>(
                        middle, e.get_pointer() + batch_size, next_head);
  // the middle element should have been merged in
  assert(std::get<0>(result) > e + (batch_size / 2));
  num_elts_merged += std::get<1>(result);

  auto bytes_used = std::get<2>(result);

  if (std::get<1>(result)) {
    if constexpr (store_density) {
      density_array[leaf_idx / elts_per_leaf()] =
          std::min(bytes_used,
                   static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()));
    }
    ASSERT(bytes_used == get_density_count(leaf_idx * sizeof(key_type), logN()),
           "got %lu, expected %lu\n", bytes_used,
           get_density_count(leaf_idx * sizeof(key_type), logN()));
    // if exceeded leaf density bound, add self to per-worker queue for
    // leaves to rebalance
    if (bytes_used > logN() * upper_density_bound(H())) {
      leaves_to_check.push_back({leaf_idx, bytes_used});
    }
  }
  // if their are elements before do them with the new bounds we have
  uint64_t ret1 = 0;
  uint64_t ret2 = 0;

  uint64_t early_stage_size = middle - e;
  uint64_t late_stage_size = 0;
  if (std::get<0>(result) < e + batch_size) {
    uint64_t late_num_elements_in = std::get<0>(result) - e;
    late_stage_size = batch_size - late_num_elements_in;
  }

  if (early_stage_size <= 20 || late_stage_size <= 20) {
    ret1 = insert_batch_internal_small_batch(
        e, early_stage_size, leaves_to_check, start_leaf_idx, leaf_idx);
    ret2 = insert_batch_internal_small_batch(
        std::get<0>(result), late_stage_size, leaves_to_check,
        leaf_idx + elts_per_leaf(), end_leaf_idx);
  } else {
    ParallelTools::par_do(
        [&]() {
          ret1 = insert_batch_internal_small_batch(
              e, early_stage_size, leaves_to_check, start_leaf_idx, leaf_idx);
        },
        [&]() {
          ret2 = insert_batch_internal_small_batch(
              std::get<0>(result), late_stage_size, leaves_to_check,
              leaf_idx + elts_per_leaf(), end_leaf_idx);
        });
  }
  num_elts_merged += ret1;
  num_elts_merged += ret2;
  return num_elts_merged;
}

// mark all leaves that exceed their density bound
template <typename traits>
uint64_t CPMA<traits>::remove_batch_internal(
    key_type *e, uint64_t batch_size,
    ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>>
        &leaves_to_check,
    uint64_t start_leaf_idx, uint64_t end_leaf_idx) {
  // current_elt_ptr is ptr into the batch
  key_type *current_elt_ptr = e;
  uint64_t num_elts_removed = 0;

  uint64_t prev_leaf_start = start_leaf_idx;
  while (current_elt_ptr < e + batch_size && prev_leaf_start < end_leaf_idx) {
    // print_pma();
    assert(get_element_count() == 0 ||
           check_leaf_heads(prev_leaf_start, end_leaf_idx));
    // find the leaf that the next batch element goes into
    // NOTE: so we are guaranteed that a merge will always take at least one
    // thing from the batch

    uint64_t leaf_idx = find_containing_leaf_index(
        *current_elt_ptr, prev_leaf_start, end_leaf_idx);
    assert(leaf_idx % (elts_per_leaf()) == 0);
    prev_leaf_start = leaf_idx + elts_per_leaf();
    assert(prev_leaf_start % (elts_per_leaf()) == 0);

    // merge as much of this batch as you can
    // N(), logN() are in terms of bytes
    // find_containing_leaf_index gives you in terms of elt idx
    uint64_t next_head = std::numeric_limits<uint64_t>::max(); // max int
    if (leaf_idx + elts_per_leaf() < end_leaf_idx) {
      next_head = index_to_head_key((leaf_idx / elts_per_leaf()) + 1);
    }

    // merge into leaf returns the pointer in the batch
    // takes in pointer to start of merge in batch, remaining size in batch,
    // idx of start leaf in merge, and head of the next leaf

    // returns pointer to new start in batch, number of (distinct) elements
    // merged in
    auto result = get_leaf(leaf_idx / elts_per_leaf())
                      .template strip_from_leaf<head_form == InPlace>(
                          current_elt_ptr, e + batch_size, next_head);
    assert(get_leaf(leaf_idx / elts_per_leaf()).head_key() < next_head);
    current_elt_ptr = std::get<0>(result);
    // number of elements merged is the number of distinct elts merged into
    // this leaf
    num_elts_removed += std::get<1>(result);

    // number of bytes used in this leaf (if exceeds logN(), merge_into_leaf
    // will have written some auxiliary memory
    auto bytes_used = std::get<2>(result);

    // if exceeded leaf density bound, add self to per-worker queue for leaves
    // to rebalance
    if (std::get<1>(result)) {
      if constexpr (store_density) {
        density_array[leaf_idx / elts_per_leaf()] = bytes_used;
      }
      ASSERT(bytes_used ==
                 get_density_count(leaf_idx * sizeof(key_type), logN()),
             "got %lu, expected %lu, removed %lu elements, used_size = %lu\n",
             bytes_used, get_density_count(leaf_idx * sizeof(key_type), logN()),
             std::get<1>(result),
             get_leaf(leaf_idx / elts_per_leaf())
                 .template used_size<head_form == InPlace>());
      if (bytes_used < logN() * lower_density_bound(H()) || bytes_used == 0) {
        leaves_to_check.push_back({leaf_idx, bytes_used});
      }
    }
  }
  return num_elts_removed;
}

template <typename traits>
std::map<uint64_t, std::pair<uint64_t, uint64_t>>
CPMA<traits>::get_ranges_to_redistibute_internal(
    std::pair<uint64_t, uint64_t> *begin,
    std::pair<uint64_t, uint64_t> *end) const {
  std::map<uint64_t, std::pair<uint64_t, uint64_t>> ranges_to_redistribute_2;

  for (auto it = begin; it < end; ++it) {
    auto p = *it;
    uint64_t current_bytes_filled = p.second;
    uint64_t leaf_idx = p.first; // pma index
    uint64_t index_len = elts_per_leaf();

    uint64_t parent = find_node(leaf_idx, (elts_per_leaf()));

    double current_density = (double)current_bytes_filled / logN();
    uint64_t byte_len = logN();
    uint64_t level = H();
    // while exceeding density bound
    while (current_density > upper_density_bound(level)) {
      byte_len *= 2;
      index_len = byte_len / sizeof(key_type);

      // start, end in index
      uint64_t old_parent = parent;
      parent = find_node(leaf_idx, index_len);
      bool left_child = parent == old_parent;

      // if the beginning of this range is already in the map, use it
      // TODO(wheatman) better map
      if (ranges_to_redistribute_2.contains(parent)) {
        if (ranges_to_redistribute_2[parent].first >= byte_len) {
          byte_len = ranges_to_redistribute_2[parent].first;
          current_bytes_filled = ranges_to_redistribute_2[parent].second;
          break;
        }
      }

      if (level > 0) {
        level--;
      }

      // if its the whole thing
      if (parent == 0 && byte_len >= N()) {
        current_bytes_filled = get_density_count(0, N());
        byte_len = N();
        break;
      }

      // if you go off the end, count this range and exit the while
      // number of leaves doesn't have to be a power of 2
      if (uint64_t end_index = parent + index_len;
          end_index > N() / sizeof(key_type)) {
        // printf("end %lu, N %lu, N idx %lu\n", end, N(), N() /
        // sizeof(key_type));
        end_index = N() / sizeof(key_type);

        current_bytes_filled = get_density_count(
            parent * sizeof(key_type), (end_index - parent) * sizeof(key_type));
        current_density = (double)current_bytes_filled /
                          ((end_index - parent) * sizeof(key_type));
        continue;
      }

      if (left_child) {
        current_bytes_filled +=
            get_density_count((parent + (index_len / 2)) * sizeof(key_type),
                              index_len * sizeof(key_type) / 2);
      } else {
        current_bytes_filled += get_density_count(
            parent * sizeof(key_type), index_len * sizeof(key_type) / 2);
      }
      ASSERT(current_bytes_filled ==
                 get_density_count(parent * sizeof(key_type),
                                   index_len * sizeof(key_type)),
             "got %lu expected %lu\n", current_bytes_filled,
             get_density_count(parent * sizeof(key_type),
                               index_len * sizeof(key_type)));

      current_density = (double)current_bytes_filled / byte_len;
    }

    if (parent + index_len > N() / sizeof(key_type)) {
      byte_len = N() - (parent * sizeof(key_type));
    }

    assert(current_bytes_filled <= density_limit() * byte_len ||
           byte_len >= N());
    ranges_to_redistribute_2[parent] = {byte_len, current_bytes_filled};
    if (parent == 0 && byte_len >= N()) {
      ranges_to_redistribute_2.clear();
      ranges_to_redistribute_2[0] = {N(), current_bytes_filled};

      break;
    }
  }
  return ranges_to_redistribute_2;
}

// TODO(wheatman) merge serial and parallel versions
template <typename traits>
[[nodiscard]] uint64_t
CPMA<traits>::get_ranges_to_redistibute_lookup_sibling_count(
    const std::vector<ParallelTools::concurrent_hash_map<uint64_t, uint64_t>>
        &ranges_check,
    uint64_t start, uint64_t length, uint64_t level, uint64_t depth) const {
  if (start * sizeof(key_type) >= N()) {
    // we are a sibling off the end
    return 0;
  }
  if (level == 0) {
    return get_density_count(start * sizeof(key_type), logN());
  }
  uint64_t value = ranges_check[level].unlocked_value(
      start, std::numeric_limits<uint64_t>::max());
  if (value == std::numeric_limits<uint64_t>::max()) {
    // look up the children
    // if (level < 3) {
    //   if (start + length > N() / sizeof(key_type)) {
    //     length = (N() / sizeof(key_type)) - start;
    //   }
    //   return get_density_count(start * sizeof(key_type), length *
    //   sizeof(key_type));
    // }
    if (depth <= 5) {
      uint64_t left = 0;
      uint64_t right = 0;

      ParallelTools::par_do(
          [&]() {
            left = get_ranges_to_redistibute_lookup_sibling_count(
                ranges_check, start, length / 2, level - 1, depth + 1);
          },
          [&]() {
            right = get_ranges_to_redistibute_lookup_sibling_count(
                ranges_check, start + length / 2, length / 2, level - 1,
                depth + 1);
          });
      return left + right;
    } else {
      uint64_t left = get_ranges_to_redistibute_lookup_sibling_count(
          ranges_check, start, length / 2, level - 1, depth + 1);
      uint64_t right = get_ranges_to_redistibute_lookup_sibling_count(
          ranges_check, start + length / 2, length / 2, level - 1, depth + 1);
      return left + right;
    }
  }
  return value;
}

template <typename traits>
[[nodiscard]] uint64_t
CPMA<traits>::get_ranges_to_redistibute_lookup_sibling_count_serial(
    const std::vector<ska::flat_hash_map<uint64_t, uint64_t>> &ranges_check,
    uint64_t start, uint64_t length, uint64_t level) const {
  if (start * sizeof(key_type) >= N()) {
    // we are a sibling off the end
    return 0;
  }
  if (level == 0) {
    return get_density_count(start * sizeof(key_type), logN());
  }
  auto it = ranges_check[level].find(start);
  if (it == ranges_check[level].end()) {
    // look up the children
    // if (level < 3) {
    //   if (start + length > N() / sizeof(key_type)) {
    //     length = (N() / sizeof(key_type)) - start;
    //   }
    //   return get_density_count(start * sizeof(key_type), length *
    //   sizeof(key_type));
    // }

    uint64_t left = get_ranges_to_redistibute_lookup_sibling_count_serial(
        ranges_check, start, length / 2, level - 1);
    uint64_t right = get_ranges_to_redistibute_lookup_sibling_count_serial(
        ranges_check, start + length / 2, length / 2, level - 1);
    return left + right;
  }
  return it->second;
}

template <typename traits>
std::pair<std::vector<std::tuple<uint64_t, uint64_t>>, std::optional<uint64_t>>
CPMA<traits>::get_ranges_to_redistibute_debug(
    const ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>>
        &leaves_to_check,
    uint64_t num_elts_merged) const {
  // post-process leaves_to_check into ranges
  // make one big list of leaves to redistribute

  uint64_t total_size = leaves_to_check.size();

  if (total_size * 2 >= total_leaves() ||
      num_elts_merged * 10 >= get_element_count()) {
    uint64_t current_bytes_filled = get_density_count(0, N());
    double current_density = (double)current_bytes_filled / N();
    if (current_density > upper_density_bound(0)) {
      return {{}, current_bytes_filled};
    }
  }

  // ranges to redistribute
  // start -> (length to redistribute, bytes)
  // needs to be sorted for deduplication
  // TODO(wheatman) better map
  std::map<uint64_t, std::pair<uint64_t, uint64_t>> ranges_to_redistribute_2;

  // copy into big list
  std::vector<std::pair<uint64_t, uint64_t>> leaves_to_redistribute =
      leaves_to_check.get();

  ranges_to_redistribute_2 = get_ranges_to_redistibute_internal(
      leaves_to_redistribute.data(),
      leaves_to_redistribute.data() + leaves_to_redistribute.size());

  // deduplicate ranges_to_redistribute_2 into ranges_to_redistribute_3
  std::vector<std::tuple<uint64_t, uint64_t>> ranges_to_redistribute_3;

  for (auto const &[key, value] : ranges_to_redistribute_2) {
    if (ranges_to_redistribute_3.empty()) {
      if (value.first == N() && value.second > upper_density_bound(0) * N()) {
        return {{}, value.second};
      }
      ranges_to_redistribute_3.emplace_back(key, value.first);
    } else {
      const auto &last =
          ranges_to_redistribute_3[ranges_to_redistribute_3.size() - 1];
      auto end_of_last_range =
          std::get<0>(last) + (std::get<1>(last) / sizeof(key_type));
      if (key >= end_of_last_range) {
        if (value.first == N() && value.second > upper_density_bound(0) * N()) {
          return {{}, value.second};
        }
        ranges_to_redistribute_3.emplace_back(key, value.first);
      }
    }
  }
  return {ranges_to_redistribute_3, {}};
}

template <typename traits>
template <class F>
std::pair<std::vector<std::tuple<uint64_t, uint64_t>>, std::optional<uint64_t>>
CPMA<traits>::get_ranges_to_redistibute_serial(
    const ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>>
        &leaves_to_check,
    uint64_t num_elts_merged, F bounds_check) const {
  // post-process leaves_to_check into ranges
  // make one big list of leaves to redistribute

  std::vector<uint64_t> sizes;
  uint64_t total_size = leaves_to_check.size();

  if (total_size * 2 >= total_leaves() ||
      num_elts_merged * 10 >= get_element_count()) {
    uint64_t current_bytes_filled = get_density_count(0, N());
    double current_density = (double)current_bytes_filled / N();
    if (bounds_check(0, current_density)) {
      return {{}, current_bytes_filled};
    }
  }

  // ranges to redistribute
  // start -> (length to redistribute)
  // needs to be sorted for deduplication
  std::map<uint64_t, uint64_t> ranges_to_redistribute_2;
  uint64_t full_opt = std::numeric_limits<uint64_t>::max();
  // (height) -> (start) -> (bytes_used)
  std::vector<ska::flat_hash_map<uint64_t, uint64_t>> ranges_check(2);

  uint64_t length_in_index = 2 * (elts_per_leaf());
  {
    uint64_t level = 1;
    uint64_t child_length_in_index = length_in_index / 2;

    leaves_to_check.serial_for_each(
        [&](const std::pair<uint64_t, uint64_t> &p) {
          auto &[child_range_start, child_byte_count] = p;

          uint64_t parent_range_start =
              find_node(child_range_start, length_in_index);
          uint64_t length_in_index_local = length_in_index;
          if (parent_range_start + length_in_index > N() / sizeof(key_type)) {
            length_in_index_local =
                (N() / sizeof(key_type)) - parent_range_start;
          }
          bool left_child = parent_range_start == child_range_start;

          // get sibling byte count
          uint64_t sibling_range_start =
              (left_child) ? parent_range_start + child_length_in_index
                           : parent_range_start;
          uint64_t sibling_byte_count =
              get_ranges_to_redistibute_lookup_sibling_count_serial(
                  ranges_check, sibling_range_start, child_length_in_index,
                  level - 1);
          uint64_t parent_byte_count = child_byte_count + sibling_byte_count;
          double density = ((double)parent_byte_count) /
                           (length_in_index_local * sizeof(key_type));

          if (length_in_index_local >= N() / sizeof(key_type) ||
              !bounds_check(H() - level, density)) {
            if (length_in_index_local == N() / sizeof(key_type) &&
                bounds_check(0, density)) {
              full_opt = parent_byte_count;
            }

            ranges_to_redistribute_2[parent_range_start] =
                length_in_index_local * sizeof(key_type);

          } else {
            ranges_check[level][parent_range_start] = parent_byte_count;
          }
        });
  }

  for (uint64_t level = 2; level <= get_depth(logN()) + 1; level++) {
    if (ranges_check[level - 1].empty()) {
      break;
    }
    length_in_index *= 2;
    uint64_t child_length_in_index = length_in_index / 2;

    ranges_check.emplace_back();

    uint64_t level_for_density = H() - level;
    if (level > H()) {
      level_for_density = 0;
    }

    for (const auto &p : ranges_check[level - 1]) {

      uint64_t child_range_start = p.first;
      uint64_t parent_range_start =
          find_node(child_range_start, length_in_index);
      uint64_t length_in_index_local = length_in_index;
      if (parent_range_start + length_in_index > N() / sizeof(key_type)) {
        length_in_index_local = (N() / sizeof(key_type)) - parent_range_start;
      }
      bool left_child = parent_range_start == child_range_start;
      uint64_t child_byte_count = p.second;

      // get sibling byte count
      uint64_t sibling_range_start =
          (left_child) ? parent_range_start + child_length_in_index
                       : parent_range_start;
      uint64_t sibling_byte_count =
          get_ranges_to_redistibute_lookup_sibling_count_serial(
              ranges_check, sibling_range_start, child_length_in_index,
              level - 1);
      uint64_t parent_byte_count = child_byte_count + sibling_byte_count;
      double density = ((double)parent_byte_count) /
                       (length_in_index_local * sizeof(key_type));
      if (length_in_index_local >= N() / sizeof(key_type) ||
          !bounds_check(level_for_density, density)) {
        if (length_in_index_local == N() / sizeof(key_type) &&
            bounds_check(0, density)) {
          full_opt = parent_byte_count;
        }

        ranges_to_redistribute_2[parent_range_start] =
            length_in_index_local * sizeof(key_type);

      } else {
        ranges_check[level][parent_range_start] = parent_byte_count;
      }
    }
  }

  if (full_opt != std::numeric_limits<uint64_t>::max()) {
    return {{}, full_opt};
  }

  std::vector<std::tuple<uint64_t, uint64_t>> ranges_to_redistribute_3;
  for (auto const &[key, value] : ranges_to_redistribute_2) {
    if (ranges_to_redistribute_3.empty()) {
      ranges_to_redistribute_3.emplace_back(key, value);
    } else {
      const auto &last =
          ranges_to_redistribute_3[ranges_to_redistribute_3.size() - 1];
      auto end_of_last_range =
          std::get<0>(last) + (std::get<1>(last) / sizeof(key_type));
      if (key >= end_of_last_range) {
        ranges_to_redistribute_3.emplace_back(key, value);
      }
    }
  }

  return {ranges_to_redistribute_3, {}};
}

template <typename traits>
template <class F>
std::pair<std::vector<std::tuple<uint64_t, uint64_t>>, std::optional<uint64_t>>
CPMA<traits>::get_ranges_to_redistibute(
    const ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>>
        &leaves_to_check,
    uint64_t num_elts_merged, F bounds_check) const {
  timer all("get_ranges_to_redistibute");
  all.start();
  timer quick_check("quick_check");
  // post-process leaves_to_check into ranges
  // make one big list of leaves to redistribute

  uint64_t total_size = leaves_to_check.size();

  if (total_size * 2 >= total_leaves() ||
      num_elts_merged * 10 >= get_element_count()) {
    quick_check.start();
    uint64_t current_bytes_filled = get_density_count(0, N());
    quick_check.stop();
    double current_density = (double)current_bytes_filled / N();
    if (bounds_check(0, current_density)) {
      all.stop();
      return {{}, current_bytes_filled};
    }
  }

  // leaves_to_check[i].clear();

  if ((ParallelTools::getWorkers() == 1 ||
       total_size < ParallelTools::getWorkers() * 100U)) {
    timer serial("serial");
    serial.start();

    // ranges to redistribute
    // start -> (length to redistribute)
    // needs to be sorted for deduplication
    tlx::btree_map<uint64_t, uint64_t> ranges_to_redistribute_2;
    uint64_t full_opt = std::numeric_limits<uint64_t>::max();
    // (height) -> (start) -> (bytes_used)
    std::vector<ska::flat_hash_map<uint64_t, uint64_t>> ranges_check(2);

    uint64_t length_in_index = 2 * (elts_per_leaf());
    {
      uint64_t level = 1;
      uint64_t child_length_in_index = length_in_index / 2;

      leaves_to_check.serial_for_each(
          [&](const std::pair<uint64_t, uint64_t> &p) {
            auto &[child_range_start, child_byte_count] = p;
            uint64_t parent_range_start =
                find_node(child_range_start, length_in_index);
            uint64_t length_in_index_local = length_in_index;
            if (parent_range_start + length_in_index > N() / sizeof(key_type)) {
              length_in_index_local =
                  (N() / sizeof(key_type)) - parent_range_start;
            }
            bool left_child = parent_range_start == child_range_start;

            // get sibling byte count
            uint64_t sibling_range_start =
                (left_child) ? parent_range_start + child_length_in_index
                             : parent_range_start;
            uint64_t sibling_byte_count =
                get_ranges_to_redistibute_lookup_sibling_count_serial(
                    ranges_check, sibling_range_start, child_length_in_index,
                    level - 1);
            uint64_t parent_byte_count = child_byte_count + sibling_byte_count;
            double density = ((double)parent_byte_count) /
                             (length_in_index_local * sizeof(key_type));

            if (length_in_index_local >= N() / sizeof(key_type) ||
                !bounds_check(H() - level, density)) {
              if (length_in_index_local == N() / sizeof(key_type) &&
                  bounds_check(0, density)) {
                full_opt = parent_byte_count;
                return;
              }

              ranges_to_redistribute_2[parent_range_start] =
                  length_in_index_local * sizeof(key_type);

            } else {
              ranges_check[level][parent_range_start] = parent_byte_count;
            }
          });
      if (full_opt != std::numeric_limits<uint64_t>::max()) {
        serial.stop();
        all.stop();
        return {{}, full_opt};
      }
    }

    for (uint64_t level = 2; level <= get_depth(logN()) + 1; level++) {
      if (ranges_check[level - 1].empty()) {
        break;
      }
      length_in_index *= 2;
      uint64_t child_length_in_index = length_in_index / 2;

      ranges_check.emplace_back();

      uint64_t level_for_density = H() - level;
      if (level > H()) {
        level_for_density = 0;
      }

      for (const auto &p : ranges_check[level - 1]) {

        uint64_t child_range_start = p.first;
        uint64_t parent_range_start =
            find_node(child_range_start, length_in_index);
        uint64_t length_in_index_local = length_in_index;
        if (parent_range_start + length_in_index > N() / sizeof(key_type)) {
          length_in_index_local = (N() / sizeof(key_type)) - parent_range_start;
        }
        bool left_child = parent_range_start == child_range_start;
        uint64_t child_byte_count = p.second;

        // get sibling byte count
        uint64_t sibling_range_start =
            (left_child) ? parent_range_start + child_length_in_index
                         : parent_range_start;
        uint64_t sibling_byte_count =
            get_ranges_to_redistibute_lookup_sibling_count_serial(
                ranges_check, sibling_range_start, child_length_in_index,
                level - 1);
        uint64_t parent_byte_count = child_byte_count + sibling_byte_count;
        double density = ((double)parent_byte_count) /
                         (length_in_index_local * sizeof(key_type));

        if (length_in_index_local >= N() / sizeof(key_type) ||
            !bounds_check(level_for_density, density)) {
          if (length_in_index_local == N() / sizeof(key_type) &&
              bounds_check(0, density)) {
            full_opt = parent_byte_count;
            serial.stop();
            all.stop();
            return {{}, full_opt};
          }

          ranges_to_redistribute_2[parent_range_start] =
              length_in_index_local * sizeof(key_type);

        } else {
          ranges_check[level][parent_range_start] = parent_byte_count;
        }
      }
    }

    if (full_opt != std::numeric_limits<uint64_t>::max()) {
      serial.stop();
      all.stop();
      return {{}, full_opt};
    }

    std::vector<std::tuple<uint64_t, uint64_t>> ranges_to_redistribute_3;
    for (auto const &[key, value] : ranges_to_redistribute_2) {
      if (ranges_to_redistribute_3.empty()) {
        ranges_to_redistribute_3.emplace_back(key, value);
      } else {
        const auto &last =
            ranges_to_redistribute_3[ranges_to_redistribute_3.size() - 1];
        auto end_of_last_range =
            std::get<0>(last) + (std::get<1>(last) / sizeof(key_type));
        if (key >= end_of_last_range) {
          ranges_to_redistribute_3.emplace_back(key, value);
        }
      }
    }
    serial.stop();
    all.stop();
    return {ranges_to_redistribute_3, {}};
  }

  {
    // ranges to redistribute
    // start -> (length to redistribute)
    // needs to be sorted for deduplication
    ParallelTools::concurrent_hash_map<uint64_t, uint64_t>
        ranges_to_redistribute_2;
    std::atomic<uint64_t> full_opt = std::numeric_limits<uint64_t>::max();

    std::vector<ParallelTools::concurrent_hash_map<uint64_t, uint64_t>>
        ranges_check(2);
    ranges_check.reserve(get_depth(logN()) + 1);
    timer first_pass("first pass");
    first_pass.start();

    uint64_t length_in_index = 2 * (elts_per_leaf());
    {
      uint64_t level = 1;
      uint64_t child_length_in_index = length_in_index / 2;

      leaves_to_check.for_each([&](const std::pair<uint64_t, uint64_t> &p) {
        const auto &[child_range_start, child_byte_count] = p;
        uint64_t parent_range_start =
            find_node(child_range_start, length_in_index);
        uint64_t length_in_index_local = length_in_index;
        if (parent_range_start + length_in_index > N() / sizeof(key_type)) {
          length_in_index_local = (N() / sizeof(key_type)) - parent_range_start;
        }
        bool left_child = parent_range_start == child_range_start;

        // get sibling byte count
        uint64_t sibling_range_start =
            (left_child) ? parent_range_start + child_length_in_index
                         : parent_range_start;
        uint64_t sibling_byte_count =
            get_ranges_to_redistibute_lookup_sibling_count(
                ranges_check, sibling_range_start, child_length_in_index,
                level - 1);
        uint64_t parent_byte_count = child_byte_count + sibling_byte_count;
        double density = ((double)parent_byte_count) /
                         (length_in_index_local * sizeof(key_type));
        if (length_in_index_local >= N() / sizeof(key_type) ||
            !bounds_check(H() - level, density)) {
          if (length_in_index_local == N() / sizeof(key_type) &&
              bounds_check(0, density)) {
            full_opt.store(parent_byte_count);
          }

          ranges_to_redistribute_2.insert_or_assign(
              parent_range_start, length_in_index_local * sizeof(key_type));
          // not theoretically true, but probably means something is wrong
          assert(length_in_index_local * sizeof(key_type) < (1UL << 60U));

        } else {
          ranges_check[level].insert_or_assign(parent_range_start,
                                               parent_byte_count);
        }
      });
    }
    first_pass.stop();
    timer second_pass("second pass");
    second_pass.start();

    for (uint64_t level = 2; level <= get_depth(logN()) + 1; level++) {
      if (ranges_check[level - 1].unlocked_empty()) {
        break;
      }
      length_in_index *= 2;
      uint64_t child_length_in_index = length_in_index / 2;
      ranges_check.emplace_back();

      uint64_t level_for_density = H() - level;
      if (level > H()) {
        level_for_density = 0;
      }

      ranges_check[level - 1].for_each([&](uint64_t child_range_start,
                                           uint64_t child_byte_count) {
        uint64_t parent_range_start =
            find_node(child_range_start, length_in_index);
        uint64_t length_in_index_local = length_in_index;
        if (parent_range_start + length_in_index > N() / sizeof(key_type)) {
          length_in_index_local = (N() / sizeof(key_type)) - parent_range_start;
        }
        bool left_child = parent_range_start == child_range_start;

        // get sibling byte count
        uint64_t sibling_range_start =
            (left_child) ? parent_range_start + child_length_in_index
                         : parent_range_start;
        uint64_t sibling_byte_count =
            get_ranges_to_redistibute_lookup_sibling_count(
                ranges_check, sibling_range_start, child_length_in_index,
                level - 1);
        uint64_t parent_byte_count = child_byte_count + sibling_byte_count;
        double density = ((double)parent_byte_count) /
                         (length_in_index_local * sizeof(key_type));
        if (length_in_index_local >= N() / sizeof(key_type) ||
            !bounds_check(level_for_density, density)) {
          if (length_in_index_local == N() / sizeof(key_type) &&
              bounds_check(0, density)) {
            full_opt.store(parent_byte_count);
          }

          ranges_to_redistribute_2.insert_or_assign(
              parent_range_start, length_in_index_local * sizeof(key_type));
          // not theoretically true, but probably means something is wrong
          assert(length_in_index_local * sizeof(key_type) < (1UL << 60U));

        } else {
          ranges_check[level].insert_or_assign(parent_range_start,
                                               parent_byte_count);
        }
      });
    }
    // _spawn ranges_check.clear();
    second_pass.stop();

    if (full_opt.load() != std::numeric_limits<uint64_t>::max()) {
      all.stop();
      return {{}, full_opt.load()};
    }
    timer finish_up("finish up");
    finish_up.start();

#if DEBUG == 1
    auto seq = ranges_to_redistribute_2.unlocked_entries();
    ParallelTools::sort(seq.begin(), seq.end());
    std::vector<std::tuple<uint64_t, uint64_t>> ranges_to_redistribute_3_debug;
    for (auto const &[key, value] : seq) {
      // not theoretically true, but probably means something is wrong
      assert(value < (1UL << 60U));
      if (ranges_to_redistribute_3_debug.empty()) {
        ranges_to_redistribute_3_debug.emplace_back(key, value);
      } else {
        const auto &last = ranges_to_redistribute_3_debug
            [ranges_to_redistribute_3_debug.size() - 1];
        auto end_of_last_range =
            std::get<0>(last) + (std::get<1>(last) / sizeof(key_type));
        if (key >= end_of_last_range) {
          ranges_to_redistribute_3_debug.emplace_back(key, value);
        }
      }
    }
#endif
    ParallelTools::Reducer_Vector<std::tuple<uint64_t, uint64_t>>
        elements_reduce;
    ranges_to_redistribute_2.for_each(
        [&](uint64_t child_range_start, uint64_t child_byte_count) {
          if (child_byte_count == 0) {
            return;
          }
          uint64_t node_size = (child_byte_count / sizeof(key_type)) * 2;
          while (node_size <= N() / sizeof(key_type)) {
            uint64_t node_start = find_node(child_range_start, node_size);
            uint64_t length =
                ranges_to_redistribute_2.unlocked_value(node_start, 0);
            if (length != child_byte_count &&
                node_start + length / sizeof(key_type) > child_range_start) {
              return;
            }
            node_size *= 2;
          }
          elements_reduce.push_back({child_range_start, child_byte_count});
        });
    finish_up.stop();
    // _sync;

#if DEBUG == 1
    auto checker = elements_reduce.get_sorted();
    if (checker.size() != ranges_to_redistribute_3_debug.size()) {
      printf("got the wrong size, got %lu, expected %lu\n", checker.size(),
             ranges_to_redistribute_3_debug.size());
      assert(false);
    }
    for (uint64_t i = 0; i < checker.size(); i++) {
      if (checker[i] != ranges_to_redistribute_3_debug[i]) {
        printf("got the wrong element in position %lu, got (%lu, %lu) expected "
               "(%lu, %lu)\n",
               i, std::get<0>(checker[i]), std::get<1>(checker[i]),
               std::get<0>(ranges_to_redistribute_3_debug[i]),
               std::get<1>(ranges_to_redistribute_3_debug[i]));
      }
    }

#endif

    all.stop();
    // return {ranges_to_redistribute_3, {}};
    return {elements_reduce.get(), {}};
  }
}

template <typename traits>
std::vector<typename CPMA<traits>::leaf_bound_t>
CPMA<traits>::get_leaf_bounds(uint64_t split_points) const {
  uint64_t leaf_stride = total_leaves() / split_points;
  std::vector<leaf_bound_t> leaf_bounds(split_points);

  // calculate split points before doing any batch insertions
  // TODO(wheatman) see why this can't be made parallel
  // right now it slows it down a ton
  ParallelTools::serial_for(0, split_points - 1, [&](uint64_t i) {
    uint64_t start_leaf_idx = i * leaf_stride * (elts_per_leaf());
    key_type start_elt = index_to_head_key(start_leaf_idx / elts_per_leaf());
    uint64_t end_leaf_idx = (i + 1) * leaf_stride * (elts_per_leaf());
    // head of leaf at start_leaf_idx
    key_type end_elt = index_to_head_key(end_leaf_idx / elts_per_leaf());

    assert(start_leaf_idx <= end_leaf_idx);
    leaf_bounds[i] = {start_elt, end_elt, start_leaf_idx, end_leaf_idx};
  });
  // last loop not in parallel due to weird compiler behavior
  {
    uint64_t i = split_points - 1;
    uint64_t start_leaf_idx = i * leaf_stride * (elts_per_leaf());
    key_type start_elt = index_to_head_key(start_leaf_idx / elts_per_leaf());
    key_type end_elt = std::numeric_limits<key_type>::max();

    uint64_t end_leaf_idx = N() / sizeof(key_type);
    assert(start_leaf_idx <= end_leaf_idx);
    leaf_bounds[i] = {start_elt, end_elt, start_leaf_idx, end_leaf_idx};
  }
  return leaf_bounds;
}

template <typename traits>
void CPMA<traits>::redistribute_ranges(
    const std::vector<std::tuple<uint64_t, uint64_t>> &ranges) {
  ParallelTools::parallel_for(0, ranges.size(), [&](uint64_t i) {
    const auto &item = ranges[i];
    auto start = std::get<0>(item);
    auto len = std::get<1>(item);

    auto merged_data =
        leaf::template merge<head_form == InPlace, store_density>(
            get_data_ptr(start), len / logN(), logN(), start / elts_per_leaf(),
            [this](uint64_t index) -> element_ref_type {
              return index_to_head(index);
            },
            density_array);

    if constexpr (std::is_same_v<leaf, delta_compressed_leaf<key_type>>) {
      assert(((double)merged_data.size) / (len / logN()) <
             (density_limit() * len));
    }

    // number of leaves, num elemtns in input leaf, num elements in
    // output leaf, dest region
    merged_data.leaf.template split<head_form == InPlace, store_density>(
        len / logN(), merged_data.size, logN(), get_data_ptr(start),
        start / elts_per_leaf(),
        [this](uint64_t index) -> element_ref_type {
          return index_to_head(index);
        },
        density_array);
    merged_data.free();
  });
}

// input: batch, number of elts in a batch
// return true if the element was inserted, false if it was already there
// return number of things inserted (not already there)
template <typename traits>
uint64_t CPMA<traits>::insert_batch(element_ptr_type e, uint64_t batch_size) {
  timer total_timer("insert_batch");
  total_timer.start();
  if (batch_size < 100) {
    uint64_t count = 0;
    for (uint64_t i = 0; i < batch_size; i++) {
      count += insert(e[i]);
    }
    return count;
  }

  // TODO(wheatman) make it work for the first batch
  if (get_element_count() == 0) {
    uint64_t count = 0;
    uint64_t end = std::min(batch_size, 1000UL);
    for (uint64_t i = 0; i < end; i++) {
      count += insert(e[i]);
    }
    if (batch_size == end) {
      total_timer.stop();
      return count;
    } else {
      e = e + end;
      batch_size -= end;
      return count + insert_batch(e, batch_size);
    }
  }

  assert(check_leaf_heads());

  timer sort_timer("sort");

  sort_timer.start();
  sort_batch(e, batch_size);
  sort_timer.stop();

  // TODO(wheatman) currently only works for unsigned types
  while (e.get() == 0) {
    has_0 = true;
    ++e;
    batch_size -= 1;
    if (batch_size == 0) {
      return 0;
    }
  }

  // total number of leaves
  uint64_t num_leaves = total_leaves();

  // which leaves were touched during the merge
  ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>> leaves_to_check;

  uint64_t num_elts_merged = 0;

  timer merge_timer("merge_timer");
  merge_timer.start();

  if (false) {
    ParallelTools::Reducer_sum<uint64_t> num_elts_merged_reduce;
    // leaves per partition
    uint64_t split_points =
        std::min({(uint64_t)num_leaves / 10, (uint64_t)batch_size / 100,
                  (uint64_t)ParallelTools::getWorkers() * 10});
    split_points = std::max(split_points, 1UL);

    std::vector<leaf_bound_t> leaf_bounds = get_leaf_bounds(split_points);

    ParallelTools::parallel_for(0, split_points, [&](uint64_t i) {
      // search for boundaries in batch
      key_type *batch_start_key =
          std::lower_bound(e.get_pointer(), e.get_pointer() + batch_size,
                           leaf_bounds[i].start_elt);
      // if we are the first batch start at the begining
      element_ptr_type batch_start = e + (batch_start_key - e.get_pointer());
      if (i == 0) {
        batch_start = e;
      }
      uint64_t end_elt = leaf_bounds[i].end_elt;
      if (i == split_points - 1) {
        end_elt = std::numeric_limits<uint64_t>::max();
      }
      key_type *batch_end_key = std::lower_bound(
          e.get_pointer(), e.get_pointer() + batch_size, end_elt);
      if (batch_start.get_pointer() == batch_end_key ||
          batch_start.get_pointer() == e.get_pointer() + batch_size) {
        return;
      }
      // number of elts we are merging
      uint64_t range_size = uint64_t(batch_end_key - batch_start.get_pointer());
      // do the merge
      num_elts_merged_reduce.add(insert_batch_internal(
          batch_start, range_size, leaves_to_check,
          leaf_bounds[i].start_leaf_index, leaf_bounds[i].end_leaf_index));
    });
    num_elts_merged = num_elts_merged_reduce.get();
  } else {
    num_elts_merged += insert_batch_internal_small_batch(
        e, batch_size, leaves_to_check, 0, num_leaves * elts_per_leaf());
  }
  merge_timer.stop();

  // if most leaves need to be redistributed, or many elements were added,
  // just check the root first to hopefully save walking up the tree
  timer range_finder_timer("range_finder_timer");
  range_finder_timer.start();
  auto [ranges_to_redistribute_3, full_opt] = get_ranges_to_redistibute(
      leaves_to_check, num_elts_merged, [&](uint64_t level, double density) {
        return density > upper_density_bound(level);
      });
  range_finder_timer.stop();

#if DEBUG == 1
  auto [ranges_debug, full_opt_debug] =
      get_ranges_to_redistibute_debug(leaves_to_check, num_elts_merged);
  if (ranges_to_redistribute_3.size() != ranges_debug.size()) {
    printf("sizes don't match, got %lu, expected %lu\n",
           ranges_to_redistribute_3.size(), ranges_debug.size());
    printf("got:\n");
    for (const auto &element : ranges_to_redistribute_3) {
      std::cout << "( " << std::get<0>(element) << ", "
                << std::get<1>(element) / sizeof(key_type) << ") ";
    }
    std::cout << std::endl;
    printf("correct:\n");
    for (const auto &element : ranges_debug) {
      std::cout << "( " << std::get<0>(element) << ", "
                << std::get<1>(element) / sizeof(key_type) << ") ";
    }
    std::cout << std::endl;
  } else {
    // Optimized code might not give these sorted
    // just sort them first to make checking easier
    std::sort(ranges_to_redistribute_3.begin(), ranges_to_redistribute_3.end());
    for (size_t i = 0; i < ranges_to_redistribute_3.size(); i++) {
      if (ranges_to_redistribute_3[i] != ranges_debug[i]) {
        printf("element %lu doesn't match, got (%lu, %lu), expected (%lu,"
               "%lu)\n",
               i, std::get<0>(ranges_to_redistribute_3[i]),
               std::get<1>(ranges_to_redistribute_3[i]),
               std::get<0>(ranges_debug[i]), std::get<1>(ranges_debug[i]));
      }
    }
  }

  assert(ranges_to_redistribute_3 == ranges_debug);
  assert(full_opt == full_opt_debug);
#endif

  // doubling everything
  if (full_opt.has_value()) {
    timer double_timer("doubling");
    double_timer.start();

    uint64_t target_size = N();
    uint64_t grow_times = 0;
    auto bytes_occupied = full_opt.value();

    // min bytes necessary to meet the density bound
    // uint64_t bytes_required = bytes_occupied / upper_density_bound(0);
    uint64_t bytes_required =
        std::max(N() * growing_factor, bytes_occupied * growing_factor);

    while (target_size <= bytes_required) {
      target_size *= growing_factor;
      grow_times += 1;
    }

    grow_list(grow_times);
    double_timer.stop();
  } else { // not doubling
    // in parallel, redistribute ranges

    timer redistribute_timer("redistributing");
    redistribute_timer.start();
    redistribute_ranges(ranges_to_redistribute_3);
    redistribute_timer.stop();
  }
  assert(check_nothing_full());

  assert(check_leaf_heads());
  count_elements_ += num_elts_merged;
  total_timer.stop();
  return num_elts_merged;
}

// input: batch, number of elts in a batch
// return number of things removed
template <typename traits>
uint64_t CPMA<traits>::remove_batch(key_type *e, uint64_t batch_size) {
  assert(check_leaf_heads());
  static_timer total_timer("remove_batch");
  total_timer.start();

  if (get_element_count() == 0 || batch_size == 0) {
    return 0;
  }
  if (batch_size <= 100) {
    uint64_t count = 0;
    for (uint64_t i = 0; i < batch_size; i++) {
      count += remove(e[i]);
    }
    return count;
  }

  static_timer sort_timer("sort_remove_batch");

  sort_timer.start();
  sort_batch(e, batch_size);
  sort_timer.stop();

  // TODO(wheatman) currently only works for unsigned types
  while (*e == 0) {
    has_0 = false;
    e += 1;
    batch_size -= 1;
    if (batch_size == 0) {
      return 0;
    }
  }

  // total number of leaves
  uint64_t num_leaves = total_leaves();

  // leaves per partition
  uint64_t split_points =
      std::min({(uint64_t)num_leaves / 10, (uint64_t)batch_size / 100,
                (uint64_t)ParallelTools::getWorkers() * 10});
  split_points = std::max(split_points, 1UL);

  // which leaves were touched during the merge
  ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>> leaves_to_check;

  uint64_t num_elts_removed = 0;

  std::vector<leaf_bound_t> leaf_bounds = get_leaf_bounds(split_points);

  ParallelTools::Reducer_sum<uint64_t> num_elts_removed_reduce;

  static_timer merge_timer("merge_timer_remove_batch");
  merge_timer.start();
  ParallelTools::parallel_for(0, split_points, [&](uint64_t i) {
    if (leaf_bounds[i].start_leaf_index == leaf_bounds[i].end_leaf_index) {
      return;
    }

    // search for boundaries in batch
    key_type *batch_start =
        std::lower_bound(e, e + batch_size, leaf_bounds[i].start_elt);
    // if we are the first batch start at the begining
    if (i == 0) {
      batch_start = e;
    }
    uint64_t end_elt = leaf_bounds[i].end_elt;
    if (i == split_points - 1) {
      end_elt = std::numeric_limits<uint64_t>::max();
    }
    key_type *batch_end = std::lower_bound(e, e + batch_size, end_elt);

    if (batch_start == batch_end || batch_start == e + batch_size) {
      return;
    }

    // number of elts we are merging
    uint64_t range_size = uint64_t(batch_end - batch_start);

    // do the merge
    num_elts_removed_reduce.add(remove_batch_internal(
        batch_start, range_size, leaves_to_check,
        leaf_bounds[i].start_leaf_index, leaf_bounds[i].end_leaf_index));
  });
  num_elts_removed = num_elts_removed_reduce.get();
  merge_timer.stop();

  auto ranges_pair = get_ranges_to_redistibute(
      leaves_to_check, num_elts_removed, [&](uint64_t level, double density) {
        return (density < lower_density_bound(level)) || (density == 0);
      });
  auto ranges_to_redistribute_3 = ranges_pair.first;
  assert(ranges_to_redistribute_3 ==
         get_ranges_to_redistibute_serial(
             leaves_to_check, num_elts_removed,
             [&](uint64_t level, double density) {
               return (density < lower_density_bound(level)) || (density == 0);
             })
             .first);
  auto full_opt = ranges_pair.second;

  // shrinking everything
  if (full_opt.has_value()) {
    static_timer shrinking_timer("shrinking");
    shrinking_timer.start();
    uint64_t target_size = N();
    double shrink = 1;
    auto bytes_occupied = full_opt.value();

    // min bytes necessary to meet the density bound
    uint64_t bytes_required = bytes_occupied / lower_density_bound(0);
    if (bytes_required == 0) {
      bytes_required = 1;
    }

    while (target_size >= bytes_required) {
      target_size /= growing_factor;
      shrink *= growing_factor;
    }

    shrink_list(shrink);
    shrinking_timer.stop();
  } else { // not doubling
    // in parallel, redistribute ranges

    static_timer redistribute_timer("redistributing_remove_batch");
    redistribute_timer.start();
    redistribute_ranges(ranges_to_redistribute_3);
    redistribute_timer.stop();
  }
  count_elements_ -= num_elts_removed;
  assert(check_nothing_full());
  assert(check_leaf_heads());
  total_timer.stop();
  return num_elts_removed;
}

template <typename traits>
void CPMA<traits>::insert_post_place(uint64_t leaf_number,
                                     uint64_t byte_count) {
  static_timer rebalence_timer("rebalence_insert_timer");

  count_elements_ += 1;
  if constexpr (store_density) {
    density_array[leaf_number] = byte_count;
  }
  rebalence_timer.start();
  const uint64_t byte_index = leaf_number * logN();
  ASSERT(byte_count == get_density_count(byte_index, logN()),
         "got %lu, expected %lu\n", byte_count,
         get_density_count(byte_index, logN()));

  uint64_t level = H();
  uint64_t len_bytes = logN();

  uint64_t node_byte_index = byte_index;

  // if we are not a power of 2, we don't want to go off the end
  uint64_t local_len_bytes = std::min(len_bytes, N() - node_byte_index);

  while (byte_count >= upper_density_bound(level) * local_len_bytes) {
    len_bytes *= 2;

    if (len_bytes <= N()) {
      if (level > 0) {
        level--;
      }
      uint64_t new_byte_node_index = find_node(node_byte_index, len_bytes);
      local_len_bytes = std::min(len_bytes, N() - new_byte_node_index);

      if (local_len_bytes == len_bytes) {
        if (new_byte_node_index < node_byte_index) {
          byte_count +=
              get_density_count_no_overflow(new_byte_node_index, len_bytes / 2);
        } else {
          byte_count += get_density_count_no_overflow(
              new_byte_node_index + len_bytes / 2, len_bytes / 2);
        }
      } else {
        // since its to the left it can never leave the range
        if (new_byte_node_index < node_byte_index) {
          byte_count +=
              get_density_count_no_overflow(new_byte_node_index, len_bytes / 2);
        } else {
          uint64_t length = len_bytes / 2;
          if (new_byte_node_index + len_bytes > N()) {
            length = N() - (new_byte_node_index + (len_bytes / 2));
          }
          // only count if there were new real elements
          if (N() > new_byte_node_index + (len_bytes / 2)) {
            byte_count += get_density_count_no_overflow(
                new_byte_node_index + len_bytes / 2, length);
          }
        }
      }

      node_byte_index = new_byte_node_index;
    } else {
      grow_list(1);
      rebalence_timer.stop();
      return;
    }
  }
  if (len_bytes > logN()) {
    auto merged_data =
        leaf::template merge<head_form == InPlace, store_density>(
            get_data_ptr(node_byte_index / sizeof(key_type)),
            local_len_bytes / logN(), logN(), node_byte_index / logN(),
            [this](uint64_t index) -> element_ref_type {
              return index_to_head(index);
            },
            density_array);

    merged_data.leaf.template split<head_form == InPlace, store_density>(
        local_len_bytes / logN(), merged_data.size, logN(),
        get_data_ptr(node_byte_index / sizeof(key_type)),
        node_byte_index / logN(),
        [this](uint64_t index) -> element_ref_type {
          return index_to_head(index);
        },
        density_array);
#if DEBUG == 1
    uint64_t start = node_byte_index;
    if (node_byte_index > logN()) {
      node_byte_index -= logN();
    }
    uint64_t end = node_byte_index + local_len_bytes;
    if (end < N()) {
      end += logN();
    }
    for (uint64_t i = start; i < end; i += logN()) {
      if constexpr (compressed) {
        if (get_density_count(i, logN()) >= logN() - leaf::max_element_size) {
          merged_data.leaf.print();
          print_array_region(node_byte_index / logN(),
                             (node_byte_index + local_len_bytes) / logN());
        }
        ASSERT(get_density_count(i, logN()) < logN() - leaf::max_element_size,
               "%lu >= %lu\n tried to split %lu bytes into %lu leaves\n i = "
               "%lu\n",
               get_density_count(i, logN()), logN() - leaf::max_element_size,
               merged_data.size, local_len_bytes / logN(),
               (i - node_byte_index) / logN());
      } else {
        ASSERT(get_density_count(i, logN()) <= logN() - leaf::max_element_size,
               "%lu > %lu\n tried to split %lu bytes into %lu leaves\n i = "
               "%lu\n",
               get_density_count(i, logN()), logN() - leaf::max_element_size,
               merged_data.size, local_len_bytes / logN(),
               (i - node_byte_index) / logN());
      }
    }
#endif
    merged_data.free();
  }
  rebalence_timer.stop();
}

// return true if the element was inserted, false if it was already there
template <typename traits> bool CPMA<traits>::insert(element_type e) {
  static_timer total_timer("total_insert_timer");
  static_timer find_timer("find_insert_timer");
  static_timer modify_timer("modify_insert_timer");

  if (std::get<0>(e) == 0) {
    bool had_before = has_0;
    has_0 = true;
    return !had_before;
  }
  total_timer.start();
  find_timer.start();
  uint64_t leaf_number = find_containing_leaf_number(std::get<0>(e));
  find_timer.stop();
  modify_timer.start();
  auto [inserted, byte_count] =
      get_leaf(leaf_number).template insert<head_form == InPlace>(e);
  modify_timer.stop();

  if (!inserted) {
    total_timer.stop();
    return false;
  }
  insert_post_place(leaf_number, byte_count);
  total_timer.stop();
  return true;
}

template <typename traits>
void CPMA<traits>::remove_post_place(uint64_t leaf_number,
                                     uint64_t byte_count) {
  static_timer rebalence_timer("rebalence_remove_timer");

  count_elements_ -= 1;
  if constexpr (store_density) {
    density_array[leaf_number] = byte_count;
  }
  rebalence_timer.start();

  uint64_t node_byte_index = leaf_number * logN();

  ASSERT(byte_count == get_density_count(node_byte_index, logN()),
         "got %lu, expected %lu\n", byte_count,
         get_density_count(node_byte_index, logN()));

  uint64_t level = H();
  uint64_t len_bytes = logN();

  // if we are not a power of 2, we don't want to go off the end
  uint64_t local_len_bytes = std::min(len_bytes, N() - node_byte_index);

  while (byte_count <= lower_density_bound(level) * local_len_bytes) {

    len_bytes *= 2;

    if (len_bytes <= N()) {

      if (level > 0) {
        level--;
      }
      uint64_t new_byte_node_index = find_node(node_byte_index, len_bytes);
      local_len_bytes = std::min(len_bytes, N() - new_byte_node_index);
      if (local_len_bytes == len_bytes) {
        if (new_byte_node_index < node_byte_index) {
          byte_count +=
              get_density_count_no_overflow(new_byte_node_index, len_bytes / 2);
        } else {
          byte_count += get_density_count_no_overflow(
              new_byte_node_index + len_bytes / 2, len_bytes / 2);
        }
      } else {
        // since its to the left it can never leave the range
        if (new_byte_node_index < node_byte_index) {
          byte_count +=
              get_density_count_no_overflow(new_byte_node_index, len_bytes / 2);
        } else {
          uint64_t length = len_bytes / 2;
          if (new_byte_node_index + len_bytes > N()) {
            length = N() - (new_byte_node_index + (len_bytes / 2));
          }
          // only count if there were new real elements
          if (N() > new_byte_node_index + (len_bytes / 2)) {
            byte_count += get_density_count_no_overflow(
                new_byte_node_index + len_bytes / 2, length);
          }
        }
      }

      node_byte_index = new_byte_node_index;
    } else {
      shrink_list(1);
      rebalence_timer.stop();
      return;
    }
  }
  if (len_bytes > logN()) {
    auto merged_data =
        leaf::template merge<head_form == InPlace, store_density>(
            get_data_ptr(node_byte_index / sizeof(key_type)),
            local_len_bytes / logN(), logN(), node_byte_index / logN(),
            [this](uint64_t index) -> element_ref_type {
              return index_to_head(index);
            },
            density_array);

    merged_data.leaf.template split<head_form == InPlace, store_density>(
        local_len_bytes / logN(), merged_data.size, logN(),
        get_data_ptr(node_byte_index / sizeof(key_type)),
        node_byte_index / logN(),
        [this](uint64_t index) -> element_ref_type {
          return index_to_head(index);
        },
        density_array);
    merged_data.free();
  }
  rebalence_timer.stop();
}

// return true if the element was removed, false if it wasn't already there
template <typename traits> bool CPMA<traits>::remove(key_type e) {
  static_timer total_timer("total_remove_timer");
  static_timer find_timer("find_remove_timer");
  static_timer modify_timer("modify_remove_timer");

  if (get_element_count() == 0) {
    return false;
  }
  if (e == 0) {
    bool had_before = has_0;
    has_0 = false;
    return had_before;
  }
  total_timer.start();
  find_timer.start();
  uint64_t leaf_number = find_containing_leaf_number(e);
  find_timer.stop();
  modify_timer.start();
  auto [removed, byte_count] =
      get_leaf(leaf_number).template remove<head_form == InPlace>(e);
  modify_timer.stop();

  if (!removed) {
    total_timer.stop();
    return false;
  }
  remove_post_place(leaf_number, byte_count);

  total_timer.stop();
  return true;
}

// return the amount of memory the structure uses
template <typename traits> uint64_t CPMA<traits>::get_size() const {
  uint64_t total_size = sizeof(*this);
  uint64_t allocated_size = N() + 32;
  if (allocated_size % 32 != 0) {
    allocated_size += 32 - (allocated_size % 32);
  }
  if constexpr (binary) {
    total_size += allocated_size;
  } else {
    total_size += SOA_type::get_size_static(allocated_size / sizeof(key_type));
  }
  if constexpr (head_form != InPlace) {
    total_size += head_array_size();
  }
  if constexpr (store_density) {
    total_size += total_leaves() * sizeof(uint16_t);
  }
  return total_size;
}

template <typename traits>
uint64_t CPMA<traits>::sum_serial(int64_t start, int64_t end) const {
#ifdef __AVX512F__
  if constexpr (!compressed && std::is_same_v<key_type, uint64_t>) {
    __m512i total_vec = _mm512_setzero();
    // __m256i total_vec = _mm256_setzero_si256();
    for (int64_t i = start; i < end; i++) {
      total_vec = _mm512_add_epi64(
          total_vec, get_leaf(i).template sum512<head_form == InPlace>());
      // total_vec = _mm256_add_epi64(
      //     total_vec, l.template sum32_256<head_form == InPlace>());
    }
    // uint64_t a = _mm256_extract_epi64(total_vec, 0);
    // a += _mm256_extract_epi64(total_vec, 1);
    // a += _mm256_extract_epi64(total_vec, 2);
    // a += _mm256_extract_epi64(total_vec, 3);
    // return a;
    return _mm512_reduce_add_epi64(total_vec);
  }
#endif
  uint64_t total = 0;
  for (int64_t i = start; i < end; i++) {
    total += get_leaf(i).template sum<head_form == InPlace>();
  }
  return total;
}
// template <typename traits> inline uint64_t CPMA<traits>::sum_parallel()
// const
// {
//   ParallelTools::Reducer_sum<uint64_t> total_red;
//   uint64_t chunk_size = 100;
//   uint64_t end = total_leaves();
//   ParallelTools::parallel_for(0, end, chunk_size, [&](int64_t i) {
//     size_t local_end = i + chunk_size;
//     if (local_end > end) {
//       local_end = end;
//     }
//     uint64_t local_sum = 0;
//     for (size_t j = i; j < local_end; j++) {
//       leaf l(index_to_head(j), index_to_data(j), leaf_size_in_bytes());
//       local_sum += l.template sum<head_form == InPlace>();
//     }
//     total_red.add(local_sum);
//   });
//   return total_red.get();
// }

// template <typename traits> inline uint64_t CPMA<traits>::sum_parallel()
// const
// {
//   ParallelTools::Reducer_sum<uint64_t> total_red;
//   uint64_t end = total_leaves();
//   ParallelTools::parallel_for(0, end, [&](int64_t i) {
//     leaf l(index_to_head(i), index_to_data(i), leaf_size_in_bytes());
//     total_red.add(l.template sum<head_form == InPlace>());
//   });
//   return total_red.get();
// }

template <typename traits>
inline uint64_t CPMA<traits>::sum_parallel2(uint64_t start, uint64_t end,
                                            int depth) const {
  if (depth > 10 || end - start < 100) {
    return sum_serial(start, end);
  }
  // uint64_t sum_a;
  // cilk_spawn sum_a = sum_parallel2(start, start + (end - start) / 2, depth
  // + 1); uint64_t sum_b = sum_parallel2(start + (end - start) / 2, end,
  // depth + 1); cilk_sync; return sum_a + sum_b;

  uint64_t sum_a;
  uint64_t sum_b;
  ParallelTools::par_do(
      [&]() {
        sum_a = sum_parallel2(start, start + (end - start) / 2, depth + 1);
      },
      [&]() {
        sum_b = sum_parallel2(start + (end - start) / 2, end, depth + 1);
      });
  return sum_a + sum_b;
}

template <typename traits> inline uint64_t CPMA<traits>::sum_parallel() const {
  return sum_parallel2(0, total_leaves(), 0);
}

// return the sum of all elements stored
// just used to see how fast it takes to iterate
template <typename traits> uint64_t CPMA<traits>::sum() const {
  int64_t num_leaves = total_leaves();
#if PARALLEL == 0
  return sum_serial(0, num_leaves);
#endif
  if (num_leaves < ParallelTools::getWorkers()) {
    return sum_serial(0, num_leaves);
  }
  return sum_parallel();
}

template <typename traits>
template <bool no_early_exit, class F>
void CPMA<traits>::map_range(F f, key_type start_key, key_type end_key) const {
#pragma clang loop unroll_count(4)
  for (auto it = lower_bound(start_key); it != end(); ++it) {
    auto el = *it;
    if (el >= end_key) {
      return;
    }
    assert(el >= start_key && el < end_key);
    if constexpr (!no_early_exit) {
      if (f(el)) {
        return;
      }
    } else {
      f(el);
    }
  }
}

template <typename traits>
template <class F>
void CPMA<traits>::map_range_length(F f, key_type start,
                                    uint64_t length) const {
  if (length == 0) {
    return;
  }
#pragma clang loop unroll_count(4)
  for (auto it = lower_bound(start); it != end(); ++it) {
    auto el = *it;
    assert(el >= start);
    f(el);
    length -= 1;
    if (length == 0) {
      return;
    }
  }
}

template <typename traits>
template <bool no_early_exit, class F>
bool CPMA<traits>::map(F f) const {
  // skips the all zeros element, but fine for now since its a self loop and
  // we don't care about it
#pragma clang loop unroll_count(4)
  for (auto el : *this) {
    if constexpr (!no_early_exit) {
      if (f(el)) {
        return true;
      }
    } else {
      f(el);
    }
  }
  return false;
}

template <typename traits>
template <bool no_early_exit, class F>
void CPMA<traits>::serial_map_with_hint(
    F f, key_type end_key, const typename leaf::iterator &hint) const {

  uint64_t leaf_number = get_leaf_number_from_leaf_iterator(hint);
  iterator it(hint, leaf_number + 1, *this);
  for (; it != end(); ++it) {
    if (*it >= end_key) {
      return;
    }
    if (f(*it)) {
      if constexpr (!no_early_exit) {
        return;
      }
    }
  }
}

template <typename traits>
template <bool no_early_exit, class F>
void CPMA<traits>::serial_map_with_hint_par(
    F f, key_type end_key, const typename leaf::iterator &hint,
    const typename leaf::iterator &end_hint) const {

  uint64_t leaf_number = get_leaf_number_from_leaf_iterator(hint);
  uint64_t leaf_number_end = get_leaf_number_from_leaf_iterator(end_hint);

  // do the first leaf
  for (auto it = hint; it != typename leaf::iterator_end(); ++it) {
    if (*it >= end_key) {
      return;
    }
    if (f(*it)) {
      if constexpr (!no_early_exit) {
        return;
      }
    }
  }
  leaf_number += 1;
  // if there is an leaves in the middle do them in parallel
  if (leaf_number < leaf_number_end) {
    ParallelTools::parallel_for(leaf_number, leaf_number_end,
                                [&](uint64_t idx) {
                                  if (index_to_head_key(idx) >= end_key) {
                                    return;
                                  }
                                  assert(idx < total_leaves());
                                  for (auto el : get_leaf(idx)) {
                                    if (f(el)) {
                                      if constexpr (!no_early_exit) {
                                        break;
                                      }
                                    }
                                  }
                                }

    );
  }

  // do the last leaf

  for (auto el : get_leaf(leaf_number_end)) {
    if (el >= end_key) {
      break;
    }
    if (f(el)) {
      if constexpr (!no_early_exit) {
        break;
      }
    }
  }
}

template <typename traits>
typename CPMA<traits>::key_type CPMA<traits>::max() const {
  return get_leaf(total_leaves() - 1).last();
}
template <typename traits> uint32_t CPMA<traits>::num_nodes() const {
  return (max() >> 32) + 1;
}

template <typename traits>
[[nodiscard]] uint64_t CPMA<traits>::head_array_size() const {
  if constexpr (head_form == InPlace) {
    return get_size();
  } else {
    return num_heads() * sizeof(key_type);
  }
}

template <typename traits>
std::unique_ptr<uint64_t, free_delete>
CPMA<traits>::getDegreeVector(typename leaf::iterator *hints) const {
  std::unique_ptr<uint64_t, free_delete> degrees =
      std::unique_ptr<uint64_t, free_delete>(
          (uint64_t *)malloc(sizeof(uint64_t) * (num_nodes())));

  ParallelTools::parallel_for(0, num_nodes(), [&](size_t i) {
    serial_map_with_hint<true>(
        [&]([[maybe_unused]] uint64_t el) {
          degrees.get()[i] += 1;
          return false;
        },
        (i + 1) << 32UL, hints[i]);
  });
  return degrees;
}

template <typename traits>
std::unique_ptr<uint64_t, free_delete>
CPMA<traits>::getApproximateDegreeVector(typename leaf::iterator *hints) const {

  std::unique_ptr<uint64_t, free_delete> degrees =
      std::unique_ptr<uint64_t, free_delete>(
          (uint64_t *)malloc(sizeof(uint64_t) * (num_nodes())));

  double average_element_size = ((double)N()) / get_element_count();

  ParallelTools::parallel_for(0, num_nodes(), [&](size_t i) {
    uintptr_t ptr_length = (uint8_t *)hints[i + 1].get_pointer() -
                           (uint8_t *)hints[i].get_pointer();
    degrees.get()[i] = (double)ptr_length / average_element_size;
  });

  return degrees;
}

#endif
