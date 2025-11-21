#ifndef CPMA_HPP
#define CPMA_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ranges>
#include <tuple>
#include <utility>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <concepts>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

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
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-Wshorten-64-to-32"

#if !defined(NO_TLX)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
#pragma clang diagnostic ignored "-Wpadded"
#include "tlx/container/btree_map.hpp"
#pragma clang diagnostic pop
#endif

#if VQSORT == 1
#include <hwy/contrib/sort/vqsort.h>
#endif

#include "ParallelTools/concurrent_hash_map.hpp"
#include "ParallelTools/flat_hash_map.hpp"
#include "ParallelTools/parallel.h"
#include "ParallelTools/reducer.h"
#include "ParallelTools/sort.hpp"

#include "StructOfArrays/multipointer.hpp"
#include "StructOfArrays/soa.hpp"

#include "parlay/internal/group_by.h"
#include "parlay/primitives.h"
#include "parlay/slice.h"
#pragma clang diagnostic pop

#include "internal/helpers.hpp"
#include "internal/leaf.hpp"
#include "internal/timers.hpp"

enum HeadForm { InPlace, Linear, Eytzinger, BNary };
// BNAry has B pointers, and B-1 elements in each block

template <typename T, typename U> struct overwrite_on_insert {
  constexpr void operator()(T current_value, U new_value) const {
    current_value = new_value;
  }
};

class make_pcsr {};

template <typename l, HeadForm h, uint64_t b = 0, bool density = false,
          bool rank = false, bool fixed_size_ = false,
          uint64_t max_fixed_size_ = 4096, bool parallel_ = true,
          bool maintain_offsets_ = false,
          typename value_update_ = overwrite_on_insert<
              typename l::element_ref_type, typename l::element_type>>
class PMA_traits {
public:
  using leaf = l;
  using key_type = typename leaf::key_type;
  static constexpr bool compressed = leaf::compressed;
  static constexpr HeadForm head_form = h;
  static constexpr uint64_t B_size = b;
  static constexpr bool store_density = density;
  static constexpr bool support_rank = rank;

  static constexpr bool fixed_size = fixed_size_;
  static constexpr uint64_t max_fixed_size = max_fixed_size_;

  static constexpr bool binary = leaf::binary;
  using element_type = typename leaf::element_type;
  using element_ref_type = typename leaf::element_ref_type;
  using element_ptr_type = typename leaf::element_ptr_type;
  using SOA_type = typename leaf::SOA_type;
  using value_type = typename leaf::value_type;

  using value_update = value_update_;
  static_assert(
      std::is_invocable_v<value_update, element_ref_type, element_type &>,
      "the value update function must take in a reference to the "
      "current value and the new value by reference");
#if defined(PMA_GROWING_FACTOR)
  static constexpr double growing_factor = PMA_GROWING_FACTOR;
#else
  static constexpr double growing_factor = 1.2; // 1.5;
#endif

#if PARALLEL == 1
  static constexpr bool parallel = parallel_;
#elif DEBUG == 1
  // make it run the parallel like paths when running in debug mode, even if it
  // is run without cilk
  static constexpr bool parallel = parallel_;
#else
  static constexpr bool parallel = false;
#endif

  static constexpr int leaf_blow_up_factor = (sizeof(key_type) == 3) ? 18 : 16;
  static constexpr uint64_t min_leaf_size = 64;
  static constexpr bool maintain_offsets = maintain_offsets_;
};
template <typename T = uint64_t>
using pma_settings = PMA_traits<uncompressed_leaf<T>, InPlace, 0, false, false>;
template <typename T = uint64_t>
using spmal_settings =
    PMA_traits<uncompressed_leaf<T>, Linear, 0, false, false>;
template <typename T = uint64_t>
using spmae_settings =
    PMA_traits<uncompressed_leaf<T>, Eytzinger, 0, false, false>;
template <typename T = uint64_t>
using spmab5_settings =
    PMA_traits<uncompressed_leaf<T>, BNary, 5, false, false>;
template <typename T = uint64_t>
using spmab9_settings =
    PMA_traits<uncompressed_leaf<T>, BNary, 9, false, false>;
template <typename T = uint64_t>
using spmab17_settings =
    PMA_traits<uncompressed_leaf<T>, BNary, 17, false, false>;
template <typename T = uint64_t> using spmab_settings = spmab17_settings<T>;

template <typename T = uint64_t>
using cpma_settings =
    PMA_traits<delta_compressed_leaf<T>, InPlace, 0, false, false>;
template <typename T = uint64_t>
using scpmal_settings =
    PMA_traits<delta_compressed_leaf<T>, Linear, 0, false, false>;
template <typename T = uint64_t>
using scpmae_settings =
    PMA_traits<delta_compressed_leaf<T>, Eytzinger, 0, false, false>;
template <typename T = uint64_t>
using scpmab5_settings =
    PMA_traits<delta_compressed_leaf<T>, BNary, 5, false, false>;
template <typename T = uint64_t>
using scpmab9_settings =
    PMA_traits<delta_compressed_leaf<T>, BNary, 9, false, false>;
template <typename T = uint64_t>
using scpmab17_settings =
    PMA_traits<delta_compressed_leaf<T>, BNary, 17, false, false>;
template <typename T = uint64_t> using scpmab_settings = scpmab17_settings<T>;

class empty_type {};

namespace PMA_precalculate {
template <typename traits>
[[nodiscard]] static constexpr uint64_t
calculate_num_leaves_rounded_up(uint64_t total_leaves) {
  static_assert(traits::head_form == Eytzinger || traits::head_form == BNary,
                "you should only be rounding the head array size of you are "
                "in either Eytzinger or BNary form");
  // Eytzinger and Bnary sometimes need to know the rounded number of leaves
  // linear order
  // make next power of 2
  if constexpr (traits::head_form == Eytzinger) {
    if (nextPowerOf2(total_leaves) > total_leaves) {
      return (nextPowerOf2(total_leaves) - 1);
    }
    return ((total_leaves * 2) - 1);
  }
  // BNary order
  if constexpr (traits::head_form == BNary) {
    uint64_t size = traits::B_size;
    while (size <= total_leaves) {
      size *= traits::B_size;
    }
    return size;
  }
}
template <typename traits>
static constexpr uint64_t num_heads(uint64_t num_total_leaves) {
  if constexpr (traits::head_form == InPlace) {
    return 0;
  }
  // linear order
  if constexpr (traits::head_form == Linear) {
    return num_total_leaves;
  }
  // make next power of 2
  if constexpr (traits::head_form == Eytzinger) {
    if (nextPowerOf2(num_total_leaves) > num_total_leaves) {
      uint64_t space =
          ((nextPowerOf2(num_total_leaves) - 1) + num_total_leaves + 1) / 2;
      return space;
    }
    return ((num_total_leaves * 2) - 1);
  }
  // BNary order
  if constexpr (traits::head_form == BNary) {
    uint64_t size = traits::B_size;
    while (size <= num_total_leaves) {
      size *= traits::B_size;
    }
    uint64_t check_size =
        ((size / traits::B_size + num_total_leaves + traits::B_size) /
         traits::B_size) *
        traits::B_size;

    return std::min(size, check_size);
  }
}

template <typename traits>
[[nodiscard]] static constexpr uint64_t
head_array_size(uint64_t num_total_leaves) {
  return traits::SOA_type::get_size_static(num_heads<traits>(num_total_leaves));
}

template <typename traits>
[[nodiscard]] static constexpr uint64_t data_array_size(uint64_t N) {
  uint64_t allocated_size = N;
  if constexpr (!traits::binary) {
    // we will place the value of the key zero here
    allocated_size += sizeof(typename traits::key_type);
  }
  if (allocated_size % 32 != 0) {
    allocated_size += 32 - (allocated_size % 32);
  }
  return traits::SOA_type::get_size_static(allocated_size /
                                           sizeof(typename traits::key_type)) +
         32;
}

template <typename traits>
[[nodiscard]] static constexpr uint64_t
density_array_size(uint64_t num_total_leaves) {
  if constexpr (!traits::store_density) {
    return 0;
  } else {
    return num_total_leaves * sizeof(uint16_t);
  }
}

template <typename traits>
[[nodiscard]] static constexpr uint64_t
rank_tree_array_size(uint64_t num_total_leaves) {
  if constexpr (!traits::support_rank) {
    return 0;
  } else {
    return nextPowerOf2(num_total_leaves) * sizeof(uint64_t);
  }
}

template <typename traits>
[[nodiscard]] static constexpr uint64_t
underlying_array_size(uint64_t N, uint64_t num_total_leaves) {
  uint64_t total_size = 0;
  if constexpr (traits::head_form != InPlace) {
    total_size += head_array_size<traits>(num_total_leaves);
    if (total_size % 32 != 0) {
      total_size += 32 - (total_size % 32);
    }
  }
  total_size += data_array_size<traits>(N);

  if constexpr (traits::store_density) {
    if (total_size % 32 != 0) {
      total_size += 32 - (total_size % 32);
    }
    total_size += density_array_size<traits>(num_total_leaves);
  }
  if constexpr (traits::support_rank) {
    if (total_size % 32 != 0) {
      total_size += 32 - (total_size % 32);
    }
    total_size += rank_tree_array_size<traits>(num_total_leaves);
  }

  if (total_size % 32 != 0) {
    total_size += 32 - (total_size % 32);
  }
  return total_size;
}

template <typename traits>
[[nodiscard]] static constexpr uint64_t
get_data_array_offset(uint64_t num_total_leaves) {
  if constexpr (traits::head_form == InPlace) {
    return 0;
  } else {
    uint64_t offset = head_array_size<traits>(num_total_leaves);
    if (offset % 32 != 0) {
      offset += 32 - (offset % 32);
    }
    return offset;
  }
}

template <typename traits>
[[nodiscard]] static constexpr uint64_t
get_density_array_offset(uint64_t N, uint64_t num_total_leaves) {
  if constexpr (!traits::store_density) {
    return 0;
  } else {

    uint64_t offset = get_data_array_offset<traits>(num_total_leaves) +
                      data_array_size<traits>(N);
    if (offset % 32 != 0) {
      offset += 32 - (offset % 32);
    }
    return offset;
  }
}

template <typename traits>
[[nodiscard]] static constexpr uint64_t
get_rank_tree_array_offset(uint64_t N, uint64_t num_total_leaves) {
  if constexpr (!traits::support_rank) {
    return 0;
  } else {
    uint64_t offset = get_data_array_offset<traits>(num_total_leaves) +
                      data_array_size<traits>(N);
    if (offset % 32 != 0) {
      offset += 32 - (offset % 32);
    }
    if constexpr (traits::store_density) {
      offset += density_array_size<traits>(num_total_leaves);
      if (offset % 32 != 0) {
        offset += 32 - (offset % 32);
      }
    }
    return offset;
  }
}

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
  uint64_t num_heads;
  uint64_t head_array_size;
  uint64_t data_array_size;
  uint64_t underlying_array_size;
  uint64_t density_array_size;
  uint64_t rank_tree_array_size;
  uint64_t data_array_offset;
  uint64_t density_array_offset;
  uint64_t rank_tree_array_offset;

  void print() const {
    printf("N = %lu, logN = %lu, loglogN = %lu, total_leaves = %lu, H = %lu, "
           "num_heads = %lu, head_array_size = %lu, data_array_size = %lu, "
           "density_array_size = %lu, rank_tree_array_size = %lu, "
           "underlying_array_size = %lu, data_array_offset = %lu, "
           "density_array_offset = %lu, rank_tree_array_offset = %lu\n",
           n, logn, loglogn, total_leaves, H, num_heads, head_array_size,
           data_array_size, density_array_size, rank_tree_array_size,
           underlying_array_size, data_array_offset, density_array_offset,
           rank_tree_array_offset);
  }
} __attribute__((aligned(128)));

template <typename traits>
static constexpr std::array<meta_data_t, 256> get_metadata_table() {

  std::array<meta_data_t, 256> res = {{}};
  uint64_t n = traits::min_leaf_size;
  uint64_t lln = bsr_long(bsr_long(n));
  uint64_t ln = traits::leaf_blow_up_factor * (1UL << lln);
  if (n < ln) {
    n = ln;
    lln = bsr_long(bsr_long(n));
  }

  while (n % ln != 0) {
    n += ln - (n % ln);
    lln = bsr_long(bsr_long(n));
    ln = traits::leaf_blow_up_factor * (1UL << lln);
  }
  uint64_t total_leaves = n / ln;
  uint64_t total_leaves_rounded_up = 0;
  if constexpr (traits::head_form == Eytzinger || traits::head_form == BNary) {
    total_leaves_rounded_up =
        PMA_precalculate::calculate_num_leaves_rounded_up<traits>(total_leaves);
  }

  uint64_t H = bsr_long(total_leaves);

  uint64_t i = 0;
  res[i] = {
      .n = n,
      .logn = ln,
      .loglogn = lln,
      .total_leaves = total_leaves,
      .H = H,
      .total_leaves_rounded_up = total_leaves_rounded_up,
      .elts_per_leaf = ln / sizeof(typename traits::key_type),
      .num_heads = num_heads<traits>(total_leaves),
      .head_array_size = (traits::head_form == InPlace)
                             ? 0
                             : head_array_size<traits>(total_leaves),
      .data_array_size = data_array_size<traits>(n),
      .underlying_array_size = underlying_array_size<traits>(n, total_leaves),
      .density_array_size = (traits::store_density)
                                ? density_array_size<traits>(total_leaves)
                                : 0,
      .rank_tree_array_size = (traits::support_rank)
                                  ? rank_tree_array_size<traits>(total_leaves)
                                  : 0,
      .data_array_offset = get_data_array_offset<traits>(total_leaves),
      .density_array_offset = get_density_array_offset<traits>(n, total_leaves),
      .rank_tree_array_offset =
          get_rank_tree_array_offset<traits>(n, total_leaves)};
  i += 1;
  for (; i < 256; i++) {
    uint64_t min_new_size = n + ln;
    uint64_t desired_new_size = std::numeric_limits<uint64_t>::max();
    // max sure that it fits in the amount of bits
    if (n < desired_new_size / traits::growing_factor) {
      desired_new_size = n * traits::growing_factor;
    }
    if (desired_new_size < min_new_size) {
      n = min_new_size;
    } else {
      n = desired_new_size;
    }
    lln = bsr_long(bsr_long(n));
    ln = traits::leaf_blow_up_factor * (1UL << lln);
    while (n % ln != 0) {
      n += ln - (n % ln);
      if (n == 0) {
        lln = 0;
      } else {
        lln = bsr_long(bsr_long(n));
      }

      ln = traits::leaf_blow_up_factor * (1U << lln);
    }
    total_leaves = n / ln;
    H = bsr_long(total_leaves);
    if constexpr (traits::head_form == Eytzinger ||
                  traits::head_form == BNary) {
      total_leaves_rounded_up =
          PMA_precalculate::calculate_num_leaves_rounded_up<traits>(
              total_leaves);
    }
    res[i] = {
        .n = n,
        .logn = ln,
        .loglogn = lln,
        .total_leaves = total_leaves,
        .H = H,
        .total_leaves_rounded_up = total_leaves_rounded_up,
        .elts_per_leaf = ln / sizeof(typename traits::key_type),
        .num_heads = num_heads<traits>(total_leaves),
        .head_array_size = (traits::head_form == InPlace)
                               ? 0
                               : head_array_size<traits>(total_leaves),
        .data_array_size = data_array_size<traits>(n),
        .underlying_array_size = underlying_array_size<traits>(n, total_leaves),
        .density_array_size = (traits::store_density)
                                  ? density_array_size<traits>(total_leaves)
                                  : 0,
        .rank_tree_array_size = (traits::support_rank)
                                    ? rank_tree_array_size<traits>(total_leaves)
                                    : 0,
        .data_array_offset = get_data_array_offset<traits>(total_leaves),
        .density_array_offset =
            get_density_array_offset<traits>(n, total_leaves),
        .rank_tree_array_offset =
            get_rank_tree_array_offset<traits>(n, total_leaves)};
  }

  return res;
}

} // namespace PMA_precalculate

template <typename traits> class CPMA {
public:
  using leaf = typename traits::leaf;
  using key_type = typename traits::key_type;
  static constexpr uint64_t B_size = traits::B_size;
  static constexpr HeadForm head_form = traits::head_form;
  static constexpr bool store_density = traits::store_density;
  static constexpr bool support_rank = traits::support_rank;
  static constexpr bool fixed_size = traits::fixed_size;
  static constexpr uint64_t max_fixed_size = traits::max_fixed_size;
  static constexpr bool parallel = traits::parallel;

  static_assert(std::is_trivially_copyable_v<key_type>,
                "T must be trivially copyable");
  static_assert(B_size == 0 || head_form == BNary,
                "B_size should only be used if we are using head_form = BNary");

  static_assert(std::is_unsigned_v<key_type>,
                "we assume that in sorted order the null sentinel, which is "
                "zero, will be first");

  static constexpr bool binary = traits::binary;
  using element_type = typename traits::element_type;
  using element_ref_type = typename traits::element_ref_type;
  using element_ptr_type = typename traits::element_ptr_type;
  using SOA_type = typename traits::SOA_type;
  using value_type = typename traits::value_type;
  using value_update = typename traits::value_update;

  // types needs for graphs
  //  When we store a graph we assume that we are storing 64 bit element paris
  //  of src dest
  using node_t = uint32_t;
  using extra_data_t =
      std::pair<std::unique_ptr<typename leaf::iterator, free_delete>,
                std::unique_ptr<uint64_t, free_delete>>;

  // bool in binary/
  // the value if there is only 1 value
  // tuple of the values if there are multiple
  static constexpr auto get_value_type() {
    if constexpr (binary) {
      return (bool)true;
    } else {
      value_type v;
      if constexpr (std::tuple_size_v<value_type> == 1) {
        return std::tuple_element_t<0, value_type>();
      } else {
        return v;
      }
    }
  }
  using weight_t = decltype(get_value_type());

private:
  static constexpr double growing_factor = traits::growing_factor;

  static constexpr int leaf_blow_up_factor = traits::leaf_blow_up_factor;
  static constexpr uint64_t min_leaf_size = traits::min_leaf_size;
  static_assert(min_leaf_size >= 64, "min_leaf_size must be at least 64 bytes");

  using meta_data_t = PMA_precalculate::meta_data_t;

  static constexpr std::array<meta_data_t, 256> meta_data =
      PMA_precalculate::get_metadata_table<traits>();

  [[nodiscard]] static constexpr uint64_t
  total_leaves(uint8_t meta_data_index) {
    return meta_data[meta_data_index].total_leaves;
  }

public:
  [[nodiscard]] uint64_t total_leaves() const {
    return meta_data[meta_data_index].total_leaves;
  }

private:
  [[nodiscard]] uint64_t num_heads() const {
    return meta_data[meta_data_index].num_heads;
  }

  [[nodiscard]] static constexpr uint64_t N(uint8_t meta_data_index) {
    return meta_data[meta_data_index].n;
  }

public:
  [[nodiscard]] uint64_t N() const { return N(meta_data_index); }

private:
  [[nodiscard]] uint64_t head_array_size() const {
    return meta_data[meta_data_index].head_array_size;
  }

  static constexpr uint64_t get_max_fixed_size() {
    uint8_t meta_data_index = 0;
    while (meta_data[meta_data_index + 1].underlying_array_size <=
           max_fixed_size) {
      meta_data_index++;
    }
    return meta_data_index;
  }

  [[nodiscard]] constexpr uint64_t underlying_array_size() const {
    return meta_data[meta_data_index].underlying_array_size;
  }

  typename std::conditional<fixed_size,
                            std::array<uint8_t, meta_data[get_max_fixed_size()]
                                                    .underlying_array_size>,
                            void *>::type underlying_array;

  key_type count_elements_ = 0;

  uint8_t meta_data_index = 0;

  bool has_0 = false;

  // when we are running in pcsr like mode, this is to store the offsets to the
  // different node regions
  // we store the following
  // - a pointer an an array of pointers to the start of each region
  // - - there is an extra one at the end to point at the overall end
  // - the number of nodes (how long the arrays are)
  // - the degree of each node
  [[no_unique_address]]
  typename std::conditional<traits::maintain_offsets, pcsr_node_info<key_type>,
                            empty_type>::type offsets_array;

  static constexpr std::array<std::array<float, sizeof(key_type) * 8>, 256>
  get_upper_density_bound_table() {
    uint64_t max_fixed_size_index = get_max_fixed_size();
    std::array<std::array<float, sizeof(key_type) * 8>, 256> res;
    for (uint64_t i = 0; i < 256; i++) {
      auto m = meta_data[i];
      for (uint64_t j = 0; j < sizeof(key_type) * 8; j++) {
        float val = 1.0F / 2.0F;

        if (m.H != 0) {
          val = 1.0F / 2.0F + (((1.0F / 2.0F) * j) / m.H);
        }
        if (val >= static_cast<float>(m.logn - (3 * leaf::max_element_size)) /
                       m.logn) {
          val = static_cast<float>(m.logn - (3 * leaf::max_element_size)) /
                    m.logn -
                .001F;
        }
        if constexpr (fixed_size) {
          if (j == 0 && (i + 1) > max_fixed_size_index) {
            val = density_limit(meta_data[i].logn) - .001;
          }
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
        float val = std::max(((float)sizeof(key_type)) / m.logn, 1.0F / 4.0F);
        if (m.H != 0) {
          val = std::max(((float)sizeof(key_type)) / m.logn,
                         1.0F / 4.0F - ((.125F * j) / m.H));
        }
        if (val <= 0) {
          val = std::numeric_limits<float>::min();
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

  [[nodiscard]] uint8_t *underlying_array_char() const {
    if constexpr (fixed_size) {
      return (uint8_t *)(underlying_array.data());
    } else {
      return static_cast<uint8_t *>(underlying_array);
    }
  }

  key_type *head_array() const {
    static_assert(head_form != InPlace);
    return reinterpret_cast<key_type *>(underlying_array_char());
  }

  key_type *key_array() const {
    auto ptr = reinterpret_cast<key_type *>(
        underlying_array_char() + meta_data[meta_data_index].data_array_offset);
    if constexpr (!binary) {
      // +1 to offset past the value for zero
      ptr += 1;
    }
    return ptr;
  }

  key_type *data_array() const {
    return reinterpret_cast<key_type *>(
        underlying_array_char() + meta_data[meta_data_index].data_array_offset);
  }

  // stored the density of each leaf to speed up merges and get_density_count
  // only use a uint16_t to save on space
  // this might not be enough to store out of place leaves after batch merge,
  // so in the event of an overflow just store max which we then just go count
  // the density as usual, which is cheap since it is stored out of place and
  // the size is just written
  auto density_array() const {
    if constexpr (!store_density) {
      return empty_type();
    } else {
      return reinterpret_cast<uint16_t *>(
          underlying_array_char() +
          meta_data[meta_data_index].density_array_offset);
    }
  }

  // stored a flattened rank tree where each node stored the number of
  // elements to the left of that node in the subtree of that node
  auto rank_tree_array() const {
    if constexpr (!support_rank) {
      return empty_type();
    } else {
      return reinterpret_cast<uint64_t *>(
          underlying_array_char() +
          meta_data[meta_data_index].rank_tree_array_offset);
    }
  }

  [[nodiscard]] uint64_t soa_num_spots() const {
    if constexpr (!binary) {
      // extra spot for the value of 0
      return N() / sizeof(key_type) + 1;
    } else {
      return N() / sizeof(key_type);
    }
  }

  template <size_t... Is> auto get_data_ref(size_t i) const {
    if constexpr (!binary) {
      // +1 to offset past the value for zero
      i += 1;
    }
    if constexpr (sizeof...(Is) == 1) {
      return std::get<0>(SOA_type::template get_static<Is...>(
          data_array(), soa_num_spots(), i));
    } else {
      return SOA_type::template get_static<Is...>(data_array(), soa_num_spots(),
                                                  i);
    }
  }
  template <size_t... Is> auto get_zero_el_ref() const {
    static_assert(!binary);
    return get_data_ref<Is...>(-1);
  }

  template <size_t... Is> auto get_data_ptr(size_t i) const {
    if constexpr (!binary) {
      // +1 to offset past the value for zero
      i += 1;
    }
    return SOA_type::template get_static_ptr<Is...>(data_array(),
                                                    soa_num_spots(), i);
  }

  template <size_t... Is> auto get_zero_el_ptr() const {
    static_assert(!binary);
    return get_data_ptr<Is...>(-1);
  }

  template <size_t... Is> auto get_head_ref(size_t i) const {
    if constexpr (sizeof...(Is) == 1) {
      return std::get<0>(
          SOA_type::template get_static<Is...>(head_array(), num_heads(), i));
    } else {
      return SOA_type::template get_static<Is...>(head_array(), num_heads(), i);
    }
  }
  template <size_t... Is> auto get_head_ptr(size_t i) const {
    return SOA_type::template get_static_ptr<Is...>(head_array(), num_heads(),
                                                    i);
  }

#if VQSORT == 1
  hwy::Sorter sorter;
#endif

  uint64_t build_rank_array_recursive(uint64_t start, uint64_t end,
                                      uint64_t rank_array_index,
                                      uint64_t *correct_array) const;

  [[nodiscard]] bool check_rank_array() const;

  void update_rank(uint64_t leaf_changed, int64_t change_amount);
  void update_rank(ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>>
                       &rank_additions);
  void update_rank(std::vector<std::pair<uint64_t, uint64_t>> &rank_additions);

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

  [[nodiscard]] std::pair<float, float> density_bound(uint64_t depth) const;

  [[nodiscard]] uint64_t loglogN() const {
    return meta_data[meta_data_index].loglogn;
  }
  [[nodiscard]] uint64_t logN() const {
    return meta_data[meta_data_index].logn;
  }
  [[nodiscard]] uint64_t H() const { return meta_data[meta_data_index].H; }

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

  [[nodiscard]] static constexpr double density_limit(uint64_t logN) {
    // we need enough space on both sides regardless of how elements are split
    if constexpr (compressed) {
      return static_cast<double>(logN - (3 * leaf::max_element_size)) / logN;
    } else {
      return static_cast<double>(logN - (1 * leaf::max_element_size)) / logN;
    }
  }

  [[nodiscard]] double density_limit() const { return density_limit(logN()); }

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
  [[nodiscard]] std::pair<
      ParallelTools::Reducer_Vector<std::tuple<uint64_t, uint64_t>>,
      std::optional<uint64_t>>
  get_ranges_to_redistibute(
      const ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>>
          &leaves_to_check,
      uint64_t num_elts_merged, F bounds_check) const;

  template <class F>
  [[nodiscard]] std::pair<std::vector<std::tuple<uint64_t, uint64_t>>,
                          std::optional<uint64_t>>
  get_ranges_to_redistibute(
      const std::vector<std::pair<uint64_t, uint64_t>> &leaves_to_check,
      uint64_t num_elts_merged, F bounds_check) const;

  void redistribute_ranges(
      const std::vector<std::tuple<uint64_t, uint64_t>> &ranges);

  void redistribute_ranges(
      const ParallelTools::Reducer_Vector<std::tuple<uint64_t, uint64_t>>
          &ranges);

  using leaf_bound_t = struct {
    key_type start_elt;
    key_type end_elt;
    uint64_t start_leaf_index;
    uint64_t end_leaf_index;
  };
  std::vector<leaf_bound_t> get_leaf_bounds(uint64_t split_points) const;

  [[nodiscard]] uint64_t get_ranges_to_redistibute_lookup_sibling_count(
      const std::vector<ParallelTools::concurrent_hash_map<uint64_t, uint64_t>>
          &ranges_check,
      uint64_t start, uint64_t length, uint64_t level,
      uint64_t depth = 0) const;
  [[nodiscard]] uint64_t get_ranges_to_redistibute_lookup_sibling_count(
      const std::vector<ska::flat_hash_map<uint64_t, uint64_t>> &ranges_check,
      uint64_t start, uint64_t length, uint64_t level) const;

  [[nodiscard]] std::pair<std::vector<std::tuple<uint64_t, uint64_t>>,
                          std::optional<uint64_t>>
  get_ranges_to_redistibute_debug(
      const std::vector<std::pair<uint64_t, uint64_t>> &leaves_to_check,
      uint64_t num_elts_merged) const;

  [[nodiscard]] std::pair<std::vector<std::tuple<uint64_t, uint64_t>>,
                          std::optional<uint64_t>>
  get_ranges_to_redistibute_debug(
      const ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>>
          &leaves_to_check,
      uint64_t num_elts_merged) const {
    return get_ranges_to_redistibute_debug(leaves_to_check.get(),
                                           num_elts_merged);
  }

  [[nodiscard]] std::map<uint64_t, std::pair<uint64_t, uint64_t>>
  get_ranges_to_redistibute_internal(
      const std::pair<uint64_t, uint64_t> *begin,
      const std::pair<uint64_t, uint64_t> *end) const;

  [[nodiscard]] uint64_t sum_parallel() const;
  [[nodiscard]] uint64_t sum_parallel2(uint64_t start, uint64_t end,
                                       int depth) const;
  void print_array_region(uint64_t start_leaf, uint64_t end_leaf) const;

  void insert_post_place(uint64_t leaf_number, uint64_t byte_count);
  void remove_post_place(uint64_t leaf_number, uint64_t byte_count);

  uint64_t rank_given_leaf(uint64_t leaf_number, key_type e) const;

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
      if constexpr (!parallel) {
        if constexpr (!std::is_integral_v<key_type>) {
          std::sort(batch, batch + batch_size);
          return;
        } else {
#if PARALLEL == 0
          // if PARALLEL == 0 then we can call the faster integer sort, but
          // since it uses parallelism internally which is hard to turn off
          // don't call it when parallelism is on a compile time, but templated
          // out
          if (batch_size > 100000) {
            std::vector<key_type> data_vector;
            wrapArrayInVector(batch, batch_size, data_vector);
            parlay::integer_sort_inplace(data_vector);
            releaseVectorWrapper(data_vector);
            return;
          }
#endif
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
      } else {
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
      }
    }
  }

public:
  // TODO(wheatman) make private
  bool check_nothing_full();
  static constexpr bool compressed = leaf::compressed;
  explicit CPMA();
  // Move constructor
  CPMA(CPMA &&other) noexcept
      : meta_data_index(other.meta_data_index), has_0(other.has_0),
        count_elements_(other.count_elements_),
        underlying_array(other.underlying_array),
        offsets_array(other.offsets_array) {
    other.count_elements_ = 0;
    other.meta_data_index = 0;
    other.has_0 = false;
    if constexpr (!fixed_size) {
      other.underlying_array = nullptr;
    }
#if VQSORT == 1
    sorter = other.sorter;
#endif
    if constexpr (traits::maintain_offsets) {
      other.offsets_array = pcsr_node_info<key_type>();
    }
  }
  // Copy constructor
  CPMA(const CPMA &other) noexcept
      : meta_data_index(other.meta_data_index), has_0(other.has_0),
        count_elements_(other.count_elements_) {
    if constexpr (fixed_size) {
      underlying_array = other.underlying_array;
    } else {
      uint64_t allocated_size = other.underlying_array_size() + 32;
      if (allocated_size % 32 != 0) {
        allocated_size += 32 - (allocated_size % 32);
      }
      underlying_array = aligned_alloc(32, allocated_size);
      ParallelTools::parallel_for(
          0, other.underlying_array_size(),
          [&](size_t i) { underlying_array[i] = other.underlying_array[i]; });
    }
#if VQSORT == 1
    sorter = other.sorter;
#endif
    if constexpr (traits::maintain_offsets) {
      offsets_array.size = other.offsets_array.size;
      offsets_array.locations =
          (decltype(offsets_array.locations))malloc(offsets_array.size * sizeof(*(offsets_array.locations)));
      offsets_array.degrees =
          ((decltype(offsets_array.degrees))malloc(offsets_array.size * sizeof(*(offsets_array.degrees)));
      ParallelTools::parallel_for(0, offsets_array.size, [&](size_t i) {
        offsets_array.locations[i] =
            underlying_array_char() +
            ((uint8_t*)other.offsets_array.locations[i] - other.underlying_array_char());
        offsets_array.degrees[i] = other.offsets_array.degrees[i];
      });
    }
  }
  //   // Move assignment
  CPMA &operator=(CPMA &&other) noexcept {
    if (this == &other) {
      return *this;
    }
    count_elements_ = other.count_elements_;
    other.count_elements_ = 0;
    meta_data_index = other.meta_data_index;
    other.meta_data_index = 0;
    has_0 = other.has_0;
    other.has_0 = false;
    free(underlying_array);
    underlying_array = other.underlying_array;
    if constexpr (!fixed_size) {
      other.underlying_array = nullptr;
    }
#if VQSORT == 1
    sorter = other.sorter;
#endif
    if constexpr (traits::maintain_offsets) {
      free(offsets_array.locations);
      free(offsets_array.degrees);
      offsets_array = other.offsets_array;
      other.offsets_array = pcsr_node_info<key_type>();
    }
    return *this;
  }
  // copy assignment
  CPMA &operator=(const CPMA &other) noexcept {
    if (this == &other) {
      return *this;
    }
    count_elements_ = other.count_elements_;
    meta_data_index = other.meta_data_index;
    has_0 = other.has_0;
    if constexpr (fixed_size) {
      underlying_array = other.underlying_array;
    } else {
      uint64_t allocated_size = other.underlying_array_size() + 32;
      if (allocated_size % 32 != 0) {
        allocated_size += 32 - (allocated_size % 32);
      }
      underlying_array = aligned_alloc(32, allocated_size);
      ParallelTools::parallel_for(
          0, other.underlying_array_size(), [&](size_t i) {
            underlying_array_char()[i] = other.underlying_array_char()[i];
          });
    }
#if VQSORT == 1
    sorter = other.sorter;
#endif
    if constexpr (traits::maintain_offsets) {
      offsets_array.size = other.offsets_array.size;
      offsets_array.locations =
          malloc(offsets_array.size * sizeof(*(offsets_array.locations)));
      offsets_array.degrees =
          malloc(offsets_array.size * sizeof(*(offsets_array.degrees)));
      ParallelTools::parallel_for(0, offsets_array.size, [&](size_t i) {
        offsets_array.locations[i] =
            underlying_array_char() +
            (other.offsets_array.locations[i] - other.underlying_array_char());
        offsets_array.degrees[i] = other.offsets_array.degrees[i];
      });
    }
    return *this;
  }
  CPMA(key_type *start, key_type *end);
  CPMA(auto &range);
  ~CPMA() {
    if (underlying_array != nullptr) {
      if constexpr (!fixed_size) {
        if constexpr (!binary) {
          if constexpr (!std::is_trivial_v<element_type>) {

            if constexpr (head_form != InPlace) {
              ParallelTools::parallel_for(0, num_heads(), [&](size_t i) {
                if (get_head_ref<0>(i) != 0) {
                  get_head_ptr(i).deconstruct();
                }
              });
            }
            if (has_0) {
              get_zero_el_ptr().deconstruct();
            }
            ParallelTools::parallel_for(0, soa_num_spots() - 1, [&](size_t i) {
              if (get_data_ref<0>(i) != 0) {
                get_data_ptr(i).deconstruct();
              }
            });
          }
        }
        free(underlying_array);
      }
    }
    if constexpr (traits::maintain_offsets) {
      free(offsets_array.locations);
      free(offsets_array.degrees);
    }
  }
  void print_pma() const;
  void print_array() const;
  bool has(key_type e) const;
  value_type value(key_type e) const;
  std::pair<bool, uint64_t> has_and_rank(key_type e) const;
  bool exists(key_type e) const { return has(e); }
  bool insert(element_type e);
  std::pair<bool, uint64_t> insert_get_rank(element_type e);
  bool insert_by_rank(element_type e, uint64_t rank);
  bool update_by_rank(element_type e, uint64_t rank);
  uint64_t rank(key_type e);
  typename traits::element_type select(uint64_t rank) const;
  uint64_t insert_batch(element_ptr_type e, uint64_t batch_size,
                        bool sorted = false);
  uint64_t remove_batch(key_type *e, uint64_t batch_size, bool sorted = false);
  // split num is the index of which partition you are
  template <class Vector_pairs>
  uint64_t insert_batch_internal(element_ptr_type e, uint64_t batch_size,
                                 Vector_pairs &leaves_to_check,
                                 uint64_t start_leaf_idx, uint64_t end_leaf_idx,
                                 Vector_pairs &rank_additions);
  template <class Vector_pairs>
  uint64_t insert_batch_internal_small_batch(element_ptr_type e,
                                             uint64_t batch_size,
                                             Vector_pairs &leaves_to_check,
                                             uint64_t start_leaf_idx,
                                             uint64_t end_leaf_idx,
                                             Vector_pairs &rank_additions);

  template <class Vector_pairs>
  uint64_t remove_batch_internal_small_batch(key_type *e, uint64_t batch_size,
                                             Vector_pairs &leaves_to_check,
                                             uint64_t start_leaf_idx,
                                             uint64_t end_leaf_idx,
                                             Vector_pairs &rank_additions);
  template <class Vector_pairs>
  uint64_t remove_batch_internal(key_type *e, uint64_t batch_size,
                                 Vector_pairs &leaves_to_check,
                                 uint64_t start_leaf_idx, uint64_t end_leaf_idx,
                                 Vector_pairs &rank_additions);

  bool remove(key_type e);
  bool remove_by_rank(key_type e, uint64_t rank);
  // return the amount of memory the structure uses

  [[nodiscard]] uint64_t get_size() const {
    uint64_t total_size = sizeof(CPMA);
    if constexpr (!fixed_size) {
      total_size += meta_data[meta_data_index].underlying_array_size;
    }
    if constexpr (traits::maintain_offsets) {
      total_size += offsets_array.size * sizeof(*(offsets_array.locations));
      total_size += offsets_array.size * sizeof(*(offsets_array.degrees));
    }
    return total_size;
  }

  [[nodiscard]] uint64_t get_element_count() const {
    return count_elements_ + has_0;
  }

  [[nodiscard]] uint64_t sum() const;
  [[nodiscard]] uint64_t sum_serial(int64_t start, int64_t end) const;
  [[nodiscard]] key_type max() const;
  [[nodiscard]] key_type min() const;
  [[nodiscard]] uint32_t num_nodes() const;
  [[nodiscard]] uint64_t size() const { return get_element_count() + has_0; }
  [[nodiscard]] uint64_t num_edges() const { return size(); }

  uint64_t get_degree(uint64_t i, const extra_data_t &extra_data) const {
    return extra_data.second.get()[i];
  }
  [[nodiscard]] uint64_t get_head_structure_size() const {
    return head_array_size();
  }

  template <bool no_early_exit, class F> bool map(F f) const;
  template <class F> bool parallel_map(F f) const;

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
  template <class T> extra_data_t getExtraData(T arg) const {
    // TODO(wheatman) handle if there is a self loop at zero
    bool skip = false;
    if constexpr (std::is_constructible_v<T, bool>) {
      skip = arg;
    }
    if (skip) {
      return {};
    }

    auto hints = (typename leaf::iterator *)malloc(
        sizeof(typename leaf::iterator) * (num_nodes() + 1));
    hints[0] = get_leaf(0).begin();
    ParallelTools::For<parallel>(1, num_nodes(), 1024, [&](uint64_t i_) {
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
    // passing in the data for the first head, but that will never be read so
    // it doesn't matter, just need something valid
    hints[num_nodes()] = typename leaf::iterator(index_to_head(0),
                                                 index_to_data(total_leaves()));
    return {std::unique_ptr<typename leaf::iterator, free_delete>(hints),
            getApproximateDegreeVector(hints)};
  }

  template <class F>
  void map_neighbors(uint64_t i, F f, const extra_data_t &extra_data,
                     bool run_parallel) const {
    // TODO(wheatman) handle if there is a self loop at zero
    auto hints = extra_data.first.get();
    assert(*(hints[i]) >= i << 32UL);
    if constexpr (parallel) {
      if (run_parallel) {
        serial_map_with_hint_par<F::no_early_exit>(
            [&](uint64_t el) { return f(el >> 32UL, el & 0xFFFFFFFFUL); },
            (i + 1) << 32U, hints[i], hints[i + 1]);
      } else {
        serial_map_with_hint<F::no_early_exit>(
            [&](uint64_t el) { return f(el >> 32UL, el & 0xFFFFFFFFUL); },
            (i + 1) << 32U, hints[i]);
      }
    } else {
      serial_map_with_hint<F::no_early_exit>(
          [&](uint64_t el) { return f(el >> 32UL, el & 0xFFFFFFFFUL); },
          (i + 1) << 32U, hints[i]);
    }
  }
  template <bool no_early_exit = true, class F>
  bool map_range(F f, key_type start_key, key_type end_key) const;

  template <class F>
  uint64_t map_range_length(F f, key_type start, uint64_t length) const;

  // used for the graph world
  template <class F>
  void map_range(F f, uint64_t start_node, uint64_t end_node,
                 [[maybe_unused]] const extra_data_t &extra_data) const {
    uint64_t start = start_node << 32UL;
    uint64_t end = end_node << 32UL;
    auto f2 = [&](uint64_t el) { return f(el >> 32UL, el & 0xFFFFFFFFUL); };
    map_range<true>(f2, start, end);
  }

  // moves half of the data to the uninitlized pma pointer at with right
  // returns the largest element which is left in the original pma
  key_type split(CPMA *right);

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
    if constexpr (!binary) {
      if (has_0) {
        // just set it up to read the special first element and then naturally
        // fall into the normal pattern without needing any other special
        // handling

        // this being zero means it knows to go to the next leaf, which is the
        // first real leaf, and since this is only for the zero element, the
        // key is always zero
        assert(get_zero_el_ref<0>() == 0);
        uint64_t next_leaf = 0;
        if (get_element_count() == 0) {
          next_leaf = 1;
        }
        return iterator(
            typename leaf::iterator(get_zero_el_ref(), get_zero_el_ptr()),
            next_leaf, *this);
      }
    } else {
      if (has_0) {
        // just set it up to read the special first element and then naturally
        // fall into the normal pattern without needing any other special
        // handling

        // this being zero means it knows to go to the next leaf, which is the
        // first real leaf, and since this is off the end, it is the extra
        // space which is kept at zero
        assert(get_data_ref<0>(soa_num_spots()) == 0);
        uint64_t next_leaf = 0;
        if (get_element_count() == 0) {
          next_leaf = 1;
        }
        return iterator(typename leaf::iterator(get_data_ref(soa_num_spots()),
                                                get_data_ptr(soa_num_spots())),
                        next_leaf, *this);
      }
    }
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
    if (key == 0) {
      return begin();
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
    uintptr_t bytes_from_start = ptr - ((uint8_t *)key_array());
    return bytes_from_start / logN();
  }
  // extra function to help with PCSR
  CPMA(make_pcsr tag, size_t num_nodes);
  CPMA(make_pcsr tag, size_t num_nodes, const auto &edges);

  bool insert_pcsr(key_type src, key_type dest, value_type val);
  bool insert_pcsr(key_type src, key_type dest) {
    static_assert(binary);
    return insert_pcsr(src, dest, {});
  }
  bool remove_pcsr(key_type src, key_type dest);
  bool contains_pcsr(key_type src, key_type dest) const;

  template <std::ranges::random_access_range R, class Vector_pairs>
  uint64_t insert_batch_internal_pcsr(
      R &es, std::invocable<typename std::ranges::range_value_t<R>> auto &&f,
      Vector_pairs &leaves_to_check, uint64_t start_leaf_idx,
      uint64_t end_leaf_idx);

  template <std::ranges::random_access_range R, class Vector_pairs>
  uint64_t remove_batch_internal_pcsr(
      R &es, std::invocable<typename std::ranges::range_value_t<R>> auto &&f,
      Vector_pairs &leaves_to_check, uint64_t start_leaf_idx,
      uint64_t end_leaf_idx);

  template <std::ranges::random_access_range R>
  uint64_t insert_batch_pcsr(
      R &es, std::invocable<typename std::ranges::range_value_t<R>> auto &&f,
      bool sorted = false);
  template <std::ranges::random_access_range R>
  uint64_t remove_batch_pcsr(
      R &es, std::invocable<typename std::ranges::range_value_t<R>> auto &&f,
      bool sorted = false);

  std::pair<uint64_t, bool> get_info_from_raw_pointer(void *ptr) const;

  template <bool no_early_exit, bool parallel, class F>
  void map_neighbors_pcsr(key_type node, F f) const;

  key_type degree_pcsr(key_type node) const {
    assert(offsets_array.size > 0);
    assert(offsets_array.degrees != nullptr);
    return offsets_array.degrees[node];
  }

  key_type num_nodes_pcsr() const {
    assert(offsets_array.size > 0);
    return offsets_array.size - 1;
  }

  static constexpr typename traits::key_type pcsr_top_bit =
      (std::numeric_limits<typename traits::key_type>::max() >> 1) + 1;
  static_assert(std::has_single_bit(static_cast<size_t>(pcsr_top_bit)));

  [[nodiscard]] bool verify_pcsr_nodes() const {

    for (key_type i = 0; i < num_nodes_pcsr(); i++) {
      void *location = offsets_array.locations[i];
      if (location == nullptr) {
        std::cout << "sentinal " << i << " is nullptr\n";
        return false;
      }
      key_type element = *((key_type *)location);
      if (element < pcsr_top_bit) {
        std::cout << "offset location " << i << " not pointing at a sentinal\n";
        std::cout << i << ", " << element << "\n";
        return false;
      }
      if ((element ^ pcsr_top_bit) != i) {
        std::cout << "offset location " << i
                  << " pointing to the wrong sentinal\n";
        return false;
      }
    }

    assert(offsets_array.locations[num_nodes_pcsr()] ==
           data_array() + soa_num_spots());
    return true;
  }

  [[nodiscard]] bool verify_pcsr_degrees() const {
    bool good = true;
    for (key_type i = 0; i < num_nodes_pcsr(); i++) {
      key_type stored_degree = degree_pcsr(i);
      key_type counted_degree = 0;
      map_neighbors_pcsr<true, false>(
          i, [&]([[maybe_unused]] const auto &arg1,
                 [[maybe_unused]] const auto &arg2) { counted_degree += 1; });
      if (stored_degree != counted_degree) {
        std::cout << "stored degree count is off for node " << i << "\n";
        std::cout << "stored count is " << stored_degree
                  << " counted degree is " << counted_degree << "\n";
        good = false;
      }
    }
    return good;
  }

  [[nodiscard]] size_t num_edges_pcsr() const {
    assert(offsets_array.size > 0);
    return get_element_count() - num_nodes_pcsr();
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
  timer double_timer("double_in_double");
  double_timer.start();
  timer merge_timer("merge_in_double");
  merge_timer.start();
  auto merged_data =
      leaf::template merge<head_form == InPlace, store_density, parallel>(
          get_data_ptr(0), total_leaves(), logN(), 0,
          [this](uint64_t index) -> element_ref_type {
            return index_to_head(index);
          },
          density_array());
  element_type zero_element;
  if constexpr (!binary) {
    tuple_set_and_zero(zero_element, get_zero_el_ref());
    // zero_element = get_zero_el_ref();
  }
  assert(((uint64_t)meta_data_index) + times <= 255);
  meta_data_index += times;
  if constexpr (fixed_size) {
    if (underlying_array_size() > max_fixed_size) {
      std::cerr << "trying to put too many elements into the fixes size PMA, "
                   "probably about to crash\n";
      std::cerr << "currently has " << size() << " elements "
                << " fixed to  a maximum of " << underlying_array_size()
                << "\n";
    }
  }
  merge_timer.stop();
  void *old_array = nullptr;
  if constexpr (!fixed_size) {
    old_array = underlying_array;
    // steal an extra few bytes to ensure we never read off the end
    uint64_t allocated_size = underlying_array_size() + 32;
    if (allocated_size % 32 != 0) {
      allocated_size += 32 - (allocated_size % 32);
    }
    underlying_array = aligned_alloc(32, allocated_size);
  }
  if constexpr (!binary) {
    get_zero_el_ptr().set_and_zero(zero_element);
  } else {
    std::get<0>(get_data_ref(soa_num_spots())) = 0;
  }
  if constexpr (head_form != InPlace) {
    ParallelTools::parallel_for(0, (head_array_size() / sizeof(key_type)),
                                [&](size_t i) { head_array()[i] = 0; });
  }

  if constexpr (support_rank) {
    std::fill(rank_tree_array(),
              rank_tree_array() + nextPowerOf2(total_leaves()), 0);
  }
  timer split_timer("split_in_double");
  split_timer.start();
  ParallelTools::par_do(
      [old_array]() { free(old_array); },
      [&]() {
        merged_data.leaf
            .template split<head_form == InPlace, store_density, support_rank,
                            parallel, traits::maintain_offsets>(
                total_leaves(), merged_data.size, logN(), get_data_ptr(0), 0,
                [this](uint64_t index) -> element_ref_type {
                  return index_to_head(index);
                },
                density_array(), rank_tree_array(), total_leaves(),
                offsets_array);
      });

  split_timer.stop();
  merged_data.free();
  if constexpr (traits::maintain_offsets) {
    offsets_array.locations[offsets_array.size - 1] =
        data_array() + soa_num_spots();
  }
#if DEBUG == 1
  for (uint64_t i = 0; i < N(); i += logN()) {
    ASSERT(get_density_count(i, logN()) <= logN() - leaf::max_element_size,
           "%lu > %lu\n tried to split %lu bytes into %lu leaves\n i = %lu\n",
           get_density_count(i, logN()), logN() - leaf::max_element_size,
           merged_data.size, total_leaves(), i / logN());
  }
  if constexpr (!compressed && !traits::maintain_offsets) {
    for (uint64_t i = 0; i < total_leaves(); i++) {
      assert(get_leaf(i).check_increasing_or_zero());
    }
  }
  if constexpr (!compressed && traits::maintain_offsets) {
    for (size_t i = 0; i < offsets_array.size - 1; i++) {
      void *loc = offsets_array.locations[i];
      key_type *loc2 = reinterpret_cast<key_type *>(loc);
      assert((*loc2) == (pcsr_top_bit | i));
    }
  }
#endif
  double_timer.stop();
}

// halves the size of the base array
// assumes we already have all the locks from the lock array and the big lock
template <typename traits> void CPMA<traits>::shrink_list(uint64_t times) {
  if (meta_data_index == 0) {
    return;
  }
  auto merged_data =
      leaf::template merge<head_form == InPlace, store_density, parallel>(
          get_data_ptr(0), total_leaves(), logN(), 0,
          [this](uint64_t index) -> element_ref_type {
            return index_to_head(index);
          },
          density_array());
  // TODO(wheatman) deal with the fact that we copy the element here
  element_type zero_element;
  if constexpr (!binary) {
    zero_element = get_zero_el_ref();
  }
  // assert(merged_data.leaf.head != 0 ||
  //        merged_data.leaf.template used_size<head_form == InPlace>() == 0);
  meta_data_index -= times;
  if constexpr (!fixed_size) {
    free(underlying_array);
    // steal an extra few bytes to ensure we never read off the end
    uint64_t allocated_size = underlying_array_size() + 32;
    if (allocated_size % 32 != 0) {
      allocated_size += 32 - (allocated_size % 32);
    }
    underlying_array = aligned_alloc(32, allocated_size);
  }
  if constexpr (!binary) {
    get_zero_el_ref() = zero_element;
  } else {
    std::get<0>(get_data_ref(soa_num_spots())) = 0;
  }
  if constexpr (head_form != InPlace) {
    std::fill(head_array(),
              head_array() + (head_array_size() / sizeof(key_type)), 0);
  }

  if constexpr (support_rank) {
    std::fill(rank_tree_array(),
              rank_tree_array() + nextPowerOf2(total_leaves()), 0);
  }

  merged_data.leaf
      .template split<head_form == InPlace, store_density, support_rank,
                      parallel, traits::maintain_offsets>(
          total_leaves(), merged_data.size, logN(), get_data_ptr(0), 0,
          [this](uint64_t index) -> element_ref_type {
            return index_to_head(index);
          },
          density_array(), rank_tree_array(), total_leaves(), offsets_array);
  merged_data.free();
  if constexpr (traits::maintain_offsets) {
    offsets_array.locations[offsets_array.size - 1] =
        data_array() + soa_num_spots();
  }
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
  if constexpr (parallel) {
    if (num_leaves > ParallelTools::getWorkers() * 1024) {
      ParallelTools::Reducer_sum<uint64_t> total_red;
      ParallelTools::parallel_for(0, num_leaves, [&](int64_t i) {
        if constexpr (store_density) {
          uint64_t val = density_array()[byte_index / logN() + i];
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
      uint64_t val = density_array()[byte_index / logN() + i];
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
  if constexpr (parallel) {
    if (num_leaves > ParallelTools::getWorkers() * 1024) {
      ParallelTools::Reducer_sum<uint64_t> total_red;
      ParallelTools::parallel_for(0, num_leaves, [&](int64_t i) {
        if constexpr (store_density) {
          total_red.add(density_array()[byte_index / logN() + i]);
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
      ASSERT(density_array()[byte_index / logN() + i] ==
                 l.template used_size_no_overflow<head_form == InPlace>(),
             "got %d, expected %lu\n",
             +density_array()[byte_index / logN() + i],
             l.template used_size_no_overflow<head_form == InPlace>());
#endif
      total += density_array()[byte_index / logN() + i];
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
  // this path shouldn't be taken since 0 is handled specially
  assert(e != 0);
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
  // TODO(wheatman) fix the prefetch in this function with the data array
  // shifted over by 1 for the zero elements value
  if (N() == logN()) {
    return 0;
  }
  // this path shouldn't be taken since 0 is handled specially
  assert(e != 0);
  search_cnt.add(1);
  if constexpr (head_form == Eytzinger) {
    if (start == 0 && end == std::numeric_limits<uint64_t>::max()) {
      uint64_t length = total_leaves_rounded_up();
      uint64_t value_to_check = length / 2;
      uint64_t length_to_add = length / 4 + 1;
      uint64_t e_index = 0;
      while (length_to_add > 0) {
        if (head_array()[e_index] == e) {
          ASSERT(value_to_check * elts_per_leaf() ==
                     find_containing_leaf_index_debug(e, start, end),
                 "got %lu, expected %lu", value_to_check * elts_per_leaf(),
                 find_containing_leaf_index_debug(e, start, end));
          __builtin_prefetch(&key_array()[value_to_check * (elts_per_leaf())]);
          return value_to_check * elts_per_leaf();
        }
        if (e <= static_cast<key_type>(head_array()[e_index] - 1)) {
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
      if (e < head_array()[e_index] && value_to_check > 0) {
        value_to_check -= 1;
      }
      ASSERT(value_to_check * elts_per_leaf() ==
                 find_containing_leaf_index_debug(e, start, end),
             "got %lu, expected %lu\n", value_to_check * elts_per_leaf(),
             find_containing_leaf_index_debug(e, start, end));
      __builtin_prefetch(&key_array()[value_to_check * (elts_per_leaf())]);
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

        if (head_array()[e_index] == e) {
          ASSERT(value_to_check * elts_per_leaf() ==
                     find_containing_leaf_index_debug(e, start, end),
                 "got %lu, expected %lu\n", value_to_check * elts_per_leaf(),
                 find_containing_leaf_index_debug(e, start, end));
          __builtin_prefetch(&key_array()[value_to_check * (elts_per_leaf())]);
          return value_to_check * elts_per_leaf();
        }
        if (e <= static_cast<key_type>(head_array()[e_index] - 1)) {
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
      } else if (e < head_array()[e_index] && value_to_check > 0) {
        value_to_check -= 1;
      }
      ASSERT(value_to_check * elts_per_leaf() ==
                 find_containing_leaf_index_debug(e, start, end),
             "got %lu, expected %lu, elts_per_leaf = %lu, start = %lu, end = "
             "%lu\n",
             value_to_check * elts_per_leaf(),
             find_containing_leaf_index_debug(e, start, end), elts_per_leaf(),
             start, end);
      __builtin_prefetch(&key_array()[value_to_check * (elts_per_leaf())]);
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
                head_array()[block_number * (B_size - 1) + i] > e;
          }

          for (uint64_t i = 0; i < B_size - 1; i++) {
            number_in_block_greater_item +=
                head_array()[block_number * (B_size - 1) + i] == 0;
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
      __builtin_prefetch(&key_array()[leaf_index * (elts_per_leaf())]);
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
          ASSERT(head_array()[block_number * (B_size - 1) + i] != 0,
                 "looked at a zero entry, shouldn't have happened, "
                 "block_number = %lu, i = %lu, amount_to_add = %lu, start = "
                 "%lu, end = %lu, leaf_index = %lu\n",
                 block_number, i, amount_to_add, start, end, leaf_index);
          number_in_block_greater_item +=
              head_array()[block_number * (B_size - 1) + i] > e;
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
    __builtin_prefetch(&key_array()[leaf_index * (elts_per_leaf())]);
    return leaf_index * elts_per_leaf();
  }
  ASSERT(end > start, "end = %lu, start = %lu\n", end, start);
  if (end > N() / sizeof(key_type)) {
    end = N() / sizeof(key_type);
  }
  assert((start * sizeof(key_type)) % logN() == 0);
  uint64_t size = (end - start) / elts_per_leaf();
  if (size <= 1) {
    return start;
  }
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
      __builtin_prefetch(&key_array()[(idx + i - 1) * (elts_per_leaf())]);
      return (idx + i - 1) * (elts_per_leaf());
    }
  }
  __builtin_prefetch(&key_array()[(idx + end_linear - 1) * (elts_per_leaf())]);
  return (idx + end_linear - 1) * (elts_per_leaf());
}

template <typename traits>
void CPMA<traits>::print_array_region(uint64_t start_leaf,
                                      uint64_t end_leaf) const {
  for (uint64_t i = start_leaf; i < end_leaf; i++) {
    printf("LEAF NUMBER %lu, STARTING IDX %lu, BYTE IDX %lu", i,
           i * elts_per_leaf(), i * logN());
    if constexpr (store_density) {
      printf(", density_array says: %d", +density_array()[i]);
    }
    printf("\n");
    get_leaf(i).print();
  }
}

template <typename traits> void CPMA<traits>::print_array() const {
  print_array_region(0, total_leaves());
  if constexpr (support_rank) {
    printf("rank tree array:\n");
    for (uint64_t i = 0; i < nextPowerOf2(total_leaves()); i++) {
      printf("%lu, ", rank_tree_array()[i]);
    }
    printf("\n");
  }
}

template <typename traits> void CPMA<traits>::print_pma() const {
  printf("N = %lu, logN = %lu, loglogN = %lu, H = %lu\n", N(), logN(),
         loglogN(), H());
  printf("count_elements %lu\n", size());
  if (has_0) {
    printf("has 0\n");
    if constexpr (!binary) {
      std::cout << "value of 0 is " << leftshift_tuple(get_zero_el_ref())
                << "\n";
    }
  }
  if (get_element_count()) {
    print_array();
  } else {
    printf("The PMA is empty\n");
  }
}

template <typename traits> CPMA<traits>::CPMA() {
  static_assert(!traits::maintain_offsets);
  if constexpr (!fixed_size) {
    uint64_t allocated_size = underlying_array_size() + 32;
    if (allocated_size % 32 != 0) {
      allocated_size += 32 - (allocated_size % 32);
    }
    underlying_array = aligned_alloc(32, allocated_size);
  }

  std::fill((uint8_t *)underlying_array,
            (uint8_t *)underlying_array + underlying_array_size(), 0);
}

template <typename traits> CPMA<traits>::CPMA(key_type *start, key_type *end) {
  static_assert(!traits::maintain_offsets);
  if constexpr (!fixed_size) {
    uint64_t allocated_size = underlying_array_size() + 32;
    if (allocated_size % 32 != 0) {
      allocated_size += 32 - (allocated_size % 32);
    }
    underlying_array = aligned_alloc(32, allocated_size);
  }
  std::fill((uint8_t *)underlying_array,
            (uint8_t *)underlying_array + underlying_array_size(), 0);
  insert_batch(start, end - start);
}

template <typename traits> CPMA<traits>::CPMA(auto &range) {
  static_assert(!traits::maintain_offsets);
  // TODO(wheatman) get this working for compressed data
  if constexpr (compressed || fixed_size) {
    if constexpr (!fixed_size) {
      uint64_t allocated_size = underlying_array_size() + 32;
      if (allocated_size % 32 != 0) {
        allocated_size += 32 - (allocated_size % 32);
      }
      underlying_array = aligned_alloc(32, allocated_size);
    }
    std::fill((uint8_t *)underlying_array,
              (uint8_t *)underlying_array + underlying_array_size(), 0);
    insert_batch(range.data(), range.size());
  } else {
    parlay::sort_inplace(range);
    void *leaf_init = nullptr;
    element_type zero_element;
    size_t soa_num_elements;
    if constexpr (binary) {
      auto elements = parlay::unique(range);
      soa_num_elements = elements.size();
      size_t start_idx = 0;
      if (elements[0] == 0) {
        has_0 = true;
        start_idx = 1;
        soa_num_elements -= 1;
      }
      leaf_init = malloc(SOA_type::get_size_static(soa_num_elements));
      ParallelTools::parallel_for(start_idx, elements.size(), [&](size_t i) {
        SOA_type::get_static(leaf_init, soa_num_elements, i - start_idx) =
            elements[i];
      });

    } else {
      auto element_pairs = parlay::delayed_map(range, [](auto elem) {
        return std::make_pair(std::get<0>(elem), leftshift_tuple(elem));
      });
      auto elements = parlay::reduce_by_key(element_pairs, value_update());
      parlay::sort_inplace(elements);
      soa_num_elements = elements.size();
      size_t start_idx = 0;
      if (std::get<0>(elements[0]) == 0) {
        has_0 = true;
        zero_element = elements[0];
        start_idx = 1;
        soa_num_elements -= 1;
      }

      leaf_init = malloc(SOA_type::get_size_static(soa_num_elements));
      ParallelTools::parallel_for(start_idx, elements.size(), [&](size_t i) {
        SOA_type::get_static(leaf_init, soa_num_elements, i - start_idx) =
            std::tuple_cat(std::make_tuple(elements[i].first),
                           elements[i].second);
      });
    }
    auto leaf_soa = SOA_type(leaf_init, soa_num_elements);

    typename traits::leaf leaf(leaf_soa.get(0), leaf_soa.get_ptr(1),
                               soa_num_elements * sizeof(key_type));

    size_t bytes_required = soa_num_elements * sizeof(key_type);
    uint64_t grow_times = 0;
    while (meta_data[grow_times].n <= bytes_required) {
      grow_times += 1;
    }
    meta_data_index = grow_times;
    count_elements_ = soa_num_elements;

    // steal an extra few bytes to ensure we never read off the end
    uint64_t allocated_size = underlying_array_size() + 32;
    if (allocated_size % 32 != 0) {
      allocated_size += 32 - (allocated_size % 32);
    }
    underlying_array = aligned_alloc(32, allocated_size);
    if constexpr (head_form != InPlace) {
      ParallelTools::parallel_for(0, (head_array_size() / sizeof(key_type)),
                                  [&](size_t i) { head_array()[i] = 0; });
    }
    if constexpr (!binary) {
      if (has_0) {
        get_zero_el_ref() = zero_element;
      }
    }
    leaf.template split<head_form == InPlace, store_density, support_rank,
                        parallel, traits::maintain_offsets>(
        total_leaves(), count_elements_, logN(), get_data_ptr(0), 0,
        [this](uint64_t index) -> element_ref_type {
          return index_to_head(index);
        },
        density_array(), rank_tree_array(), total_leaves(), offsets_array);
  }
}

template <typename traits> bool CPMA<traits>::has(key_type e) const {
  if (size() == 0) {
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
  if (size() == 0) {
    return {};
  }
  if (e == 0) {
    if (has_0) {
      // if we have a zero its stored right before the normal data
      return leftshift_tuple(get_zero_el_ref());
    } else {
      return {};
    }
  }
  uint64_t leaf_number = find_containing_leaf_number(e);
  return get_leaf(leaf_number).template value<head_form == InPlace>(e);
}

// rank only has meaning if it has the element
template <typename traits>
std::pair<bool, uint64_t> CPMA<traits>::has_and_rank(key_type e) const {
  if (size() == 0) {
    return {false, std::numeric_limits<uint64_t>::max()};
  }
  if (e == 0) {
    if (has_0) {
      return {true, 0};
    } else {
      return {false, std::numeric_limits<uint64_t>::max()};
    }
  }
  uint64_t leaf_number = find_containing_leaf_number(e);
  bool found = get_leaf(leaf_number).contains(e);
  if (found) {
    return {true, rank_given_leaf(leaf_number, e) + has_0};
  } else {
    return {false, std::numeric_limits<uint64_t>::max()};
  }
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
      printf("idx is %lu, byte_idx = %lu, logN = %lu, N = %lu, size = "
             "%lu\n",
             idx / sizeof(key_type), idx, logN(), N(), size());
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

template <bool head_in_place, bool parallel, typename leaf, typename key_type>
bool everything_from_batch_added(leaf l, key_type *start, key_type *end) {
  bool have_everything = true;
  if (end - start < 10000000) {
    ParallelTools::For<parallel>(0, end - start, [&](size_t i) {
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
template <class Vector_pairs>
uint64_t CPMA<traits>::insert_batch_internal(element_ptr_type e,
                                             uint64_t batch_size,
                                             Vector_pairs &leaves_to_check,
                                             uint64_t start_leaf_idx,
                                             uint64_t end_leaf_idx,
                                             Vector_pairs &rank_additions) {
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
            .template merge_into_leaf<head_form == InPlace, parallel>(
                current_elt_ptr, e.get_pointer() + batch_size, next_head);

    assert((everything_from_batch_added<head_form == InPlace, parallel>(
        get_leaf(leaf_idx / elts_per_leaf()), current_elt_ptr.get_pointer(),
        std::get<0>(result).get_pointer())));
    current_elt_ptr = std::get<0>(result);
    // number of elements merged is the number of distinct elts merged into
    // this leaf
    num_elts_merged += std::get<1>(result);
    if constexpr (support_rank) {
      rank_additions.push_back(
          {leaf_idx / elts_per_leaf(), std::get<1>(result)});
    }
    // number of bytes used in this leaf (if exceeds logN(), merge_into_leaf
    // will have written some auxiliary memory
    auto bytes_used = std::get<2>(result);
    if (std::get<1>(result)) {
      if constexpr (store_density) {
        density_array()[leaf_idx / elts_per_leaf()] = std::min(
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
template <class Vector_pairs>
uint64_t CPMA<traits>::insert_batch_internal_small_batch(
    element_ptr_type e, uint64_t batch_size, Vector_pairs &leaves_to_check,
    uint64_t start_leaf_idx, uint64_t end_leaf_idx,
    Vector_pairs &rank_additions) {
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
        get_leaf(leaf_number)
            .template insert<head_form == InPlace, value_update,
                             traits::maintain_offsets>(*e, value_update(),
                                                       offsets_array);
    if (!inserted) {
      return 0;
    }
    if constexpr (support_rank) {
      rank_additions.push_back({leaf_number, 1});
    }
    if constexpr (store_density) {
      density_array()[leaf_number] =
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

  if (batch_size * 10 >= (end_leaf_idx - start_leaf_idx) / elts_per_leaf() &&
      batch_size < 1000) {
    while (start_leaf_idx + elts_per_leaf() < end_leaf_idx &&
           e.get() <
               index_to_head_key((start_leaf_idx / elts_per_leaf()) + 1)) {
      auto result =
          get_leaf(start_leaf_idx / elts_per_leaf())
              .template merge_into_leaf<head_form == InPlace, parallel,
                                        value_update, traits::maintain_offsets>(
                  e, e.get_pointer() + batch_size,
                  index_to_head_key((start_leaf_idx / elts_per_leaf()) + 1),
                  value_update(), offsets_array);
      num_elts_merged += std::get<1>(result);

      if constexpr (support_rank) {
        if (std::get<1>(result)) {
          rank_additions.push_back(
              {start_leaf_idx / elts_per_leaf(), std::get<1>(result)});
        }
      }
      if (std::get<1>(result)) {
        auto bytes_used = std::get<2>(result);
        if constexpr (store_density) {
          density_array()[start_leaf_idx / elts_per_leaf()] = std::min(
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
  auto result =
      get_leaf(leaf_idx / elts_per_leaf())
          .template merge_into_leaf<head_form == InPlace, parallel,
                                    value_update, traits::maintain_offsets>(
              middle, e.get_pointer() + batch_size, next_head, value_update(),
              offsets_array);
  // the middle element should have been merged in
  assert(std::get<0>(result) > e + (batch_size / 2));
  num_elts_merged += std::get<1>(result);

  if constexpr (support_rank) {
    if (std::get<1>(result)) {
      rank_additions.push_back(
          {leaf_idx / elts_per_leaf(), std::get<1>(result)});
    }
  }
  auto bytes_used = std::get<2>(result);

  if (std::get<1>(result)) {
    if constexpr (store_density) {
      density_array()[leaf_idx / elts_per_leaf()] =
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

  if constexpr (!parallel) {
    ret1 = insert_batch_internal_small_batch(e, early_stage_size,
                                             leaves_to_check, start_leaf_idx,
                                             leaf_idx, rank_additions);
    ret2 = insert_batch_internal_small_batch(
        std::get<0>(result), late_stage_size, leaves_to_check,
        leaf_idx + elts_per_leaf(), end_leaf_idx, rank_additions);
  } else {

    if (early_stage_size <= 20 || late_stage_size <= 20) {
      ret1 = insert_batch_internal_small_batch(e, early_stage_size,
                                               leaves_to_check, start_leaf_idx,
                                               leaf_idx, rank_additions);
      ret2 = insert_batch_internal_small_batch(
          std::get<0>(result), late_stage_size, leaves_to_check,
          leaf_idx + elts_per_leaf(), end_leaf_idx, rank_additions);
    } else {
      ParallelTools::par_do(
          [&]() {
            ret1 = insert_batch_internal_small_batch(
                e, early_stage_size, leaves_to_check, start_leaf_idx, leaf_idx,
                rank_additions);
          },
          [&]() {
            ret2 = insert_batch_internal_small_batch(
                std::get<0>(result), late_stage_size, leaves_to_check,
                leaf_idx + elts_per_leaf(), end_leaf_idx, rank_additions);
          });
    }
  }
  num_elts_merged += ret1;
  num_elts_merged += ret2;
  return num_elts_merged;
}

template <typename traits>
template <class Vector_pairs>
uint64_t CPMA<traits>::remove_batch_internal_small_batch(
    key_type *e, uint64_t batch_size, Vector_pairs &leaves_to_check,
    uint64_t start_leaf_idx, uint64_t end_leaf_idx,
    Vector_pairs &rank_additions) {
  if (batch_size == 0 || start_leaf_idx == end_leaf_idx) {
    return 0;
  }
  // not technically true, but probably true
  assert(batch_size < 1000000000000UL);
  ASSERT(start_leaf_idx < end_leaf_idx,
         "start_leaf_idx = %lu, end_leaf_idx = %lu, total_size = %lu\n",
         start_leaf_idx, end_leaf_idx, N() / sizeof(key_type));
  if (batch_size == 1) {
    uint64_t leaf_number =
        find_containing_leaf_number(*e, start_leaf_idx, end_leaf_idx);
    auto [removed, bytes_used] =
        get_leaf(leaf_number)
            .template remove<head_form == InPlace, traits::maintain_offsets>(
                *e, offsets_array);
    if (!removed) {
      return 0;
    }
    if constexpr (support_rank) {
      rank_additions.push_back({leaf_number, -1});
    }
    if constexpr (store_density) {
      density_array()[leaf_number] =
          std::min(bytes_used,
                   static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()));
    }
    ASSERT(bytes_used == get_density_count(leaf_number * logN(), logN()),
           "got %lu, expected %lu\n", bytes_used,
           get_density_count(leaf_number * logN(), logN()));
    if (bytes_used < logN() * lower_density_bound(H()) || bytes_used == 0) {
      leaves_to_check.push_back({leaf_number * elts_per_leaf(), bytes_used});
    }
    return 1;
  }

  uint64_t num_elts_removed = 0;

  if (batch_size * 10 >= (end_leaf_idx - start_leaf_idx) / elts_per_leaf() &&
      batch_size < 1000) {
    while (start_leaf_idx + elts_per_leaf() < end_leaf_idx &&
           *e < index_to_head_key((start_leaf_idx / elts_per_leaf()) + 1)) {
      auto result =
          get_leaf(start_leaf_idx / elts_per_leaf())
              .template strip_from_leaf<head_form == InPlace>(
                  e, e + batch_size,
                  index_to_head_key((start_leaf_idx / elts_per_leaf()) + 1));
      num_elts_removed += std::get<1>(result);

      if constexpr (support_rank) {
        if (std::get<1>(result)) {
          rank_additions.push_back(
              {start_leaf_idx / elts_per_leaf(), -1 * std::get<1>(result)});
        }
      }
      if (std::get<1>(result)) {
        auto bytes_used = std::get<2>(result);
        if constexpr (store_density) {
          density_array()[start_leaf_idx / elts_per_leaf()] = std::min(
              bytes_used,
              static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()));
        }
        ASSERT(bytes_used ==
                   get_density_count(start_leaf_idx * sizeof(key_type), logN()),
               "got %lu, expected %lu\n", bytes_used,
               get_density_count(start_leaf_idx * sizeof(key_type), logN()));
        // if exceeded leaf density bound, add self to per-worker queue for
        // leaves to rebalance
        if (bytes_used < logN() * lower_density_bound(H()) || bytes_used == 0) {
          leaves_to_check.push_back({start_leaf_idx, bytes_used});
        }
      }
      uint64_t num_elements_in = std::get<0>(result) - e;
      batch_size = batch_size - num_elements_in;
      if (batch_size == 0) {
        return num_elts_removed;
      }
      e = std::get<0>(result);
      start_leaf_idx += elts_per_leaf();
    }
  }

  // else we want to start with the middle element
  key_type *middle = e + (batch_size / 2);
  uint64_t leaf_idx =
      find_containing_leaf_index(*middle, start_leaf_idx, end_leaf_idx);
  // then we want to find the first element in the batch which is in the same
  // leaf as the middle element
  assert(leaf_idx / elts_per_leaf() < total_leaves());
  key_type head = index_to_head_key(leaf_idx / elts_per_leaf());
  if (leaf_idx == 0) {
    middle = e;
  }
  while ((middle - 1 >= e) && ((middle[-1]) >= head)) {
    middle = middle - 1;
  }

  uint64_t next_head = std::numeric_limits<uint64_t>::max(); // max int
  if (leaf_idx + elts_per_leaf() < end_leaf_idx) {
    next_head = index_to_head_key((leaf_idx / elts_per_leaf()) + 1);
  }
  assert(*middle < next_head);
  auto result = get_leaf(leaf_idx / elts_per_leaf())
                    .template strip_from_leaf<head_form == InPlace>(
                        middle, e + batch_size, next_head);
  num_elts_removed += std::get<1>(result);

  if constexpr (support_rank) {
    if (std::get<1>(result)) {
      rank_additions.push_back(
          {leaf_idx / elts_per_leaf(), -1 * std::get<1>(result)});
    }
  }
  auto bytes_used = std::get<2>(result);

  if (std::get<1>(result)) {
    if constexpr (store_density) {
      density_array()[leaf_idx / elts_per_leaf()] =
          std::min(bytes_used,
                   static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()));
    }
    ASSERT(bytes_used == get_density_count(leaf_idx * sizeof(key_type), logN()),
           "got %lu, expected %lu\n", bytes_used,
           get_density_count(leaf_idx * sizeof(key_type), logN()));
    // if exceeded leaf density bound, add self to per-worker queue for
    // leaves to rebalance
    if (bytes_used < logN() * lower_density_bound(H()) || bytes_used == 0) {
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

  if constexpr (!parallel) {
    ret1 = remove_batch_internal_small_batch(e, early_stage_size,
                                             leaves_to_check, start_leaf_idx,
                                             leaf_idx, rank_additions);
    ret2 = remove_batch_internal_small_batch(
        std::get<0>(result), late_stage_size, leaves_to_check,
        leaf_idx + elts_per_leaf(), end_leaf_idx, rank_additions);
  } else {

    if (early_stage_size <= 20 || late_stage_size <= 20) {
      ret1 = remove_batch_internal_small_batch(e, early_stage_size,
                                               leaves_to_check, start_leaf_idx,
                                               leaf_idx, rank_additions);
      ret2 = remove_batch_internal_small_batch(
          std::get<0>(result), late_stage_size, leaves_to_check,
          leaf_idx + elts_per_leaf(), end_leaf_idx, rank_additions);
    } else {
      ParallelTools::par_do(
          [&]() {
            ret1 = remove_batch_internal_small_batch(
                e, early_stage_size, leaves_to_check, start_leaf_idx, leaf_idx,
                rank_additions);
          },
          [&]() {
            ret2 = remove_batch_internal_small_batch(
                std::get<0>(result), late_stage_size, leaves_to_check,
                leaf_idx + elts_per_leaf(), end_leaf_idx, rank_additions);
          });
    }
  }
  num_elts_removed += ret1;
  num_elts_removed += ret2;
  return num_elts_removed;
}

// mark all leaves that exceed their density bound
template <typename traits>
template <class Vector_pairs>
uint64_t CPMA<traits>::remove_batch_internal(key_type *e, uint64_t batch_size,
                                             Vector_pairs &leaves_to_check,
                                             uint64_t start_leaf_idx,
                                             uint64_t end_leaf_idx,
                                             Vector_pairs &rank_additions) {
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
    if constexpr (support_rank) {
      rank_additions.push_back(
          {leaf_idx / elts_per_leaf(), -1 * std::get<1>(result)});
    }
    // number of bytes used in this leaf (if exceeds logN(), merge_into_leaf
    // will have written some auxiliary memory
    auto bytes_used = std::get<2>(result);

    // if exceeded leaf density bound, add self to per-worker queue for leaves
    // to rebalance
    if (std::get<1>(result)) {
      if constexpr (store_density) {
        density_array()[leaf_idx / elts_per_leaf()] = bytes_used;
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
    const std::pair<uint64_t, uint64_t> *begin,
    const std::pair<uint64_t, uint64_t> *end) const {
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
CPMA<traits>::get_ranges_to_redistibute_lookup_sibling_count(
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

    if (length >= 1000) {
      uint64_t left = 0;
      uint64_t right = 0;

      ParallelTools::par_do(
          [&]() {
            left = get_ranges_to_redistibute_lookup_sibling_count(
                ranges_check, start, length / 2, level - 1);
          },
          [&]() {
            right = get_ranges_to_redistibute_lookup_sibling_count(
                ranges_check, start + length / 2, length / 2, level - 1);
          });
      return left + right;
    } else {
      uint64_t left = get_ranges_to_redistibute_lookup_sibling_count(
          ranges_check, start, length / 2, level - 1);
      uint64_t right = get_ranges_to_redistibute_lookup_sibling_count(
          ranges_check, start + length / 2, length / 2, level - 1);
      return left + right;
    }
  }
  return it->second;
}

template <typename traits>
std::pair<std::vector<std::tuple<uint64_t, uint64_t>>, std::optional<uint64_t>>
CPMA<traits>::get_ranges_to_redistibute_debug(
    const std::vector<std::pair<uint64_t, uint64_t>> &leaves_to_check,
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

  ranges_to_redistribute_2 = get_ranges_to_redistibute_internal(
      leaves_to_check.data(), leaves_to_check.data() + leaves_to_check.size());

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
std::pair<ParallelTools::Reducer_Vector<std::tuple<uint64_t, uint64_t>>,
          std::optional<uint64_t>>
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
#if defined(NO_TLX)
    std::map<uint64_t, uint64_t> ranges_to_redistribute_2;
#else
    tlx::btree_map<uint64_t, uint64_t> ranges_to_redistribute_2;
#endif
    uint64_t full_opt = std::numeric_limits<uint64_t>::max();
    // (height) -> (start) -> (bytes_used)
    std::vector<ska::flat_hash_map<uint64_t, uint64_t>> ranges_check(2);

    uint64_t length_in_index = 2 * (elts_per_leaf());
    {
      uint64_t level = 1;
      uint64_t child_length_in_index = length_in_index / 2;

      leaves_to_check.serial_for_each(
          [&](const std::pair<uint64_t, uint64_t> &p) {
            const auto &[child_range_start, child_byte_count] = p;
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
    if (ranges_to_redistribute_2.unlocked_value(0, 0) >= N()) {
      elements_reduce.push_back({0, N()});
    } else {

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
    }
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
    // TODO(wheatman) get rid of this copy by allowing the followup function to
    // take in either a vector or  reducer vector
    return {elements_reduce, {}};
  }
}
template <typename traits>
template <class F>
std::pair<std::vector<std::tuple<uint64_t, uint64_t>>, std::optional<uint64_t>>
CPMA<traits>::get_ranges_to_redistibute(
    const std::vector<std::pair<uint64_t, uint64_t>> &leaves_to_check,
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
    float current_density = (float)current_bytes_filled / N();
    if (bounds_check(0, current_density)) {
      all.stop();
      return {{}, current_bytes_filled};
    }
  }

  // leaves_to_check[i].clear();

  timer serial("serial");
  serial.start();

  // ranges to redistribute
  // start -> (length to redistribute)
  // needs to be sorted for deduplication
#if defined(NO_TLX)
  std::map<uint64_t, uint64_t> ranges_to_redistribute_2;
#else
  tlx::btree_map<uint64_t, uint64_t> ranges_to_redistribute_2;
#endif
  // (height) -> (start) -> (bytes_used)
  std::vector<ska::flat_hash_map<uint64_t, uint64_t>> ranges_check(2);

  uint64_t length_in_index = 2 * (elts_per_leaf());
  {
    uint64_t level = 1;
    uint64_t child_length_in_index = length_in_index / 2;
    for (const std::pair<uint64_t, uint64_t> &p : leaves_to_check) {
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
      float density = ((float)parent_byte_count) /
                      (length_in_index_local * sizeof(key_type));

      if (length_in_index_local >= N() / sizeof(key_type) ||
          !bounds_check(H() - level, density)) {
        if (length_in_index_local == N() / sizeof(key_type) &&
            bounds_check(0, density)) {
          serial.stop();
          all.stop();
          return {{}, parent_byte_count};
        }

        ranges_to_redistribute_2[parent_range_start] =
            length_in_index_local * sizeof(key_type);

      } else {
        ranges_check[level][parent_range_start] = parent_byte_count;
      }
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
          get_ranges_to_redistibute_lookup_sibling_count(
              ranges_check, sibling_range_start, child_length_in_index,
              level - 1);
      uint64_t parent_byte_count = child_byte_count + sibling_byte_count;
      float density = ((float)parent_byte_count) /
                      (length_in_index_local * sizeof(key_type));

      if (length_in_index_local >= N() / sizeof(key_type) ||
          !bounds_check(level_for_density, density)) {
        if (length_in_index_local == N() / sizeof(key_type) &&
            bounds_check(0, density)) {
          serial.stop();
          all.stop();
          return {{}, parent_byte_count};
        }

        ranges_to_redistribute_2[parent_range_start] =
            length_in_index_local * sizeof(key_type);

      } else {
        ranges_check[level][parent_range_start] = parent_byte_count;
      }
    }
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
  for (const auto &[start, len] : ranges) {
    auto merged_data =
        leaf::template merge<head_form == InPlace, store_density, parallel>(
            get_data_ptr(start), len / logN(), logN(), start / elts_per_leaf(),
            [this](uint64_t index) -> element_ref_type {
              return index_to_head(index);
            },
            density_array());

    if constexpr (std::is_same_v<leaf, delta_compressed_leaf<key_type>>) {
      assert(((double)merged_data.size) / (len / logN()) <
             (density_limit() * len));
    }

    // number of leaves, num elemtns in input leaf, num elements in
    // output leaf, dest region
    merged_data.leaf
        .template split<head_form == InPlace, store_density, support_rank,
                        parallel, traits::maintain_offsets>(
            len / logN(), merged_data.size, logN(), get_data_ptr(start),
            start / elts_per_leaf(),
            [this](uint64_t index) -> element_ref_type {
              return index_to_head(index);
            },
            density_array(), rank_tree_array(), total_leaves(), offsets_array);
    merged_data.free();
  }
}

template <typename traits>
void CPMA<traits>::redistribute_ranges(
    const ParallelTools::Reducer_Vector<std::tuple<uint64_t, uint64_t>>
        &ranges) {
  ranges.for_each([&](const auto &item) {
    auto start = std::get<0>(item);
    auto len = std::get<1>(item);

    auto merged_data =
        leaf::template merge<head_form == InPlace, store_density, parallel>(
            get_data_ptr(start), len / logN(), logN(), start / elts_per_leaf(),
            [this](uint64_t index) -> element_ref_type {
              return index_to_head(index);
            },
            density_array());

    if constexpr (std::is_same_v<leaf, delta_compressed_leaf<key_type>>) {
      assert(((double)merged_data.size) / (len / logN()) <
             (density_limit() * len));
    }

    // number of leaves, num elemtns in input leaf, num elements in
    // output leaf, dest region
    merged_data.leaf
        .template split<head_form == InPlace, store_density, support_rank,
                        parallel, traits::maintain_offsets>(
            len / logN(), merged_data.size, logN(), get_data_ptr(start),
            start / elts_per_leaf(),
            [this](uint64_t index) -> element_ref_type {
              return index_to_head(index);
            },
            density_array(), rank_tree_array(), total_leaves(), offsets_array);
    merged_data.free();
  });
}

// input: batch, number of elts in a batch
// return true if the element was inserted, false if it was already there
// return number of things inserted (not already there)
template <typename traits>
uint64_t CPMA<traits>::insert_batch(element_ptr_type e, uint64_t batch_size,
                                    bool sorted) {
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
  if (!sorted) {
    sort_timer.start();
    sort_batch(e, batch_size);
    sort_timer.stop();
  }
  bool inserted_zero = false;

  // TODO(wheatman) currently only works for unsigned types
  while (e.get() == 0) {
    if constexpr (!binary) {
      if (has_0) {
        value_update()(get_zero_el_ref(), e[0]);
      } else {
        get_zero_el_ref() = e[0];
      }
    }
    inserted_zero = true;
    has_0 = true;

    ++e;
    batch_size -= 1;
    if (batch_size == 0) {
      return inserted_zero;
    }
  }

  // total number of leaves
  uint64_t num_leaves = total_leaves();

  if constexpr (parallel) {

    // which leaves were touched during the merge
    ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>>
        leaves_to_check;

    ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>> rank_additions;

    uint64_t num_elts_merged = 0;

    timer merge_timer("merge_timer");
    merge_timer.start();

    if constexpr (false) {
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
        uint64_t range_size =
            uint64_t(batch_end_key - batch_start.get_pointer());
        // do the merge
        num_elts_merged_reduce.add(insert_batch_internal(
            batch_start, range_size, leaves_to_check,
            leaf_bounds[i].start_leaf_index, leaf_bounds[i].end_leaf_index,
            rank_additions));
      });
      num_elts_merged = num_elts_merged_reduce.get();
    } else {
      num_elts_merged += insert_batch_internal_small_batch(
          e, batch_size, leaves_to_check, 0, num_leaves * elts_per_leaf(),
          rank_additions);
    }
    merge_timer.stop();
    if constexpr (support_rank) {
      update_rank(rank_additions);
    }

    // if most leaves need to be redistributed, or many elements were added,
    // just check the root first to hopefully save walking up the tree
    timer range_finder_timer("range_finder_timer");
    range_finder_timer.start();
    auto [ranges_to_redistribute_3, full_opt] = get_ranges_to_redistibute(
        leaves_to_check, num_elts_merged, [&](uint64_t level, float density) {
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
      ranges_to_redistribute_3.for_each([](const auto &element) {
        std::cout << "( " << std::get<0>(element) << ", "
                  << std::get<1>(element) / sizeof(key_type) << ") ";
      });
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
      auto ranges_to_redistribute_4 = ranges_to_redistribute_3.get_sorted();
      for (size_t i = 0; i < ranges_to_redistribute_4.size(); i++) {
        if (ranges_to_redistribute_4[i] != ranges_debug[i]) {
          printf("element %lu doesn't match, got (%lu, %lu), expected (%lu,"
                 "%lu)\n",
                 i, std::get<0>(ranges_to_redistribute_4[i]),
                 std::get<1>(ranges_to_redistribute_4[i]),
                 std::get<0>(ranges_debug[i]), std::get<1>(ranges_debug[i]));
        }
      }
    }

    assert(ranges_to_redistribute_3.get_sorted() == ranges_debug);
    assert(full_opt == full_opt_debug);
#endif

    // doubling everything
    if (full_opt.has_value()) {
      timer double_timer("doubling");
      double_timer.start();

      uint64_t grow_times = 0;
      auto bytes_occupied = full_opt.value();
      assert(bytes_occupied < 1UL << 60UL);

      // min bytes necessary to meet the density bound
      // uint64_t bytes_required = bytes_occupied / upper_density_bound(0);
      uint64_t bytes_required =
          std::max(N() * growing_factor, bytes_occupied * growing_factor);

      while (meta_data[meta_data_index + grow_times].n <= bytes_required) {
        grow_times += 1;
      }

      grow_list(grow_times);
      assert(check_rank_array());
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
    assert(check_rank_array());
    count_elements_ += num_elts_merged;
    total_timer.stop();
    return num_elts_merged + inserted_zero;
  } else {

    // which leaves were touched during the merge
    std::vector<std::pair<uint64_t, uint64_t>> leaves_to_check;

    std::vector<std::pair<uint64_t, uint64_t>> rank_additions;

    uint64_t num_elts_merged = 0;

    timer merge_timer("merge_timer");
    merge_timer.start();

    num_elts_merged += insert_batch_internal_small_batch(
        e, batch_size, leaves_to_check, 0, num_leaves * elts_per_leaf(),
        rank_additions);

    merge_timer.stop();
    if constexpr (support_rank) {
      update_rank(rank_additions);
    }

    // if most leaves need to be redistributed, or many elements were added,
    // just check the root first to hopefully save walking up the tree
    timer range_finder_timer("range_finder_timer");
    range_finder_timer.start();
    auto [ranges_to_redistribute_3, full_opt] = get_ranges_to_redistibute(
        leaves_to_check, num_elts_merged, [&](uint64_t level, float density) {
          return density > upper_density_bound(level);
        });
    range_finder_timer.stop();

    // doubling everything
    if (full_opt.has_value()) {
      timer double_timer("doubling");
      double_timer.start();

      uint64_t grow_times = 0;
      auto bytes_occupied = full_opt.value();

      // min bytes necessary to meet the density bound
      // uint64_t bytes_required = bytes_occupied / upper_density_bound(0);
      uint64_t bytes_required =
          std::max(N() * growing_factor, bytes_occupied * growing_factor);
      // not technically true, but we don't have enough memory anyway
      assert((bytes_required < std::numeric_limits<uint64_t>::max() / 10));

      while (meta_data[meta_data_index + grow_times].n <= bytes_required) {
        grow_times += 1;
      }

      grow_list(grow_times);
      assert(check_rank_array());
      double_timer.stop();
    } else { // not doubling

      timer redistribute_timer("redistributing");
      redistribute_timer.start();
      redistribute_ranges(ranges_to_redistribute_3);
      redistribute_timer.stop();
    }
    assert(check_nothing_full());

    assert(check_leaf_heads());
    assert(check_rank_array());
    count_elements_ += num_elts_merged;
    total_timer.stop();
    return num_elts_merged + inserted_zero;
  }
}

// input: batch, number of elts in a batch
// return number of things removed
template <typename traits>
uint64_t CPMA<traits>::remove_batch(key_type *e, uint64_t batch_size,
                                    bool sorted) {
  assert(check_leaf_heads());
  static_timer total_timer("remove_batch");
  total_timer.start();

  if (get_element_count() == 0 || batch_size == 0) {
    return 0;
  }
  if (batch_size < 100) {
    uint64_t count = 0;
    for (uint64_t i = 0; i < batch_size; i++) {
      count += remove(e[i]);
    }
    return count;
  }

  static_timer sort_timer("sort_remove_batch");

  if (!sorted) {
    sort_timer.start();
    sort_batch(e, batch_size);
    sort_timer.stop();
  }

  // TODO(wheatman) currently only works for unsigned types
  while (*e == 0) {
    has_0 = false;
    if constexpr (!binary) {
      get_zero_el_ref() = element_type();
    }
    e += 1;
    batch_size -= 1;
    if (batch_size == 0) {
      return 0;
    }
  }

  // total number of leaves
  uint64_t num_leaves = total_leaves();

  if constexpr (parallel) {

    // which leaves were touched during the merge
    ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>>
        leaves_to_check;

    ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>> rank_additions;

    uint64_t num_elts_removed = 0;

    static_timer merge_timer("merge_timer_remove_batch");
    merge_timer.start();

    if constexpr (false) {

      // leaves per partition
      uint64_t split_points =
          std::min({(uint64_t)num_leaves / 10, (uint64_t)batch_size / 100,
                    (uint64_t)ParallelTools::getWorkers() * 10});
      split_points = std::max(split_points, 1UL);

      std::vector<leaf_bound_t> leaf_bounds = get_leaf_bounds(split_points);

      ParallelTools::Reducer_sum<uint64_t> num_elts_removed_reduce;

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
            leaf_bounds[i].start_leaf_index, leaf_bounds[i].end_leaf_index,
            rank_additions));
      });
      num_elts_removed = num_elts_removed_reduce.get();
      merge_timer.stop();
    } else {
      num_elts_removed += remove_batch_internal_small_batch(
          e, batch_size, leaves_to_check, 0, num_leaves * elts_per_leaf(),
          rank_additions);
    }
    if constexpr (support_rank) {
      update_rank(rank_additions);
    }

    auto ranges_pair = get_ranges_to_redistibute(
        leaves_to_check, num_elts_removed, [&](uint64_t level, float density) {
          return (density < lower_density_bound(level));
        });
    auto ranges_to_redistribute_3 = ranges_pair.first;

    auto full_opt = ranges_pair.second;

    // shrinking everything
    if (full_opt.has_value()) {
      static_timer shrinking_timer("shrinking");
      shrinking_timer.start();
      uint64_t shrink_times = 0;
      auto bytes_occupied = full_opt.value();

      // min bytes necessary to meet the density bound
      uint64_t bytes_required = bytes_occupied / lower_density_bound(0);
      if (bytes_required == 0) {
        bytes_required = 1;
      }

      while (meta_data[meta_data_index - shrink_times].n >= bytes_required) {
        shrink_times += 1;
        if (meta_data_index == shrink_times) {
          break;
        }
      }

      shrink_list(shrink_times);
      shrinking_timer.stop();
      assert(check_rank_array());
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
    assert(check_rank_array());
    return num_elts_removed;
  } else {
    // leaves per partition
    uint64_t split_points =
        std::min({(uint64_t)num_leaves / 10, (uint64_t)batch_size / 100,
                  (uint64_t)1000});
    split_points = std::max(split_points, 1UL);

    // which leaves were touched during the merge
    std::vector<std::pair<uint64_t, uint64_t>> leaves_to_check;

    uint64_t num_elts_removed = 0;

    std::vector<leaf_bound_t> leaf_bounds = get_leaf_bounds(split_points);

    static_timer merge_timer("merge_timer_remove_batch");
    merge_timer.start();
    std::vector<std::pair<uint64_t, uint64_t>> rank_additions;
    for (uint64_t i = 0; i < split_points; i++) {
      if (leaf_bounds[i].start_leaf_index == leaf_bounds[i].end_leaf_index) {
        continue;
      }

      // search for boundaries in batch
      key_type *batch_start =
          std::lower_bound(e, e + batch_size, leaf_bounds[i].start_elt);
      // if we are the first batch start at the begining
      if (i == 0) {
        batch_start = e;
      }
      if (batch_start == e + batch_size) {
        break;
      }
      uint64_t end_elt = leaf_bounds[i].end_elt;
      if (i == split_points - 1) {
        end_elt = std::numeric_limits<uint64_t>::max();
      }
      key_type *batch_end = std::lower_bound(e, e + batch_size, end_elt);

      if (batch_start == batch_end) {
        continue;
      }

      // number of elts we are merging
      uint64_t range_size = uint64_t(batch_end - batch_start);

      // do the merge
      num_elts_removed +=
          remove_batch_internal(batch_start, range_size, leaves_to_check,
                                leaf_bounds[i].start_leaf_index,
                                leaf_bounds[i].end_leaf_index, rank_additions);
    }
    merge_timer.stop();
    if constexpr (support_rank) {
      update_rank(rank_additions);
    }

    auto ranges_pair = get_ranges_to_redistibute(
        leaves_to_check, num_elts_removed, [&](uint64_t level, float density) {
          return (density < lower_density_bound(level));
        });
    auto ranges_to_redistribute_3 = ranges_pair.first;

    auto full_opt = ranges_pair.second;

    // shrinking everything
    if (full_opt.has_value()) {
      static_timer shrinking_timer("shrinking");
      shrinking_timer.start();
      uint64_t shrink_times = 0;
      auto bytes_occupied = full_opt.value();

      // min bytes necessary to meet the density bound
      uint64_t bytes_required = bytes_occupied / lower_density_bound(0);
      if (bytes_required == 0) {
        bytes_required = 1;
      }

      while (meta_data[meta_data_index - shrink_times].n >= bytes_required) {
        shrink_times += 1;
        if (meta_data_index == shrink_times) {
          break;
        }
      }
      shrink_list(shrink_times);
      shrinking_timer.stop();
      assert(check_rank_array());
    } else { // not doubling

      static_timer redistribute_timer("redistributing_remove_batch");
      redistribute_timer.start();
      redistribute_ranges(ranges_to_redistribute_3);
      redistribute_timer.stop();
    }
    count_elements_ -= num_elts_removed;
    assert(check_nothing_full());
    assert(check_leaf_heads());
    total_timer.stop();
    assert(check_rank_array());
    return num_elts_removed;
  }
}

template <typename traits>
void CPMA<traits>::insert_post_place(uint64_t leaf_number,
                                     uint64_t byte_count) {
  static_timer rebalence_timer("rebalence_insert_timer");

  count_elements_ += 1;
  update_rank(leaf_number, 1);
  assert(check_rank_array());
  if constexpr (store_density) {
    density_array()[leaf_number] = byte_count;
  }
  rebalence_timer.start();
  const uint64_t byte_index = leaf_number * logN();
  ASSERT(byte_count == get_density_count(byte_index, logN()),
         "got %lu, expected %lu\n", byte_count,
         get_density_count(byte_index, logN()));

  int64_t level = H();
  uint64_t len_bytes = logN();

  uint64_t node_byte_index = byte_index;

  // if we are not a power of 2, we don't want to go off the end
  uint64_t local_len_bytes = std::min(len_bytes, N() - node_byte_index);

  while (byte_count >= upper_density_bound(level) * local_len_bytes) {
    len_bytes *= 2;
    uint64_t new_byte_node_index = find_node(node_byte_index, len_bytes);
    local_len_bytes = std::min(len_bytes, N() - new_byte_node_index);
    if (local_len_bytes <= N()) {
      level--;

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
      if (level == -1) {
        if (byte_count >= upper_density_bound(0) * local_len_bytes) {
          grow_list(1);
          rebalence_timer.stop();
          assert(check_rank_array());
          return;
        } else {
          break;
        }
      }
    }
  }
  if (len_bytes > logN()) {
    auto merged_data =
        leaf::template merge<head_form == InPlace, store_density, parallel>(
            get_data_ptr(node_byte_index / sizeof(key_type)),
            local_len_bytes / logN(), logN(), node_byte_index / logN(),
            [this](uint64_t index) -> element_ref_type {
              return index_to_head(index);
            },
            density_array());

    merged_data.leaf
        .template split<head_form == InPlace, store_density, support_rank,
                        parallel, traits::maintain_offsets>(
            local_len_bytes / logN(), merged_data.size, logN(),
            get_data_ptr(node_byte_index / sizeof(key_type)),
            node_byte_index / logN(),
            [this](uint64_t index) -> element_ref_type {
              return index_to_head(index);
            },
            density_array(), rank_tree_array(), total_leaves(), offsets_array);
    assert(check_rank_array());
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
    if constexpr (!binary) {
      if (had_before) {
        value_update()(get_zero_el_ref(), e);
      } else {
        get_zero_el_ref() = std::move(e);
      }
    }
    has_0 = true;
    return !had_before;
  }
  total_timer.start();
  find_timer.start();
  uint64_t leaf_number = find_containing_leaf_number(std::get<0>(e));
  find_timer.stop();
  modify_timer.start();
  auto [inserted, byte_count] =
      get_leaf(leaf_number)
          .template insert<head_form == InPlace, value_update,
                           traits::maintain_offsets>(
              std::move(e), value_update(), offsets_array);
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
std::pair<bool, uint64_t> CPMA<traits>::insert_get_rank(element_type e) {
  static_timer total_timer("total_insert_timer");
  static_timer find_timer("find_insert_timer");
  static_timer modify_timer("modify_insert_timer");

  total_timer.start();
  find_timer.start();
  uint64_t leaf_number = find_containing_leaf_number(std::get<0>(e));
  find_timer.stop();
  modify_timer.start();
  auto [inserted, byte_count] =
      get_leaf(leaf_number).template insert<head_form == InPlace>(e);
  modify_timer.stop();

  uint64_t rank = rank_given_leaf(leaf_number, std::get<0>(e));
  if (has_0) {
    rank += 1;
  }

  if (!inserted) {
    total_timer.stop();
    return {false, rank};
  }
  insert_post_place(leaf_number, byte_count);
  total_timer.stop();
  return {true, rank};
}

template <typename traits>
bool CPMA<traits>::insert_by_rank(element_type e, uint64_t rank) {
  // can't handel the zero element when things aren't sorted since it is stored
  // out of the main array
  assert(std::get<0>(e) != 0);
  uint64_t length = nextPowerOf2(total_leaves());
  uint64_t add_amount = length / 2;
  uint64_t rank_tree_index = 0;
  uint64_t leaf_number = 0;
  while (add_amount > 0) {
    if (rank < rank_tree_array()[rank_tree_index] ||
        rank_tree_array()[rank_tree_index] == 0) {
      rank_tree_index = rank_tree_index * 2 + 1;
    } else {
      rank -= rank_tree_array()[rank_tree_index];
      rank_tree_index = rank_tree_index * 2 + 2;
      leaf_number += add_amount;
    }
    add_amount /= 2;
  }
  if (leaf_number >= total_leaves()) {
    // we got shoved off the end
    rank = std::numeric_limits<uint64_t>::max();
    leaf_number = total_leaves() - 1;
  }

  auto [inserted, byte_count] =
      get_leaf(leaf_number)
          .template insert_by_rank<head_form == InPlace>(e, rank);

  insert_post_place(leaf_number, byte_count);
  return true;
}

template <typename traits>
bool CPMA<traits>::update_by_rank(element_type e, uint64_t rank) {
  // can't handel the zero element when things aren't sorted since it is stored
  // out of the main array
  assert(std::get<0>(e) != 0);
  uint64_t length = nextPowerOf2(total_leaves());
  uint64_t add_amount = length / 2;
  uint64_t rank_tree_index = 0;
  uint64_t leaf_number = 0;
  while (add_amount > 0) {
    if (rank < rank_tree_array()[rank_tree_index] ||
        rank_tree_array()[rank_tree_index] == 0) {
      rank_tree_index = rank_tree_index * 2 + 1;
    } else {
      rank -= rank_tree_array()[rank_tree_index];
      rank_tree_index = rank_tree_index * 2 + 2;
      leaf_number += add_amount;
    }
    add_amount /= 2;
  }
  if (leaf_number >= total_leaves()) {
    // we got shoved off the end
    rank = std::numeric_limits<uint64_t>::max();
    leaf_number = total_leaves() - 1;
  }

  auto [inserted, byte_count] =
      get_leaf(leaf_number)
          .template update_by_rank<head_form == InPlace>(e, rank);

  if (inserted) {
    insert_post_place(leaf_number, byte_count);
    return true;
  }

  return false;
}

template <typename traits>
void CPMA<traits>::remove_post_place(uint64_t leaf_number,
                                     uint64_t byte_count) {
  static_timer rebalence_timer("rebalence_remove_timer");
  static_timer rank_timer("rank_remove_timer");

  count_elements_ -= 1;
  rank_timer.start();
  update_rank(leaf_number, -1);
  assert(check_rank_array());
  rank_timer.stop();
  if constexpr (store_density) {
    density_array()[leaf_number] = byte_count;
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
        leaf::template merge<head_form == InPlace, store_density, parallel>(
            get_data_ptr(node_byte_index / sizeof(key_type)),
            local_len_bytes / logN(), logN(), node_byte_index / logN(),
            [this](uint64_t index) -> element_ref_type {
              return index_to_head(index);
            },
            density_array());

    merged_data.leaf
        .template split<head_form == InPlace, store_density, support_rank,
                        parallel, traits::maintain_offsets>(
            local_len_bytes / logN(), merged_data.size, logN(),
            get_data_ptr(node_byte_index / sizeof(key_type)),
            node_byte_index / logN(),
            [this](uint64_t index) -> element_ref_type {
              return index_to_head(index);
            },
            density_array(), rank_tree_array(), total_leaves(), offsets_array);
    merged_data.free();
    assert(check_rank_array());
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
    if constexpr (!binary) {
      get_zero_el_ref() = element_type();
    }
    has_0 = false;
    return had_before;
  }
  total_timer.start();
  find_timer.start();
  uint64_t leaf_number = find_containing_leaf_number(e);
  find_timer.stop();
  modify_timer.start();
  auto [removed, byte_count] =
      get_leaf(leaf_number)
          .template remove<head_form == InPlace, traits::maintain_offsets>(
              e, offsets_array);
  modify_timer.stop();

  if (!removed) {
    total_timer.stop();
    return false;
  }
  remove_post_place(leaf_number, byte_count);

  total_timer.stop();
  return true;
}

template <typename traits>
void CPMA<traits>::update_rank(uint64_t leaf_changed, int64_t change_amount) {
  if constexpr (support_rank) {
    uint64_t node_size = nextPowerOf2(total_leaves()) / 2;
    uint64_t current_check = 0;
    uint64_t rank_tree_index = 0;

    while (node_size >= 1) {
      if (leaf_changed >= current_check + node_size) {
        current_check += node_size;
        rank_tree_index = rank_tree_index * 2 + 2;
      } else {
        rank_tree_array()[rank_tree_index] += change_amount;
        rank_tree_index = rank_tree_index * 2 + 1;
      }
      node_size /= 2;
    }
  }
}

template <typename traits>
void CPMA<traits>::update_rank(
    ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>>
        &rank_additions) {
  ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>> rank_additions2;
  rank_additions.for_each([&](const std::pair<uint64_t, uint64_t> &el) {
    if (el.first == total_leaves() - 1 &&
        total_leaves() == nextPowerOf2(total_leaves())) {
      return;
    }
    uint64_t ei = e_index(el.first, total_leaves() - 1);
    rank_tree_array()[ei] += el.second;
    if (ei == 0) {
      return;
    }
    uint64_t parent_ei = (ei - 1) / 2;
    while ((parent_ei * 2) + 1 != ei) {
      ei = parent_ei;
      parent_ei = (ei - 1) / 2;
      if (ei == 0) {
        return;
      }
    }
    rank_additions2.push_back({parent_ei, el.second});
  });
  while (!rank_additions2.empty()) {
    ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>>
        rank_additions3;

    auto elements = rank_additions2.get_sorted();
    // place a sentinal so we don't need to check if we are at the end
    elements.emplace_back(std::numeric_limits<uint64_t>::max(), 0);
    ParallelTools::parallel_for(0, elements.size() - 1, [&](size_t i) {
      if (i > 0 && elements[i].first == elements[i - 1].first) {
        return;
      }
      auto start = elements[i].first;
      do {
        auto &el = elements[i];
        rank_tree_array()[el.first] += el.second;
        i += 1;
        if (el.first == 0) {
          continue;
        }
        uint64_t parent_ei = (el.first - 1) / 2;
        uint64_t ei = el.first;
        bool add_to_set = true;
        while ((parent_ei * 2) + 1 != ei) {
          ei = parent_ei;
          parent_ei = (ei - 1) / 2;
          if (ei == 0) {
            add_to_set = false;
            break;
          }
        }
        if (add_to_set) {
          rank_additions3.push_back({parent_ei, el.second});
        }
      } while (elements[i].first == start);
    });
    rank_additions2 = rank_additions3;
  }
}

template <typename traits>
void CPMA<traits>::update_rank(
    std::vector<std::pair<uint64_t, uint64_t>> &rank_additions) {
  std::vector<std::pair<uint64_t, uint64_t>> rank_additions2;
  for (const std::pair<uint64_t, uint64_t> &el : rank_additions) {
    if (el.first == total_leaves() - 1 &&
        total_leaves() == nextPowerOf2(total_leaves())) {
      return;
    }
    uint64_t ei = e_index(el.first, total_leaves() - 1);
    rank_tree_array()[ei] += el.second;
    if (ei == 0) {
      return;
    }
    uint64_t parent_ei = (ei - 1) / 2;
    while ((parent_ei * 2) + 1 != ei) {
      ei = parent_ei;
      parent_ei = (ei - 1) / 2;
      if (ei == 0) {
        return;
      }
    }
    rank_additions2.emplace_back(parent_ei, el.second);
  }
  while (!rank_additions2.empty()) {
    std::vector<std::pair<uint64_t, uint64_t>> rank_additions3;
    for (const std::pair<uint64_t, uint64_t> &el : rank_additions2) {
      rank_tree_array()[el.first] += el.second;
      if (el.first == 0) {
        return;
      }
      uint64_t parent_ei = (el.first - 1) / 2;
      uint64_t ei = el.first;
      while ((parent_ei * 2) + 1 != ei) {
        ei = parent_ei;
        parent_ei = (ei - 1) / 2;
        if (ei == 0) {
          return;
        }
      }
      rank_additions3.emplace_back(parent_ei, el.second);
    }

    rank_additions2 = rank_additions3;
  }
}

template <typename traits>
uint64_t CPMA<traits>::rank_given_leaf(uint64_t leaf_number, key_type e) const {
  static_assert(support_rank);
  uint64_t node_size = nextPowerOf2(total_leaves()) / 2;
  uint64_t current_check = 0;
  uint64_t rank_tree_index = 0;
  uint64_t current_count = 0;
  while (node_size >= 1) {
    if (leaf_number >= current_check + node_size) {
      current_check += node_size;
      current_count += rank_tree_array()[rank_tree_index];
      rank_tree_index = rank_tree_index * 2 + 2;
    } else {
      rank_tree_index = rank_tree_index * 2 + 1;
    }
    node_size /= 2;
  }
  return current_count + get_leaf(leaf_number).rank(e);
}

template <typename traits> uint64_t CPMA<traits>::rank(key_type e) {
  static_assert(support_rank);
  if (e == 0) {
    return 0;
  }
  return rank_given_leaf(find_containing_leaf_number(e), e);
}

template <typename traits>
typename traits::element_type CPMA<traits>::select(uint64_t rank) const {
  static_assert(support_rank);
  if (rank == 0) {
    if (has_0) {
      if constexpr (binary) {
        return 0;
      } else {
        get_zero_el_ref();
      }
    }
  }
  if (has_0) {
    rank -= 1;
  }
  uint64_t length = nextPowerOf2(total_leaves());
  uint64_t add_amount = length / 2;
  uint64_t rank_tree_index = 0;
  uint64_t leaf_number = 0;
  while (add_amount > 0) {
    if (rank < rank_tree_array()[rank_tree_index] ||
        rank_tree_array()[rank_tree_index] == 0) {
      rank_tree_index = rank_tree_index * 2 + 1;
    } else {
      rank -= rank_tree_array()[rank_tree_index];
      rank_tree_index = rank_tree_index * 2 + 2;
      leaf_number += add_amount;
    }
    add_amount /= 2;
  }
  if (leaf_number >= total_leaves()) {
    leaf_number = total_leaves() - 1;
    rank = std::numeric_limits<uint64_t>::max();
  }
  return get_leaf(leaf_number).select(rank);
}

template <typename traits>
uint64_t CPMA<traits>::sum_serial(int64_t start, int64_t end) const {
#ifdef __AVX512F__
  if constexpr (!compressed && std::is_same_v<key_type, uint64_t>) {
    __m512i total_vec = _mm512_setzero_si512();
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
  if constexpr (parallel) {
    if (num_leaves < ParallelTools::getWorkers()) {
      return sum_serial(0, num_leaves);
    }
    return sum_parallel();
  }
  return sum_serial(0, num_leaves);
}

template <typename traits>
template <bool no_early_exit, class F>
bool CPMA<traits>::map_range(F f, key_type start_key, key_type end_key) const {
#pragma clang loop unroll_count(4)
  for (auto it = lower_bound(start_key); it != end(); ++it) {
    auto el = *it;
    if (el >= end_key) {
      return false;
    }
    assert(el >= start_key && el < end_key);
    if constexpr (!no_early_exit) {
      if (f(el)) {
        return false;
      }
    } else {
      f(el);
    }
  }
  return true;
}

template <typename traits>
template <class F>
uint64_t CPMA<traits>::map_range_length(F f, key_type start,
                                        uint64_t length) const {
  if (length == 0) {
    return 0;
  }
  uint64_t count = 0;
#pragma clang loop unroll_count(4)
  for (auto it = lower_bound(start); it != end(); ++it) {
    auto el = *it;
    assert(el >= start);
    f(el);
    count += 1;
    length -= 1;
    if (length == 0) {
      return count;
    }
  }
  return count;
}

template <typename traits>
template <bool no_early_exit, class F>
bool CPMA<traits>::map(F f) const {
  if (!size()) {
    return false;
  }
  if (has_0) {
    if constexpr (no_early_exit) {
      if constexpr (binary) {
        f(0);
      } else {
        f(get_zero_el_ref());
      }
    } else {
      if constexpr (binary) {
        if (f(0)) {
          return true;
        }
      } else {
        if (f(get_zero_el_ref())) {
          return true;
        }
      }
    }
  }
  for (size_t i = 0; i < total_leaves(); i++) {
    if constexpr (no_early_exit) {
      get_leaf(i).template map<no_early_exit, F>(f);
    } else {
      if (get_leaf(i).template map<no_early_exit, F>(f)) {
        return true;
      }
    }
  }
  return false;
}

template <typename traits>
template <class F>
bool CPMA<traits>::parallel_map(F f) const {
  if (!size()) {
    return false;
  }
  if (has_0) {
    if constexpr (binary) {
      f(0);
    } else {
      f(get_zero_el_ref());
    }
  }
  if (total_leaves() > 100) {
    ParallelTools::parallel_for(
        0, total_leaves(),
        [&](uint64_t idx) { get_leaf(idx).template map<true, F>(f); }

    );
  } else {
    for (size_t idx = 0; idx < total_leaves(); idx++) {
      get_leaf(idx).template map<true, F>(f);
    }
  }
  return false;
}

template <typename traits>
template <bool no_early_exit, class F>
void CPMA<traits>::serial_map_with_hint(
    F f, key_type end_key, const typename leaf::iterator &hint) const {
  // TODO(wheatman) handle the zero element

  uint64_t leaf_number = get_leaf_number_from_leaf_iterator(hint);
  iterator it(hint, leaf_number + 1, *this);
  for (; it != end(); ++it) {
    if (*it >= end_key) {
      return;
    }
    if constexpr (no_early_exit) {
      f(*it);
    } else {
      if (f(*it)) {
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
  // TODO(wheatman) handle the zero element

  uint64_t leaf_number = get_leaf_number_from_leaf_iterator(hint);
  uint64_t leaf_number_end = get_leaf_number_from_leaf_iterator(end_hint);

  // do the first leaf
  for (auto it = hint; it != typename leaf::iterator_end(); ++it) {
    if (*it >= end_key) {
      return;
    }
    if constexpr (no_early_exit) {
      f(*it);
    } else {
      if (f(*it)) {
        return;
      }
    }
  }
  if (leaf_number == leaf_number_end) {
    return;
  }
  leaf_number += 1;
  // if there is an leaves in the middle do them in parallel
  if (leaf_number < leaf_number_end) {
    if (leaf_number_end - leaf_number > 50) {
      ParallelTools::parallel_for(
          leaf_number, leaf_number_end,
          [&](uint64_t idx) { get_leaf(idx).template map<no_early_exit, F>(f); }

      );
    } else {
      for (size_t idx = leaf_number; idx < leaf_number_end; idx++) {
        get_leaf(idx).template map<no_early_exit, F>(f);
      }
    }
  }

  // do the last leaf

  for (auto el : get_leaf(leaf_number_end)) {
    if (el >= end_key) {
      break;
    }
    if constexpr (no_early_exit) {
      f(el);
    } else {
      if (f(el)) {
        break;
      }
    }
  }
}

template <typename traits>
typename CPMA<traits>::key_type CPMA<traits>::max() const {
  return get_leaf(total_leaves() - 1).last();
}
template <typename traits>
typename CPMA<traits>::key_type CPMA<traits>::min() const {
  if (has_0) {
    return 0;
  }
  return index_to_head_key(0);
}
template <typename traits> uint32_t CPMA<traits>::num_nodes() const {
  return (max() >> 32) + 1;
}

template <typename traits>
uint64_t
CPMA<traits>::build_rank_array_recursive(uint64_t start, uint64_t end,
                                         uint64_t rank_array_index,
                                         uint64_t *correct_array) const {
  static_assert(support_rank);
  if (start >= total_leaves()) {
    return 0;
  }
  if (end - start == 1) {
    return get_leaf(start).element_count();
  }
  uint64_t mid_point = start + (end - start) / 2;
  uint64_t left_count = build_rank_array_recursive(
      start, mid_point, rank_array_index * 2 + 1, correct_array);

  correct_array[rank_array_index] = left_count;

  uint64_t right_count = build_rank_array_recursive(
      mid_point, end, rank_array_index * 2 + 2, correct_array);

  return left_count + right_count;
}

template <typename traits> bool CPMA<traits>::check_rank_array() const {
  if constexpr (support_rank) {
    uint64_t start = 0;
    uint64_t end = nextPowerOf2(total_leaves());
    uint64_t *correct_array =
        (uint64_t *)malloc(nextPowerOf2(total_leaves()) * sizeof(uint64_t));
    std::fill(correct_array, &correct_array[nextPowerOf2(total_leaves())], 0);

    build_rank_array_recursive(start, end, 0, correct_array);
    bool ret = true;

    for (uint64_t i = 0; i < end - 1; i++) {
      if (correct_array[i] != rank_tree_array()[i]) {
        printf("wrong item in rank tree array in index %lu, got %lu, "
               "expected %lu\n",
               i, rank_tree_array()[i], correct_array[i]);
        ret = false;
      }
    }
    if (!ret) {
      print_pma();
    }
    free(correct_array);
    return ret;
  } else {
    return true;
  }
}

template <typename traits>
std::unique_ptr<uint64_t, free_delete>
CPMA<traits>::getDegreeVector(typename leaf::iterator *hints) const {
  std::unique_ptr<uint64_t, free_delete> degrees =
      std::unique_ptr<uint64_t, free_delete>(
          (uint64_t *)malloc(sizeof(uint64_t) * (num_nodes())));

  ParallelTools::For<parallel>(0, num_nodes(), [&](size_t i) {
    degrees.get()[i] = 0;
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

  ParallelTools::For<parallel>(0, num_nodes(), [&](size_t i) {
    uintptr_t ptr_length = (uint8_t *)hints[i + 1].get_pointer() -
                           (uint8_t *)hints[i].get_pointer();
    degrees.get()[i] = (double)ptr_length / average_element_size;
  });

  return degrees;
}

template <typename traits>
typename CPMA<traits>::key_type CPMA<traits>::split(CPMA<traits> *right) {
  uint64_t total_elements = get_element_count();
  uint64_t mid_point = total_elements / 2;
  SOA_type soa(total_elements - mid_point);
  key_type last_in_left_key;
  uint64_t i = 0;
#pragma clang loop unroll_count(4)
  for (auto el : *this) {
    if (i == mid_point - 1) {
      if constexpr (binary) {
        last_in_left_key = el;
      } else {
        last_in_left_key = std::get<0>(el);
      }
    }
    if (i >= mid_point) {
      soa.get(i - mid_point) = el;
    }
    i++;
  }
  remove_batch(soa.template get_ptr<0>(0), total_elements - mid_point);
  new (right) CPMA<traits>();
  right->insert_batch(soa.get_ptr(0), total_elements - mid_point);
  return last_in_left_key;
}

// extra functions to help with PCSR
template <typename traits>
CPMA<traits>::CPMA([[maybe_unused]] make_pcsr tag, size_t num_nodes) {
  static_assert(traits::maintain_offsets);

  static_assert(!traits::compressed);
  typename traits::SOA_type leaf_init(num_nodes);
  ParallelTools::parallel_for(0, num_nodes, [&](size_t i) {
    // the values for the sentinals are zero initialized
    leaf_init.template get_ptr(i).zero();
    std::get<0>(leaf_init.template get<0>(i)) = i | pcsr_top_bit;
  });
  typename traits::leaf leaf(leaf_init.get(0), leaf_init.get_ptr(1),
                             num_nodes * sizeof(key_type));

  uint64_t grow_times = 0;
  auto bytes_occupied = (num_nodes + 1) * sizeof(key_type);

  // min bytes necessary to meet the density bound
  // uint64_t bytes_required = bytes_occupied / upper_density_bound(0);
  uint64_t bytes_required =
      std::max(N() * growing_factor, bytes_occupied * growing_factor);

  while (meta_data[grow_times].n <= bytes_required) {
    grow_times += 1;
  }
  meta_data_index = grow_times;
  has_0 = false;
  count_elements_ = num_nodes;

  // steal an extra few bytes to ensure we never read off the end
  uint64_t allocated_size = underlying_array_size() + 32;
  if (allocated_size % 32 != 0) {
    allocated_size += 32 - (allocated_size % 32);
  }
  underlying_array = aligned_alloc(32, allocated_size);

  if constexpr (head_form != InPlace) {
    ParallelTools::parallel_for(0, (head_array_size() / sizeof(key_type)),
                                [&](size_t i) { head_array()[i] = 0; });
  }
  offsets_array.locations =
      (void **)malloc((num_nodes + 1) * sizeof(*(offsets_array.locations)));
  offsets_array.size = num_nodes + 1;
  offsets_array.locations[num_nodes] = data_array() + soa_num_spots();

  offsets_array.degrees =
      (key_type *)malloc((num_nodes + 1) * sizeof(*(offsets_array.degrees)));
  ParallelTools::parallel_for(0, num_nodes,
                              [&](size_t i) { offsets_array.degrees[i] = 0; });

  leaf.template split<head_form == InPlace, store_density, support_rank,
                      parallel, traits::maintain_offsets>(
      total_leaves(), count_elements_, logN(), get_data_ptr(0), 0,
      [this](uint64_t index) -> element_ref_type {
        return index_to_head(index);
      },
      density_array(), rank_tree_array(), total_leaves(), offsets_array);
#if DEBUG == 1
  if constexpr (!compressed) {
    for (size_t i = 0; i < num_nodes; i++) {
      void *loc = offsets_array.locations[i];
      key_type *loc2 = reinterpret_cast<key_type *>(loc);
      assert((*loc2) == (pcsr_top_bit | i));
    }
  }
#endif

  assert(verify_pcsr_nodes());
}

// the edges should be sorted
template <typename traits>
CPMA<traits>::CPMA([[maybe_unused]] make_pcsr tag, size_t num_nodes,
                   const auto &edges) {
  static_assert(traits::maintain_offsets);

  static_assert(!traits::compressed);
  void *leaf_init = nullptr;
  size_t soa_num_elements;
  if constexpr (binary) {
    auto elements = parlay::unique(edges);
    soa_num_elements = elements.size() + num_nodes;
    leaf_init = malloc(SOA_type::get_size_static(soa_num_elements));
    auto first_src = std::get<0>(elements[0]);
    // first the sentinals before the first element so the rest can be done in
    // parallel
    ParallelTools::parallel_for(0, first_src + 1, [&](size_t i) {
      SOA_type::get_static_ptr(leaf_init, soa_num_elements, i).zero();
      std::get<0>(SOA_type::get_static(leaf_init, soa_num_elements, i)) =
          i | pcsr_top_bit;
    });
    // deal with the first element
    SOA_type::get_static(leaf_init, soa_num_elements, first_src + 1) =
        leftshift_tuple(elements[0]);

    ParallelTools::parallel_for(1, elements.size(), [&](size_t i) {
      auto src = std::get<0>(elements[i]);
      auto prev_src = std::get<0>(elements[i - 1]);
      if (src != prev_src) {
        assert(prev_src < src);
        // add the sentinals that need to be added
        ParallelTools::parallel_for(prev_src + 1, src + 1, [&](size_t j) {
          SOA_type::get_static_ptr(leaf_init, soa_num_elements, i + j).zero();
          std::get<0>(SOA_type::get_static(leaf_init, soa_num_elements,
                                           i + j)) = j | pcsr_top_bit;
        });
      }
      SOA_type::get_static(leaf_init, soa_num_elements, i + src + 1) =
          leftshift_tuple(elements[i]);
    });
  } else {
    auto offsets = parlay::filter(
        parlay::delayed_tabulate(
            edges.size(),
            [&edges](size_t i) {
              if (i == 0 || std::make_pair(std::get<0>(edges[i]),
                                           std::get<1>(edges[i])) !=
                                std::make_pair(std::get<0>(edges[i - 1]),
                                               std::get<1>(edges[i - 1]))) {
                return i;
              } else {
                return std::numeric_limits<size_t>::max();
              }
            }),
        [](size_t i) { return i < std::numeric_limits<size_t>::max(); });
    offsets.push_back(edges.size());
    auto element_srcs =
        parlay::delayed_tabulate(offsets.size() - 1, [&](size_t i) {
          size_t start = offsets[i];
          auto src = std::get<0>(edges[start]);
          return src;
        });
    auto element_values =
        parlay::delayed_tabulate(offsets.size() - 1, [&](size_t i) {
          size_t start = offsets[i];
          size_t end = offsets[i + 1];
          element_type base = leftshift_tuple(edges[start]);
          element_ref_type base_ref = MakeTupleRef(base);
          for (size_t j = start + 1; j < end; j++) {
            value_update()(base_ref, leftshift_tuple(edges[j]));
          }
          return base;
        });

    soa_num_elements = element_srcs.size() + num_nodes;
    leaf_init = malloc(SOA_type::get_size_static(soa_num_elements));
    auto first_src = element_srcs[0];
    // first the sentinals before the first element so the rest can be done in
    // parallel
    ParallelTools::parallel_for(0, first_src + 1, [&](size_t i) {
      SOA_type::get_static_ptr(leaf_init, soa_num_elements, i).zero();
      std::get<0>(SOA_type::get_static(leaf_init, soa_num_elements, i)) =
          i | pcsr_top_bit;
    });
    // deal with the first element
    auto l_value = element_values[0];
    SOA_type::get_static_ptr(leaf_init, soa_num_elements, first_src + 1)
        .set_and_zero(MakeTupleRef(l_value));

    ParallelTools::parallel_for(1, element_srcs.size(), [&](size_t i) {
      auto src = element_srcs[i];
      auto prev_src = element_srcs[i - 1];
      if (src != prev_src) {
        assert(prev_src < src);
        // add the sentinals that need to be added
        ParallelTools::parallel_for(prev_src + 1, src + 1, [&](size_t j) {
          SOA_type::get_static_ptr(leaf_init, soa_num_elements, i + j).zero();
          std::get<0>(SOA_type::get_static(leaf_init, soa_num_elements,
                                           i + j)) = j | pcsr_top_bit;
        });
      }
      auto l_value = element_values[i];
      SOA_type::get_static_ptr(leaf_init, soa_num_elements, i + src + 1)
          .set_and_zero(MakeTupleRef(l_value));
    });
  }

  auto leaf_soa = SOA_type(leaf_init, soa_num_elements);

  typename traits::leaf leaf(leaf_soa.get(0), leaf_soa.get_ptr(1),
                             soa_num_elements * sizeof(key_type));

  uint64_t grow_times = 0;
  auto bytes_occupied = (soa_num_elements + 1) * sizeof(key_type);

  // min bytes necessary to meet the density bound
  // uint64_t bytes_required = bytes_occupied / upper_density_bound(0);
  uint64_t bytes_required =
      std::max(N() * growing_factor, bytes_occupied * growing_factor);

  while (meta_data[grow_times].n <= bytes_required) {
    grow_times += 1;
  }
  meta_data_index = grow_times;
  has_0 = false;
  count_elements_ = soa_num_elements;

  // steal an extra few bytes to ensure we never read off the end
  uint64_t allocated_size = underlying_array_size() + 32;
  if (allocated_size % 32 != 0) {
    allocated_size += 32 - (allocated_size % 32);
  }
  underlying_array = aligned_alloc(32, allocated_size);

  if constexpr (head_form != InPlace) {
    ParallelTools::parallel_for(0, (head_array_size() / sizeof(key_type)),
                                [&](size_t i) { head_array()[i] = 0; });
  }
  offsets_array.locations =
      (void **)malloc((num_nodes + 1) * sizeof(*(offsets_array.locations)));
  offsets_array.size = num_nodes + 1;
  offsets_array.locations[num_nodes] = data_array() + soa_num_spots();

  leaf.template split<head_form == InPlace, store_density, support_rank,
                      parallel, traits::maintain_offsets>(
      total_leaves(), count_elements_, logN(), get_data_ptr(0), 0,
      [this](uint64_t index) -> element_ref_type {
        return index_to_head(index);
      },
      density_array(), rank_tree_array(), total_leaves(), offsets_array);
  assert(verify_pcsr_nodes());

  offsets_array.degrees =
      (key_type *)malloc((num_nodes + 1) * sizeof(*(offsets_array.degrees)));
  ParallelTools::parallel_for(0, num_nodes, [&](size_t i) {
    key_type counted_degree = 0;
    map_neighbors_pcsr<true, false>(
        i, [&]([[maybe_unused]] const auto &arg1,
               [[maybe_unused]] const auto &arg2) { counted_degree += 1; });
    offsets_array.degrees[i] = counted_degree;
  });

#if DEBUG == 1
  if constexpr (!compressed) {
    for (size_t i = 0; i < num_nodes; i++) {
      void *loc = offsets_array.locations[i];
      key_type *loc2 = reinterpret_cast<key_type *>(loc);
      assert((*loc2) == (pcsr_top_bit | i));
    }
  }
#endif

  assert(verify_pcsr_degrees());
}
template <typename traits>
std::pair<uint64_t, bool>
CPMA<traits>::get_info_from_raw_pointer(void *ptr) const {
  assert(ptr != nullptr);
  if constexpr (head_form != InPlace) {
    // this would mean that the start of a region lines up with a leaf head
    if (ptr < key_array()) {
      if constexpr (head_form == Linear) {
        size_t leaf_number = reinterpret_cast<key_type *>(ptr) - head_array();

        ASSERT(leaf_number <= total_leaves(),
               "leaf_number = %lu,  total_leaves() = %lu\n", leaf_number,
               total_leaves());
        return {leaf_number, true};
      } else if constexpr (head_form == Eytzinger) {
        size_t leaf_number_in_eyt_order =
            reinterpret_cast<key_type *>(ptr) - head_array();
        size_t leaf_number =
            e_index_reverse(leaf_number_in_eyt_order, total_leaves());
        ASSERT(leaf_number <= total_leaves(),
               "leaf_number = %lu,  total_leaves() = %lu\n", leaf_number,
               total_leaves());
        return {leaf_number, true};
      }
      static_assert(head_form != BNary);
      // TODO(wheatman) deal with getting back the leaf number when it is stored
      // in bnary order
    } else {
      // the region points somewhere in the middle of a leaf
      uintptr_t bytes_from_start =
          reinterpret_cast<uint8_t *>(ptr) - ((uint8_t *)key_array());
      ASSERT(bytes_from_start / logN() <= total_leaves(),
             "bytes_from_start / logN() = %lu, total_leaves() = %lu\n",
             bytes_from_start / logN(), total_leaves());
      return {bytes_from_start / logN(), false};
    }
  } else {
    // head_form == InPlace
    // this was we don't need to worry about the heads being somewhere else
    uintptr_t bytes_from_start =
        reinterpret_cast<uint8_t *>(ptr) - ((uint8_t *)key_array());
    bool is_head = (bytes_from_start % logN()) == 0;
    assert(bytes_from_start / logN() <= total_leaves());
    return {bytes_from_start / logN(), is_head};
  }
}

template <typename traits>
bool CPMA<traits>::insert_pcsr(typename traits::key_type src,
                               typename traits::key_type dest, value_type val) {
  element_type element = std::tuple_cat(std::tuple(dest), val);
  static_assert(traits::maintain_offsets);
  auto start_region = offsets_array.locations[src];
  auto end_region = offsets_array.locations[src + 1];

  const auto &[start_leaf, start_is_head] =
      get_info_from_raw_pointer(start_region);
  assert(start_leaf < total_leaves());

  const auto &[end_leaf, end_is_head] = get_info_from_raw_pointer(end_region);
  assert(end_leaf <= total_leaves());

  // we don't need to worry much about the end, since we always read in order
  // and sentinals are always bigger than data
  size_t end_index = end_leaf * elts_per_leaf();
  if (!end_is_head) {
    // get to the next leaf
    end_index += elts_per_leaf();
  }

  // // this is the simple case, we just need to find the search range and do
  // the
  // // normal thing
  // if (start_is_head) {
  //   uint64_t leaf_number = find_containing_leaf_number(
  //       dest, start_leaf * elts_per_leaf(), end_index);
  //   auto [inserted, byte_count] =
  //       get_leaf(leaf_number)
  //           .template insert<head_form == InPlace, value_update>(dest);
  //   if (!inserted) {
  //     return false;
  //   }
  //   insert_post_place(leaf_number, byte_count);
  //   return true;
  // }
  // the start is not the head so we need to deal with a partial leaf
  // first see if we are even going into the second leaf
  if ((end_leaf > start_leaf + 1 || (end_leaf > start_leaf && !end_is_head)) &&
      start_leaf + 1 < total_leaves()) {
    // if we are going in to a second leaf, then check if we will be in that
    // portion and can just insert like normal into the second portion
    if (index_to_head_key(start_leaf + 1) <= dest) {
      uint64_t leaf_number = find_containing_leaf_number(
          dest, (start_leaf + 1) * elts_per_leaf(), end_index);
      assert(leaf_number < total_leaves());
      auto [inserted, byte_count] =
          get_leaf(leaf_number)
              .template insert<head_form == InPlace, value_update,
                               traits::maintain_offsets>(
                  element, value_update(), offsets_array);
      if (!inserted) {
        return false;
      }
      offsets_array.degrees[src] += 1;
      insert_post_place(leaf_number, byte_count);
      return true;
    }
  }
  // std::cout << "inserting into the first leaf\n";
  // we are in the first partial leaf
  // just make a fake leaf and insert into it
  static_assert(!traits::compressed);

  if (start_is_head) {

    typename traits::leaf l(*index_to_data(start_leaf),
                            index_to_data(start_leaf) + 1,
                            leaf_size_in_bytes() - sizeof(key_type));
    auto [inserted, bad_byte_count] =
        l.template insert<head_form == InPlace, value_update,
                          traits::maintain_offsets>(element, value_update(),
                                                    offsets_array);
    if (!inserted) {
      return false;
    }
    offsets_array.degrees[src] += 1;
    insert_post_place(
        start_leaf,
        get_leaf(start_leaf).template used_size<head_form == InPlace>());
    return true;

  } else {
    size_t elements_from_start =
        reinterpret_cast<key_type *>(start_region) - key_array();
    assert(*reinterpret_cast<key_type *>(start_region) == (pcsr_top_bit | src));
    // +2 to account for moving past the sentinal and the fake head
    size_t num_into_leaf = (elements_from_start % elts_per_leaf()) + 2;

    typename traits::leaf l(get_data_ref(elements_from_start + 1),
                            get_data_ptr(elements_from_start + 2),
                            (elts_per_leaf() - num_into_leaf) *
                                sizeof(key_type));

    auto [inserted, bad_byte_count] =
        l.template insert<head_form == InPlace, value_update,
                          traits::maintain_offsets>(element, value_update(),
                                                    offsets_array);
    if (!inserted) {
      return false;
    }
    offsets_array.degrees[src] += 1;
    insert_post_place(
        start_leaf,
        get_leaf(start_leaf).template used_size<head_form == InPlace>());
    return true;
  }
}

template <typename traits>
bool CPMA<traits>::remove_pcsr(typename traits::key_type src,
                               typename traits::key_type dest) {
  static_assert(traits::maintain_offsets);
  auto start_region = offsets_array.locations[src];
  auto end_region = offsets_array.locations[src + 1];

  const auto &[start_leaf, start_is_head] =
      get_info_from_raw_pointer(start_region);
  assert(start_leaf < total_leaves());

  const auto &[end_leaf, end_is_head] = get_info_from_raw_pointer(end_region);
  assert(end_leaf <= total_leaves());

  // we don't need to worry much about the end, since we always read in order
  // and sentinals are always bigger than data
  size_t end_index = end_leaf * elts_per_leaf();
  if (!end_is_head) {
    // get to the next leaf
    end_index += elts_per_leaf();
  }

  // // this is the simple case, we just need to find the search range and do
  // the
  // // normal thing
  // if (start_is_head) {
  //   uint64_t leaf_number = find_containing_leaf_number(
  //       dest, start_leaf * elts_per_leaf(), end_index);
  //   auto [inserted, byte_count] =
  //       get_leaf(leaf_number)
  //           .template insert<head_form == InPlace, value_update>(dest);
  //   if (!inserted) {
  //     return false;
  //   }
  //   insert_post_place(leaf_number, byte_count);
  //   return true;
  // }
  // the start is not the head so we need to deal with a partial leaf
  // first see if we are even going into the second leaf
  if ((end_leaf > start_leaf + 1 || (end_leaf > start_leaf && !end_is_head)) &&
      start_leaf + 1 < total_leaves()) {
    // if we are going in to a second leaf, then check if we will be in that
    // portion and can just insert like normal into the second portion
    if (index_to_head_key(start_leaf + 1) <= dest) {
      uint64_t leaf_number = find_containing_leaf_number(
          dest, (start_leaf + 1) * elts_per_leaf(), end_index);
      assert(leaf_number < total_leaves());
      auto [removed, byte_count] =
          get_leaf(leaf_number)
              .template remove<head_form == InPlace, traits::maintain_offsets>(
                  dest, offsets_array);
      if (!removed) {
        return false;
      }
      offsets_array.degrees[src] -= 1;
      remove_post_place(leaf_number, byte_count);
      return true;
    }
  }
  // std::cout << "inserting into the first leaf\n";
  // we are in the first partial leaf
  // just make a fake leaf and insert into it
  static_assert(!traits::compressed);

  if (start_is_head) {

    typename traits::leaf l(*index_to_data(start_leaf),
                            index_to_data(start_leaf) + 1,
                            leaf_size_in_bytes() - sizeof(key_type));
    auto [removed, bad_byte_count] =
        l.template remove<head_form == InPlace, traits::maintain_offsets>(
            dest, offsets_array);
    if (!removed) {
      return false;
    }
    offsets_array.degrees[src] -= 1;
    remove_post_place(
        start_leaf,
        get_leaf(start_leaf).template used_size<head_form == InPlace>());
    return true;

  } else {
    size_t elements_from_start =
        reinterpret_cast<key_type *>(start_region) - key_array();
    // +2 to account for moving past the sentinal and the fake head
    size_t num_into_leaf = (elements_from_start % elts_per_leaf()) + 2;

    typename traits::leaf l(get_data_ref(elements_from_start + 1),
                            get_data_ptr(elements_from_start + 2),
                            (elts_per_leaf() - num_into_leaf) *
                                sizeof(key_type));

    auto [removed, bad_byte_count] =
        l.template remove<head_form == InPlace, traits::maintain_offsets>(
            dest, offsets_array);
    if (!removed) {
      return false;
    }
    offsets_array.degrees[src] -= 1;
    remove_post_place(
        start_leaf,
        get_leaf(start_leaf).template used_size<head_form == InPlace>());
    return true;
  }
}

template <typename traits>
bool CPMA<traits>::contains_pcsr(typename traits::key_type src,
                                 typename traits::key_type dest) const {
  static_assert(traits::maintain_offsets);
  auto start_region = offsets_array.locations[src];
  auto end_region = offsets_array.locations[src + 1];

  const auto &[start_leaf, start_is_head] =
      get_info_from_raw_pointer(start_region);
  assert(start_leaf < total_leaves());

  const auto &[end_leaf, end_is_head] = get_info_from_raw_pointer(end_region);
  assert(end_leaf <= total_leaves());

  // we don't need to worry much about the end, since we always read in order
  // and sentinals are always bigger than data
  size_t end_index = end_leaf * elts_per_leaf();
  if (!end_is_head) {
    // get to the next leaf
    end_index += elts_per_leaf();
  }

  // the start is not the head so we need to deal with a partial leaf
  // first see if we are even going into the second leaf
  if ((end_leaf > start_leaf + 1 || (end_leaf > start_leaf && !end_is_head)) &&
      start_leaf + 1 < total_leaves()) {
    // if we are going in to a second leaf, then check if we will be in that
    // portion and can just insert like normal into the second portion
    if (index_to_head_key(start_leaf + 1) <= dest) {
      uint64_t leaf_number = find_containing_leaf_number(
          dest, (start_leaf + 1) * elts_per_leaf(), end_index);
      assert(leaf_number < total_leaves());
      return get_leaf(leaf_number)
          .template contains<head_form == InPlace>(dest);
    }
  }
  // std::cout << "inserting into the first leaf\n";
  // we are in the first partial leaf
  // just make a fake leaf and insert into it
  static_assert(!traits::compressed);

  if (start_is_head) {

    typename traits::leaf l(*index_to_data(start_leaf),
                            index_to_data(start_leaf) + 1,
                            leaf_size_in_bytes() - sizeof(key_type));
    return l.template contains<head_form == InPlace>(dest);
  } else {
    size_t elements_from_start =
        reinterpret_cast<key_type *>(start_region) - key_array();
    // +2 to account for moving past the sentinal and the fake head
    size_t num_into_leaf = (elements_from_start % elts_per_leaf()) + 2;

    typename traits::leaf l(get_data_ref(elements_from_start + 1),
                            get_data_ptr(elements_from_start + 2),
                            (elts_per_leaf() - num_into_leaf) *
                                sizeof(key_type));

    return l.template contains<head_form == InPlace>(dest);
  }
}

template <typename traits>
template <bool no_early_exit, bool parallel, class F>
void CPMA<traits>::map_neighbors_pcsr(key_type node, F f) const {
  auto start_region = offsets_array.locations[node];
  auto end_region = offsets_array.locations[node + 1];
  const auto &[start_leaf, start_is_head] =
      get_info_from_raw_pointer(start_region);

  const auto &[end_leaf, end_is_head] = get_info_from_raw_pointer(end_region);
  if (end_is_head) {
    // reset the end rehion into the data array, since we never actually look at
    // it and it makes the math easier
    if constexpr (head_form == InPlace) {
      end_region = index_to_data(end_leaf).get_pointer() - 1;
    } else {
      end_region = index_to_data(end_leaf).get_pointer();
    }
  }

  // do the first leaf

  bool single_leaf = start_leaf == end_leaf;

  auto f2 = [&](const auto &el) {
    ASSERT(element_or_first_element(el) <= num_nodes_pcsr(),
           "std::get<0>(el) = %lu,  num_nodes_pcsr() = %lu\n",
           static_cast<uint64_t>(element_or_first_element(el)),
           static_cast<uint64_t>(num_nodes_pcsr()));
    return f(node, el);
  };

  if (start_is_head) {
    if (reinterpret_cast<uint8_t *>(index_to_data(start_leaf).get_pointer()) <
        static_cast<uint8_t *>(end_region)) {
      uint64_t leaf_size = static_cast<uint8_t *>(end_region) -
                           reinterpret_cast<uint8_t *>(
                               index_to_data(start_leaf).get_pointer() + 1);
      if (!single_leaf) {
        leaf_size = leaf_size_in_bytes() - sizeof(key_type);
      }
      assert(leaf_size < leaf_size_in_bytes());

      typename traits::leaf first_leaf = leaf(
          *index_to_data(start_leaf), index_to_data(start_leaf) + 1, leaf_size);
      if (first_leaf.template map<no_early_exit>(f2)) {
        return;
      }
    }

  } else {
    if (reinterpret_cast<key_type *>(start_region) + 1 < end_region) {
      size_t elements_from_start =
          reinterpret_cast<key_type *>(start_region) - key_array();
      // +2 to account for moving past the sentinal and the fake head
      size_t num_into_leaf = (elements_from_start % elts_per_leaf()) + 2;

      uint64_t leaf_size = static_cast<uint8_t *>(end_region) -
                           reinterpret_cast<uint8_t *>(
                               reinterpret_cast<key_type *>(start_region) + 2);
      if (!single_leaf) {
        leaf_size = (elts_per_leaf() - num_into_leaf) * sizeof(key_type);
      }

      typename traits::leaf first_leaf =
          leaf(get_data_ref(elements_from_start + 1),
               get_data_ptr(elements_from_start + 2), leaf_size);
      if (first_leaf.template map<no_early_exit>(f2)) {
        return;
      }
    }
  }

  size_t start_leaf_idx = start_leaf + 1;
  // if there is an leaves in the middle do them in parallel
  if (start_leaf_idx < end_leaf) {
    if constexpr (parallel) {
      if (end_leaf - start_leaf_idx > 50) {
        // don't bother to early exit in parallel
        ParallelTools::parallel_for(
            start_leaf_idx, end_leaf, [&](uint64_t idx) {
              get_leaf(idx).template map<no_early_exit>(f2);
            });
      } else {
        for (size_t idx = start_leaf_idx; idx < end_leaf; idx++) {
          if (get_leaf(idx).template map<no_early_exit>(f2)) {
            return;
          }
        }
      }
    } else {
      for (size_t idx = start_leaf_idx; idx < end_leaf; idx++) {
        if (get_leaf(idx).template map<no_early_exit>(f2)) {
          return;
        }
      }
    }
  }
  // do the last leaf
  if (!end_is_head && !single_leaf && end_leaf < total_leaves()) {
    typename traits::leaf last_leaf(
        index_to_head(end_leaf), index_to_data(end_leaf),
        static_cast<uint8_t *>(end_region) -
            reinterpret_cast<uint8_t *>(index_to_data(end_leaf).get_pointer()));
    last_leaf.template map<no_early_exit>(f2);
  } else {
  }
}

template <typename traits>
template <std::ranges::random_access_range R, class Vector_pairs>
uint64_t CPMA<traits>::insert_batch_internal_pcsr(
    R &es, std::invocable<typename std::ranges::range_value_t<R>> auto &&f,
    Vector_pairs &leaves_to_check, const uint64_t start_leaf_idx,
    const uint64_t end_leaf_idx) {
  // std::cout << "start_leaf_number = " << start_leaf_idx / elts_per_leaf()
  //           << " end_leaf_number = " << end_leaf_idx / elts_per_leaf() <<
  //           "\n";

  uint64_t batch_size = es.size();
  // not technically true, but true in practice
  assert(batch_size < 1000000000000UL);
  if (batch_size == 0) {
    return 0;
  }

#if DEBUG == 1
  {
    auto element = es[0];
    auto &element_src = std::get<0>(element);
    auto start_region = offsets_array.locations[element_src];
    auto end_region = offsets_array.locations[element_src + 1];
    const auto &[start_leaf, start_is_head] =
        get_info_from_raw_pointer(start_region);
    const auto &[end_leaf, end_is_head] = get_info_from_raw_pointer(end_region);
    assert(std::max(start_leaf, end_leaf) >= start_leaf_idx / elts_per_leaf());
    assert(std::min(start_leaf, end_leaf) <= end_leaf_idx / elts_per_leaf());
  }
  {
    auto element = es[es.size() - 1];
    auto &element_src = std::get<0>(element);
    auto start_region = offsets_array.locations[element_src];
    auto end_region = offsets_array.locations[element_src + 1];
    const auto &[start_leaf, start_is_head] =
        get_info_from_raw_pointer(start_region);
    const auto &[end_leaf, end_is_head] = get_info_from_raw_pointer(end_region);
    assert(std::max(start_leaf, end_leaf) >= start_leaf_idx / elts_per_leaf());
    assert(std::min(start_leaf, end_leaf) <= end_leaf_idx / elts_per_leaf());
  }
#endif

  uint64_t num_elts_merged = 0;
  // TODO(wheatman) maybe special case batch small batch sizes and dense batch
  // sizes
  int64_t middle_index = es.size() / 2;
  auto middle_element = f(es[middle_index]);
  auto &middle_element_src = std::get<0>(middle_element);
  auto &middle_element_dest = std::get<1>(middle_element);
  auto start_region = offsets_array.locations[middle_element_src];
  auto end_region = offsets_array.locations[middle_element_src + 1];

  const auto &[start_leaf_saved, start_is_head_saved] =
      get_info_from_raw_pointer(start_region);
  auto start_leaf = start_leaf_saved;
  assert(start_leaf < total_leaves());

  const auto &[end_leaf_saved, end_is_head_saved] =
      get_info_from_raw_pointer(end_region);
  auto end_leaf = end_leaf_saved;
  auto end_is_head = end_is_head_saved;
  assert(end_leaf <= total_leaves());
  if (!end_is_head) {
    // get to the next leaf
    end_leaf += 1;
  }
  // if the recursion gives us something more restrictive at the end just take
  // it
  if (end_leaf_idx / elts_per_leaf() < end_leaf) {
    end_leaf = end_leaf_idx / elts_per_leaf();
    end_is_head = false;
  }

  // either we are on the first leaf and we special case it, or we are in a
  // latter leaf and it should be easy

  uint64_t leaf_number;
  auto next_head_src = middle_element_src;
  auto this_head_src = middle_element_src;

  // first check that there is a second leaf, which is either that end_leaf >=
  // start_leaf+2 so there is something in the middle, or itys just greater
  // (meaning might only be one), but in this case we also need the end not to
  // be the head, since that means we can't actually insert it there.  Next we
  // check that we are not the last leaf.  Now we know that there is another
  // leaf we can check if the head of that second leaf  is less than or equal to
  // our dest, this means we can ignore the first leaf and just do a normal case
  if ((end_leaf > start_leaf + 1) && start_leaf + 1 < total_leaves() &&
      index_to_head_key(start_leaf + 1) <= middle_element_dest) {
    start_leaf += 1;
    if (start_leaf < start_leaf_idx / elts_per_leaf()) {
      start_leaf = start_leaf_idx / elts_per_leaf();
    }
    if (start_leaf < end_leaf) {
      leaf_number = find_containing_leaf_number(middle_element_dest,
                                                start_leaf * elts_per_leaf(),
                                                end_leaf * elts_per_leaf());
    } else {
      // std::cout << "start_leaf = end_leaf\n";
      leaf_number = end_leaf;
    }

    // since we are the second leaf of the node, we can't be the first leaf
    // overall
    assert(leaf_number > 0);

    key_type head = index_to_head_key(leaf_number);
    // now we need to walk backwords in the batch to find the first element in
    // the batch that goes to this same leaf keep in mind that this cannot be
    // for a different node since we are at least in the second leaf of a node
    // in this case

    while (middle_index > 0) {

      if (head < pcsr_top_bit) {
        if (std::get<1>(f(es[middle_index - 1])) < head) {
          break;
        }
      }
      if (std::get<0>(f(es[middle_index - 1])) != middle_element_src) {
        break;
      }

      middle_index -= 1;
    }
  } else {
    // we know we are in the first leaf of a range
    leaf_number = start_leaf;
    if (leaf_number < start_leaf_idx / elts_per_leaf()) {
      leaf_number = start_leaf_idx / elts_per_leaf();
    }

    // we need to walk backwords to find the start of the leaf, dealing with
    // seninals and knowing what the new starting src is

#if DEBUG == 1
    size_t counter = 0;
#endif

    while (this_head_src > 0 &&
           std::get<0>(get_info_from_raw_pointer(
               offsets_array.locations[this_head_src])) == leaf_number &&
           get_info_from_raw_pointer(offsets_array.locations[this_head_src]) !=
               std::pair(leaf_number, true)) {
      this_head_src -= 1;
#if DEBUG == 1
      // just make sure this doesn't turn into an infinite loop, there could
      // be at most max number of elements, which could be up to num bytes
      // sentinals in this leaf
      counter += 1;
      assert(counter < logN());
#endif
    }

    // then we need to walk back in the batch to find the first element that
    // will go into this leaf now that we know the starting source
    key_type head = index_to_head_key(leaf_number);
    if (head >= pcsr_top_bit) {
#if DEBUG == 1
      int64_t debug_middle_index = middle_index;
      while (debug_middle_index > 0 &&
             std::get<0>(f(es[debug_middle_index - 1])) >= this_head_src) {
        debug_middle_index -= 1;
      }
#endif
      int64_t check_distance = 1;
      int64_t bottom_range = 0;
      int64_t top_range = middle_index;
      while (check_distance < middle_index) {
        if (std::get<0>(f(es[middle_index - check_distance])) < this_head_src) {
          bottom_range = middle_index - check_distance + 1;
          break;
        }
        top_range = middle_index - check_distance + 1;
        check_distance *= 2;
      }
      if (bottom_range == top_range) {
        middle_index = bottom_range;
      } else {
        auto it = std::lower_bound(
            es.begin() + bottom_range, es.begin() + top_range, this_head_src,
            [&](const auto &el, [[maybe_unused]] auto value) {
              return std::get<0>(f(el)) < this_head_src;
            });
        middle_index = it - es.begin();
      }
#if DEBUG == 1
      ASSERT(middle_index == debug_middle_index, "got %ld, expected %ld\n",
             middle_index, debug_middle_index);
#endif
    } else {
      while (middle_index > 0) {
        auto batch_element = f(es[middle_index - 1]);
        if (std::get<0>(batch_element) == this_head_src &&
            std::get<1>(batch_element) < head) {
          break;
        }
        if (std::get<0>(batch_element) < this_head_src) {
          break;
        }
        middle_index -= 1;
      }
    }
  }

  key_type next_head_dest =
      std::numeric_limits<key_type>::max() >> 1; // max int
  if (leaf_number + 1 < end_leaf_idx / elts_per_leaf()) {
    next_head_dest = index_to_head_key(leaf_number + 1);
    assert(next_head_dest != 0);
    // this might actually be a sentinal, but thats fine, since in that case its
    // part of the next node and the src will will us to add what we should
  }

// in this case we need to figure out what source controls the head of the
// next leaf, we can just walk forward in the offsets array to do this
#if DEBUG == 1
  size_t counter = 0;
#endif
  while (
      next_head_src < num_nodes_pcsr() &&
      (std::get<0>(get_info_from_raw_pointer(
           offsets_array.locations[next_head_src + 1])) == leaf_number ||
       get_info_from_raw_pointer(offsets_array.locations[next_head_src + 1]) ==
           std::pair(leaf_number + 1, true))) {
    next_head_src += 1;
#if DEBUG == 1
    // just make sure this soesn't turn into an infinite loop, there could
    // be at most max number of elements, which could be up to num bytes
    // sentinals in this leaf
    counter += 1;
    assert(counter < logN());
#endif
  }
  auto subrange = std::ranges::subrange(es.begin() + middle_index, es.end());
  assert(leaf_number >= start_leaf_idx / elts_per_leaf());
  auto result = get_leaf(leaf_number)
                    .template merge_into_leaf_pcsr<head_form == InPlace,
                                                   parallel, value_update,
                                                   traits::maintain_offsets>(
                        subrange, f, this_head_src, next_head_src,
                        next_head_dest, value_update(), offsets_array);

  static_assert(!support_rank);
  num_elts_merged += std::get<1>(result);
  auto bytes_used = std::get<2>(result);
  auto index_merged_up_to = middle_index + std::get<0>(result);
  assert(index_merged_up_to > middle_index);
  if (std::get<1>(result)) {
    static_assert(!store_density);
    ASSERT(bytes_used ==
               get_density_count(
                   leaf_number * elts_per_leaf() * sizeof(key_type), logN()),
           "got %lu, expected %lu\n", bytes_used,
           get_density_count(leaf_number * elts_per_leaf() * sizeof(key_type),
                             logN()));
    if (bytes_used > logN() * upper_density_bound(H())) {
      leaves_to_check.push_back({leaf_number * elts_per_leaf(), bytes_used});
    }
  }

  // do the recursion splitting the batch into the early and the late

  uint64_t ret1 = 0;
  uint64_t ret2 = 0;

  auto early_range =
      std::ranges::subrange(es.begin(), es.begin() + middle_index);
  auto late_range =
      std::ranges::subrange(es.begin() + index_merged_up_to, es.end());

  const auto early_range_start_leaf_idx = start_leaf_idx;
  const auto early_range_end_leaf_idx = leaf_number * elts_per_leaf();
  const auto late_range_start_leaf_idx = (leaf_number + 1) * elts_per_leaf();
  const auto late_range_end_leaf_idx = end_leaf_idx;

  if (early_range.size() == 0 && late_range.size() == 0) {
    return num_elts_merged;
  }
  if (early_range.size() == 0) {
    ret2 = insert_batch_internal_pcsr(late_range, f, leaves_to_check,
                                      late_range_start_leaf_idx,
                                      late_range_end_leaf_idx);
  } else if (late_range.size() == 0) {
    ret1 = insert_batch_internal_pcsr(early_range, f, leaves_to_check,
                                      early_range_start_leaf_idx,
                                      early_range_end_leaf_idx);
  } else {
    if constexpr (!parallel) {
      ret1 = insert_batch_internal_pcsr(early_range, f, leaves_to_check,
                                        early_range_start_leaf_idx,
                                        early_range_end_leaf_idx);
      ret2 = insert_batch_internal_pcsr(late_range, f, leaves_to_check,
                                        late_range_start_leaf_idx,
                                        late_range_end_leaf_idx);
    } else {
      if (early_range.size() <= 20 || late_range.size() <= 20) {
        ret1 = insert_batch_internal_pcsr(early_range, f, leaves_to_check,
                                          early_range_start_leaf_idx,
                                          early_range_end_leaf_idx);
        ret2 = insert_batch_internal_pcsr(late_range, f, leaves_to_check,
                                          late_range_start_leaf_idx,
                                          late_range_end_leaf_idx);
      } else {
        ParallelTools::par_do(
            [&]() {
              ret1 = insert_batch_internal_pcsr(early_range, f, leaves_to_check,
                                                early_range_start_leaf_idx,
                                                early_range_end_leaf_idx);
            },
            [&]() {
              ret2 = insert_batch_internal_pcsr(late_range, f, leaves_to_check,
                                                late_range_start_leaf_idx,
                                                late_range_end_leaf_idx);
            });
      }
    }
  }

  num_elts_merged += ret1;
  num_elts_merged += ret2;
  return num_elts_merged;
}

template <typename traits>
template <std::ranges::random_access_range R>
uint64_t CPMA<traits>::insert_batch_pcsr(
    R &es, std::invocable<typename std::ranges::range_value_t<R>> auto &&f,
    bool sorted) {
  assert(verify_pcsr_nodes());

  static_assert(!support_rank, "we don't support rank while doing pcsr stuff");
  uint64_t batch_size = es.size();
  if (batch_size < 100) {
    uint64_t count = 0;
    for (const auto &e : es) {
      auto elem = f(e);
      if constexpr (binary) {
        count += insert_pcsr(std::get<0>(elem), std::get<1>(elem));
      } else {
        count += insert_pcsr(std::get<0>(elem), std::get<1>(elem),
                             leftshift_tuple(leftshift_tuple(elem)));
      }
    }
    return count;
  }
  // the sentinals will always already be there
  assert(get_element_count() > 0);

  // don't need to deal with zero since all of the elements will be greater
  // than zero since we bump up the dests by 1, which is what the f function
  // does in practice

  if (!sorted) {
    parlay::integer_sort_inplace(es, [](const auto &elem) {
      return (static_cast<uint64_t>(std::get<0>(elem)) << 32U) |
             std::get<1>(elem);
    });
  }
  std::conditional_t<
      parallel, ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>>,
      std::vector<std::pair<uint64_t, uint64_t>>>
      leaves_to_check;

  uint64_t num_elts_merged = insert_batch_internal_pcsr(
      es, f, leaves_to_check, 0, total_leaves() * elts_per_leaf());

  // TODO(wheatman) maybe do this stuff in parallel with some other stuff if we
  // need more parallelism
  //  the batch has been marked with the high bit of the src marked if it was
  //  actually inserted, use this to help update the degree array
  // TODO(wheatman) this can probably be faster, but its simple and this should
  // be a resonable approach
  auto filtered =
      parlay::map_maybe(es, [](const auto &elem) -> std::optional<key_type> {
        if (std::get<0>(elem) >= pcsr_top_bit) {
          return std::get<0>(elem) ^ pcsr_top_bit;
        } else {
          return {};
        }
      });
  auto counts = parlay::histogram_by_key(filtered);
  filtered.clear();
  ParallelTools::parallel_for(0, counts.size(), [&](size_t i) {
    offsets_array.degrees[counts[i].first] += counts[i].second;
  });

  ASSERT(num_elts_merged ==
             parlay::delayed::reduce(parlay::delayed::map(
                 counts, [](const auto &elem) { return elem.second; })),
         "num_elts_removed = %lu, num_marked = %lu\n", num_elts_merged,
         parlay::delayed::reduce(parlay::delayed::map(
             counts, [](const auto &elem) { return elem.second; })));
  counts.clear();

  auto [ranges_to_redistribute_3, full_opt] = get_ranges_to_redistibute(
      leaves_to_check, num_elts_merged, [&](uint64_t level, float density) {
        return density > upper_density_bound(level);
      });
  if (full_opt.has_value()) {

    uint64_t grow_times = 0;
    auto bytes_occupied = full_opt.value();
    assert(bytes_occupied < 1UL << 60UL);

    // min bytes necessary to meet the density bound
    // uint64_t bytes_required = bytes_occupied / upper_density_bound(0);
    uint64_t bytes_required =
        std::max(N() * growing_factor, bytes_occupied * growing_factor);

    while (meta_data[meta_data_index + grow_times].n <= bytes_required) {
      grow_times += 1;
    }

    grow_list(grow_times);
  } else { // not doubling
    // in parallel, redistribute ranges
    redistribute_ranges(ranges_to_redistribute_3);
  }
  count_elements_ += num_elts_merged;
  assert(verify_pcsr_nodes());
  assert(verify_pcsr_degrees());
#if DEBUG == 1
  size_t edge_count = 0;
  for (size_t i = 0; i < num_nodes_pcsr(); i++) {
    map_neighbors_pcsr<true, false>(
        i, [&edge_count]([[maybe_unused]] auto arg1,
                         [[maybe_unused]] auto arg2) { edge_count += 1; });
  }
  ASSERT(edge_count == num_edges_pcsr(),
         "edge_count = %lu, num_edges_pcsr() = %lu\n", edge_count,
         num_edges_pcsr());
#endif
  return num_elts_merged;
}

template <typename traits>
template <std::ranges::random_access_range R, class Vector_pairs>
uint64_t CPMA<traits>::remove_batch_internal_pcsr(
    R &es, std::invocable<typename std::ranges::range_value_t<R>> auto &&f,
    Vector_pairs &leaves_to_check, const uint64_t start_leaf_idx,
    const uint64_t end_leaf_idx) {
  if (start_leaf_idx == end_leaf_idx) {
    return 0;
  }
  // std::cout << "start_leaf_number = " << start_leaf_idx / elts_per_leaf()
  //           << " end_leaf_number = " << end_leaf_idx / elts_per_leaf() <<
  //           "\n";

  uint64_t batch_size = es.size();
  // not technically true, but true in practice
  assert(batch_size < 1000000000000UL);
  if (batch_size == 0) {
    return 0;
  }

#if DEBUG == 1
  {
    auto element = es[0];
    auto &element_src = std::get<0>(element);
    auto start_region = offsets_array.locations[element_src];
    auto end_region = offsets_array.locations[element_src + 1];
    const auto &[start_leaf, start_is_head] =
        get_info_from_raw_pointer(start_region);
    const auto &[end_leaf, end_is_head] = get_info_from_raw_pointer(end_region);
    assert(std::max(start_leaf, end_leaf) >= start_leaf_idx / elts_per_leaf());
    assert(std::min(start_leaf, end_leaf) <= end_leaf_idx / elts_per_leaf());
  }
  {
    auto element = es[es.size() - 1];
    auto &element_src = std::get<0>(element);
    auto start_region = offsets_array.locations[element_src];
    auto end_region = offsets_array.locations[element_src + 1];
    const auto &[start_leaf, start_is_head] =
        get_info_from_raw_pointer(start_region);
    const auto &[end_leaf, end_is_head] = get_info_from_raw_pointer(end_region);
    assert(std::max(start_leaf, end_leaf) >= start_leaf_idx / elts_per_leaf());
    assert(std::min(start_leaf, end_leaf) <= end_leaf_idx / elts_per_leaf());
  }
#endif

  uint64_t num_elts_removed = 0;
  // TODO(wheatman) maybe special case batch small batch sizes and dense batch
  // sizes
  int64_t middle_index = es.size() / 2;
  auto middle_element = f(es[middle_index]);
  auto &middle_element_src = std::get<0>(middle_element);
  auto &middle_element_dest = std::get<1>(middle_element);
  auto start_region = offsets_array.locations[middle_element_src];
  auto end_region = offsets_array.locations[middle_element_src + 1];

  const auto &[start_leaf_saved, start_is_head_saved] =
      get_info_from_raw_pointer(start_region);
  auto start_leaf = start_leaf_saved;
  assert(start_leaf < total_leaves());

  const auto &[end_leaf_saved, end_is_head_saved] =
      get_info_from_raw_pointer(end_region);
  auto end_leaf = end_leaf_saved;
  auto end_is_head = end_is_head_saved;
  assert(end_leaf <= total_leaves());
  if (!end_is_head) {
    // get to the next leaf
    end_leaf += 1;
  }
  // if the recursion gives us something more restrictive at the end just take
  // it
  if (end_leaf_idx / elts_per_leaf() < end_leaf) {
    end_leaf = end_leaf_idx / elts_per_leaf();
    end_is_head = false;
  }

  // either we are on the first leaf and we special case it, or we are in a
  // latter leaf and it should be easy

  uint64_t leaf_number;
  auto next_head_src = middle_element_src;
  auto this_head_src = middle_element_src;

  // first check that there is a second leaf, which is either that end_leaf >=
  // start_leaf+2 so there is something in the middle, or itys just greater
  // (meaning might only be one), but in this case we also need the end not to
  // be the head, since that means we can't actually insert it there.  Next we
  // check that we are not the last leaf.  Now we know that there is another
  // leaf we can check if the head of that second leaf  is less than or equal to
  // our dest, this means we can ignore the first leaf and just do a normal case
  if ((end_leaf > start_leaf + 1) && start_leaf + 1 < total_leaves() &&
      index_to_head_key(start_leaf + 1) <= middle_element_dest) {
    start_leaf += 1;
    if (start_leaf < start_leaf_idx / elts_per_leaf()) {
      start_leaf = start_leaf_idx / elts_per_leaf();
    }
    if (start_leaf < end_leaf) {
      leaf_number = find_containing_leaf_number(middle_element_dest,
                                                start_leaf * elts_per_leaf(),
                                                end_leaf * elts_per_leaf());
    } else {
      // std::cout << "start_leaf = end_leaf\n";
      leaf_number = end_leaf;
    }

    // since we are the second leaf of the node, we can't be the first leaf
    // overall
    assert(leaf_number > 0);

    key_type head = index_to_head_key(leaf_number);
    // now we need to walk backwords in the batch to find the first element in
    // the batch that goes to this same leaf keep in mind that this cannot be
    // for a different node since we are at least in the second leaf of a node
    // in this case

    while (middle_index > 0) {

      if (head < pcsr_top_bit) {
        if (std::get<1>(f(es[middle_index - 1])) < head) {
          break;
        }
      }
      if (std::get<0>(f(es[middle_index - 1])) != middle_element_src) {
        break;
      }

      middle_index -= 1;
    }
  } else {
    // we know we are in the first leaf of a range
    leaf_number = start_leaf;
    if (leaf_number < start_leaf_idx / elts_per_leaf()) {
      leaf_number = start_leaf_idx / elts_per_leaf();
    }

    // we need to walk backwords to find the start of the leaf, dealing with
    // seninals and knowing what the new starting src is

#if DEBUG == 1
    size_t counter = 0;
#endif

    while (this_head_src > 0 &&
           std::get<0>(get_info_from_raw_pointer(
               offsets_array.locations[this_head_src])) == leaf_number &&
           get_info_from_raw_pointer(offsets_array.locations[this_head_src]) !=
               std::pair(leaf_number, true)) {
      this_head_src -= 1;
#if DEBUG == 1
      // just make sure this soesn't turn into an infinite loop, there could
      // be at most max number of elements, which could be up to num bytes
      // sentinals in this leaf
      counter += 1;
      assert(counter < logN());
#endif
    }

    // then we need to walk back in the batch to find the first element that
    // will go into this leaf now that we know the starting source
    key_type head = index_to_head_key(leaf_number);
    if (head >= pcsr_top_bit) {
      while (middle_index > 0 &&
             std::get<0>(f(es[middle_index - 1])) >= this_head_src) {
        middle_index -= 1;
      }
    } else {
      while (middle_index > 0) {
        auto batch_element = f(es[middle_index - 1]);
        if (std::get<0>(batch_element) == this_head_src &&
            std::get<1>(batch_element) < head) {
          break;
        }
        if (std::get<0>(batch_element) < this_head_src) {
          break;
        }
        middle_index -= 1;
      }
    }
  }

  key_type next_head_dest =
      std::numeric_limits<key_type>::max() >> 1; // max int
  if (leaf_number + 1 < end_leaf_idx / elts_per_leaf()) {
    next_head_dest = index_to_head_key(leaf_number + 1);
    assert(next_head_dest != 0);
    // this might actually be a sentinal, but thats fine, since in that case its
    // part of the next node and the src will will us to add what we should
  }

// in this case we need to figure out what source controls the head of the
// next leaf, we can just walk forward in the offsets array to do this
#if DEBUG == 1
  size_t counter = 0;
#endif
  while (
      next_head_src < num_nodes_pcsr() &&
      (std::get<0>(get_info_from_raw_pointer(
           offsets_array.locations[next_head_src + 1])) == leaf_number ||
       get_info_from_raw_pointer(offsets_array.locations[next_head_src + 1]) ==
           std::pair(leaf_number + 1, true))) {
    next_head_src += 1;
#if DEBUG == 1
    // just make sure this soesn't turn into an infinite loop, there could
    // be at most max number of elements, which could be up to num bytes
    // sentinals in this leaf
    counter += 1;
    assert(counter < logN());
#endif
  }
  auto subrange = std::ranges::subrange(es.begin() + middle_index, es.end());
  assert(leaf_number >= start_leaf_idx / elts_per_leaf());
  // std::cout << "working on leaf  " << leaf_number << "\n";
  assert(index_to_head_key(leaf_number) != 0);
  auto result =
      get_leaf(leaf_number)
          .template strip_from_leaf_pcsr<head_form == InPlace, parallel,
                                         traits::maintain_offsets>(
              subrange, f, this_head_src, next_head_src, next_head_dest,
              offsets_array);

  static_assert(!support_rank);
  num_elts_removed += std::get<1>(result);
#if DEBUG == 1
  {
    size_t num_marked = 0;
    for (auto i = middle_index; i < middle_index + std::get<0>(result); i++) {
      if (std::get<0>(es[i]) >= pcsr_top_bit) {
        num_marked += 1;
      }
    }
    assert(num_marked == std::get<1>(result));
  }

#endif
  auto bytes_used = std::get<2>(result);
  auto index_removed_up_to = middle_index + std::get<0>(result);
  if (std::get<1>(result)) {
    static_assert(!store_density);
    ASSERT(bytes_used ==
               get_density_count(
                   leaf_number * elts_per_leaf() * sizeof(key_type), logN()),
           "got %lu, expected %lu\n", bytes_used,
           get_density_count(leaf_number * elts_per_leaf() * sizeof(key_type),
                             logN()));
    if (bytes_used < logN() * lower_density_bound(H())) {
      leaves_to_check.push_back({leaf_number * elts_per_leaf(), bytes_used});
    }
  }

  // do the recursion splitting the batch into the early and the late

  uint64_t ret1 = 0;
  uint64_t ret2 = 0;

  auto early_range =
      std::ranges::subrange(es.begin(), es.begin() + middle_index);
  auto late_range =
      std::ranges::subrange(es.begin() + index_removed_up_to, es.end());

  const auto early_range_start_leaf_idx = start_leaf_idx;
  const auto early_range_end_leaf_idx = leaf_number * elts_per_leaf();
  const auto late_range_start_leaf_idx = (leaf_number + 1) * elts_per_leaf();
  const auto late_range_end_leaf_idx = end_leaf_idx;

  if (early_range.size() == 0 && late_range.size() == 0) {
    return num_elts_removed;
  }
  if (early_range.size() == 0) {
    ret2 = remove_batch_internal_pcsr(late_range, f, leaves_to_check,
                                      late_range_start_leaf_idx,
                                      late_range_end_leaf_idx);
  } else if (late_range.size() == 0) {
    ret1 = remove_batch_internal_pcsr(early_range, f, leaves_to_check,
                                      early_range_start_leaf_idx,
                                      early_range_end_leaf_idx);
  } else {
    if constexpr (!parallel) {
      ret1 = remove_batch_internal_pcsr(early_range, f, leaves_to_check,
                                        early_range_start_leaf_idx,
                                        early_range_end_leaf_idx);
      ret2 = remove_batch_internal_pcsr(late_range, f, leaves_to_check,
                                        late_range_start_leaf_idx,
                                        late_range_end_leaf_idx);
    } else {
      if (early_range.size() <= 20 || late_range.size() <= 20) {
        ret1 = remove_batch_internal_pcsr(early_range, f, leaves_to_check,
                                          early_range_start_leaf_idx,
                                          early_range_end_leaf_idx);
        ret2 = remove_batch_internal_pcsr(late_range, f, leaves_to_check,
                                          late_range_start_leaf_idx,
                                          late_range_end_leaf_idx);
      } else {
        ParallelTools::par_do(
            [&]() {
              ret1 = remove_batch_internal_pcsr(early_range, f, leaves_to_check,
                                                early_range_start_leaf_idx,
                                                early_range_end_leaf_idx);
            },
            [&]() {
              ret2 = remove_batch_internal_pcsr(late_range, f, leaves_to_check,
                                                late_range_start_leaf_idx,
                                                late_range_end_leaf_idx);
            });
      }
    }
  }

  num_elts_removed += ret1;
  num_elts_removed += ret2;
  return num_elts_removed;
}

template <typename traits>
template <std::ranges::random_access_range R>
uint64_t CPMA<traits>::remove_batch_pcsr(
    R &es, std::invocable<typename std::ranges::range_value_t<R>> auto &&f,
    [[maybe_unused]] bool sorted) {
  // print_pma();
  // std::cout << "batch\n";
  // for (const auto &elem : es) {
  //   std::cout << std::get<0>(elem) << ", " << std::get<1>(elem) << "\n";
  // }
  assert(verify_pcsr_nodes());
  static_assert(!support_rank, "we don't support rank while doing pcsr stuff");
  uint64_t batch_size = es.size();
  if (batch_size < 100) {
    uint64_t count = 0;
    for (const auto &e : es) {
      auto elem = f(e);
      count += remove_pcsr(std::get<0>(elem), std::get<1>(elem));
    }
    return count;
  }

  // the sentinals will always already be there
  assert(get_element_count() > 0);

  if (!sorted) {
    parlay::integer_sort_inplace(es, [](const auto &elem) {
      return (static_cast<uint64_t>(std::get<0>(elem)) << 32U) |
             std::get<1>(elem);
    });
  }

  // don't need to deal with zero since all of the elements will be greater
  // than zero since we bump up the dests by 1, which is what the f function
  // does in practice

  std::conditional_t<
      parallel, ParallelTools::Reducer_Vector<std::pair<uint64_t, uint64_t>>,
      std::vector<std::pair<uint64_t, uint64_t>>>
      leaves_to_check;

  const uint64_t num_elts_removed = remove_batch_internal_pcsr(
      es, f, leaves_to_check, 0, total_leaves() * elts_per_leaf());

  // TODO(wheatman) maybe do this stuff in parallel with some other stuff if we
  // need more parallelism
  //   the batch has been marked with the high bit of the src marked if it was
  //   actually removed, use this to help update the degree array
  //  TODO(wheatman) this can probably be faster, but its simple and this should
  //  be a resonable approach
  auto filtered =
      parlay::map_maybe(es, [](const auto &elem) -> std::optional<key_type> {
        if (std::get<0>(elem) >= pcsr_top_bit) {
          return std::get<0>(elem) ^ pcsr_top_bit;
        } else {
          return {};
        }
      });

  auto counts = parlay::histogram_by_key(filtered);
  ParallelTools::parallel_for(0, counts.size(), [&](size_t i) {
    offsets_array.degrees[counts[i].first] -= counts[i].second;
  });

  ASSERT(num_elts_removed ==
             parlay::delayed::reduce(parlay::delayed::map(
                 counts, [](const auto &elem) { return elem.second; })),
         "num_elts_removed = %lu, num_marked = %lu\n", num_elts_removed,
         parlay::delayed::reduce(parlay::delayed::map(
             counts, [](const auto &elem) { return elem.second; })));

  auto [ranges_to_redistribute_3, full_opt] = get_ranges_to_redistibute(
      leaves_to_check, num_elts_removed, [&](uint64_t level, float density) {
        return (density < lower_density_bound(level));
      });
  if (full_opt.has_value()) {

    uint64_t shrink_times = 0;
    auto bytes_occupied = full_opt.value();

    // min bytes necessary to meet the density bound
    uint64_t bytes_required = bytes_occupied / lower_density_bound(0);
    if (bytes_required == 0) {
      bytes_required = 1;
    }

    while (meta_data[meta_data_index - shrink_times].n >= bytes_required) {
      shrink_times += 1;
      if (meta_data_index == shrink_times) {
        break;
      }
    }
    shrink_list(shrink_times);
  } else { // not doubling
    // in parallel, redistribute ranges
    redistribute_ranges(ranges_to_redistribute_3);
  }
  count_elements_ -= num_elts_removed;

  assert(verify_pcsr_nodes());
  assert(verify_pcsr_degrees());
#if DEBUG == 1
  size_t edge_count = 0;
  for (size_t i = 0; i < num_nodes_pcsr(); i++) {
    map_neighbors_pcsr<true, false>(
        i, [&edge_count]([[maybe_unused]] auto args1,
                         [[maybe_unused]] auto arg2) { edge_count += 1; });
  }
  ASSERT(edge_count == num_edges_pcsr(),
         "edge_count = %lu, num_edges_pcsr() = %lu\n", edge_count,
         num_edges_pcsr());
#endif
  return num_elts_removed;
}

#endif
