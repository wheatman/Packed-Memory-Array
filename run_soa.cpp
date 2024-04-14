#include "PMA/CPMA.hpp"
#include "PMA/internal/helpers.hpp"
#include "PMA/internal/leaf.hpp"
#include "StructOfArrays/soa.hpp"

#include <iostream>
#include <limits>
#include <tuple>

#include "PMA/internal/test_map.hpp"

#if !defined(KEY_TYPE)
#define KEY_TYPE uint64_t
#endif
using key_type = KEY_TYPE;

#if !defined(HEADFORM)
#define HEADFORM InPlace
#endif
static constexpr HeadForm head_form = HEADFORM;
static constexpr uint64_t B_size = (head_form == BNary) ? 17 : 0;

#if STORE_DENSITY
static constexpr bool store_density = true;
#else
static constexpr bool store_density = false;
#endif

#if SUPPORT_RANK
static constexpr bool support_rank = true;
#else
static constexpr bool support_rank = false;
#endif

template <typename T, typename U> struct sum_on_duplicate {
  constexpr void operator()(T current_value, U new_value) {
    add_to_tuple(leftshift_tuple(current_value), leftshift_tuple(new_value));
  }
};

int main([[maybe_unused]] int32_t argc, char *argv[]) {

  // uint64_t num_elements = 1000000;

  // test_tlx_btree_ordered_insert<uint32_t, uint32_t>(num_elements);
  // test_tlx_btree_ordered_insert<uint32_t, uint32_t, uint32_t>(num_elements);
  // test_tlx_btree_ordered_insert<uint32_t, uint32_t, uint32_t, uint32_t>(
  //     num_elements);

  // test_cpma_ordered_insert<PMA_traits<uncompressed_leaf<uint32_t>, InPlace>>(
  //     num_elements);

  // test_cpma_ordered_insert<
  //     PMA_traits<uncompressed_leaf<uint32_t, uint32_t>,
  //     InPlace>>(num_elements);

  // test_cpma_ordered_insert<
  //     PMA_traits<uncompressed_leaf<uint32_t, uint32_t, uint32_t>, InPlace>>(
  //     num_elements);
  // test_cpma_ordered_insert<PMA_traits<
  //     uncompressed_leaf<uint32_t, uint32_t, uint32_t, uint32_t>, InPlace>>(
  //     num_elements);
  // std::seed_seq seed{0};
  // test_tlx_btree_unordered_insert<uint32_t, uint32_t, uint32_t>(num_elements,
  //                                                               seed, 10);
  // std::seed_seq seed2{0};
  // test_cpma_unordered_insert<
  //     PMA_traits<uncompressed_leaf<uint32_t, uint32_t, uint32_t>, InPlace>>(
  //     num_elements, seed2, 10);

  if (std::string("verify") == argv[1]) {

    if (verify_cpma_different_sizes<PMA_traits<
            uncompressed_leaf<key_type, uint16_t>, head_form, B_size>>(
            {{100, false}, {1000, false}, {10000, false}, {20000, true}})) {
      return 1;
    }
    if (verify_cpma_different_sizes<PMA_traits<
            uncompressed_leaf<key_type, double, uint32_t>, head_form, B_size>>(
            {{100, false}, {1000, false}, {10000, false}, {20000, true}})) {
      return 1;
    }

    if (verify_cpma_different_sizes<
            PMA_traits<uncompressed_leaf<key_type, float, double, uint8_t>,
                       head_form, B_size>>(
            {{100, false}, {1000, false}, {10000, false}, {20000, true}})) {
      return 1;
    }
    if (verify_cpma_different_sizes<
            PMA_traits<uncompressed_leaf<key_type, uint32_t>, head_form, B_size,
                       store_density, support_rank, false, 0, true, false,
                       sum_on_duplicate<std::tuple<key_type &, uint32_t &>,
                                        std::tuple<key_type, uint32_t>>>>(
            {{100, false}, {1000, false}, {10000, false}, {20000, true}})) {
      return 1;
    }
  }
}
