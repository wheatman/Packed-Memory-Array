#pragma once

#include "PMA/CPMA.hpp"
#include "PMA/internal/leaf.hpp"

template <class key_t, class val_t, bool compressed_keys = true> class PMAkv {
  CPMA<PMA_traits<
      typename std::conditional<compressed_keys, delta_compressed_leaf<key_t>,
                                uncompressed_leaf<key_t>>::type,
      Eytzinger, 0, false, true>>
      key_pma;

  CPMA<PMA_traits<uncompressed_leaf<val_t>, Eytzinger, 0, false, true>>
      value_pma;

public:
  bool insert_or_update(key_t key, val_t value) {
    auto [new_element, rank] = key_pma.insert_get_rank(key);
    if (new_element) {
      value_pma.insert_by_rank(value, rank);
      return true;
    } else {
      value_pma.update_by_rank(value, rank);
      return false;
    }
  }

  std::pair<bool, val_t> get(key_t key) {
    if (key_pma.has(key)) {
      uint64_t rank = key_pma.rank(key);
      return {true, value_pma.select(rank)};
    }
    return {false, {}};
  }
};
