#pragma once

#include <bit>
#include <cstddef>
#include <limits>
#include <ranges>
#include <type_traits>

#include "PMA/internal/helpers.hpp"
#include "PMA/internal/leaf.hpp"
#include "ParallelTools/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "CPMA.hpp"

template <typename l, typename value_update_ =
                          overwrite_on_insert<typename l::element_ref_type,
                                              typename l::element_type>>
using pcsr_settings =
    PMA_traits<l, Linear, 0, false, false, false, 0, true, true, value_update_>;

template <typename key_type = uint32_t>
using simple_pcsr_settings = PMA_traits<uncompressed_leaf<key_type>, Linear, 0,
                                        false, false, false, 0, true, true>;

template <typename key_type, typename value_type>
using simple_wpcsr_settings =
    PMA_traits<uncompressed_leaf<key_type, value_type>, Linear, 0, false, false,
               false, 0, true, true>;

template <typename traits> class PCSR {
  using T = typename traits::key_type;
  static_assert(std::is_integral_v<T>);
  static constexpr bool binary = traits::binary;

  static constexpr T top_bit = (std::numeric_limits<T>::max() >> 1) + 1;
  static_assert(std::has_single_bit(top_bit));

  CPMA<traits> pma;

public:
  using node_t = T;
  using value_type = typename traits::value_type;
  // bool in binary/
  // the value if there is only 1 value
  // tuple of the values if there are multiple
  static constexpr auto get_value_type() {
    if constexpr (binary) {
      return (bool)true;
    } else {
      value_type v;
      if constexpr (std::tuple_size_v<value_type> == 1) {
        return std::get<0>(v);
      } else {
        return v;
      }
    }
  }
  using weight_t = decltype(get_value_type());
  using extra_data_t = std::nullptr_t;

  PCSR(T num_nodes) : pma(make_pcsr(), num_nodes) {}
  template <std::ranges::random_access_range R,
            typename Projection = std::identity>
  PCSR(T num_nodes, R &edges, bool sorted = false, Projection projection = {})
      : pma(make_pcsr(), 16) {
    if (!sorted) {
      parlay::sort_inplace(edges);
    }
    auto maped_edges = parlay::delayed_map(edges, [projection](const auto &e) {
      auto elem = projection(e);
      assert(std::get<0>(elem) < top_bit);
      assert(std::get<1>(elem) < top_bit);
      std::get<1>(elem) = std::get<1>(elem) + 1;
      return elem;
    });
    pma = CPMA<traits>(make_pcsr(), num_nodes, maped_edges);
  }
  bool contains(T src, T dest) const {
    assert(src < top_bit);
    assert(dest < top_bit);
    return pma.contains_pcsr(src, dest + 1);
  }
  bool insert(T src, T dest) {
    static_assert(binary);
    assert(src < top_bit);
    assert(dest < top_bit);
    bool ret = pma.insert_pcsr(src, dest + 1);
    assert(contains(src, dest));
    return ret;
  }

  bool insert(T src, T dest, value_type val) {
    static_assert(!binary);
    assert(src < top_bit);
    assert(dest < top_bit);
    bool ret = pma.insert_pcsr(src, dest + 1, val);
    assert(contains(src, dest));
    return ret;
  }

  bool remove(T src, T dest) {
    assert(src < top_bit);
    assert(dest < top_bit);
    bool ret = pma.remove_pcsr(src, dest + 1);
    assert(!contains(src, dest));
    return ret;
  }

  template <std::ranges::random_access_range R,
            typename Projection = std::identity>
  uint64_t insert_batch(R &es, bool sorted = false,
                        Projection projection = {}) {
    auto ret = pma.insert_batch_pcsr(
        es,
        [projection](const auto &e) {
          auto elem = projection(e);
          assert(std::get<0>(elem) < top_bit);
          assert(std::get<1>(elem) < top_bit);
          std::get<1>(elem) = std::get<1>(elem) + 1;
          return elem;
        },
        sorted);
#if DEBUG == 1
    bool contains_all = true;
    for (auto &elem : es) {
      auto src = std::get<0>(elem);
      if (src >= top_bit) {
        src ^= top_bit;
      }
      if (!contains(src, std::get<1>(elem))) {
        contains_all = false;
        std::cout << "don't have something after inserts\n";
      }
    }
    assert(contains_all);
#endif
    return ret;
  }
  template <std::ranges::random_access_range R>
  uint64_t remove_batch(R &es, bool sorted = false) {
    auto ret = pma.remove_batch_pcsr(
        es,
        [](const auto &e) {
          auto elem = e;
          assert(std::get<0>(elem) < top_bit);
          assert(std::get<1>(elem) < top_bit);
          std::get<1>(elem) = std::get<1>(elem) + 1;
          return elem;
        },
        sorted);
#if DEBUG == 1
    bool contains_none = true;
    for (auto &elem : es) {
      auto src = std::get<0>(elem);
      if (src >= top_bit) {
        src ^= top_bit;
      }
      if (contains(src, std::get<1>(elem))) {
        contains_none = false;
        std::cout << "have " << src << ", " << std::get<1>(elem)
                  << " after deletes\n";
      }
    }
    assert(contains_none);
#endif
    return ret;
  }
  size_t get_memory_size() { return pma.get_size(); }

  void print() const {
    pma.print_pma();
    T node = 0;
    if constexpr (binary) {
      pma.template map<true>([&node](T elem) {
        if (elem >= top_bit) {
          std::cout << "\n node " << (elem ^ top_bit) << ":\n";
          node += 1;
        } else {
          std::cout << elem - 1 << ", ";
        }
      });
      std::cout << "\n";
    } else {
      pma.template map<true>([&node](const auto &elem) {
        auto dest = std::get<0>(elem);
        if (dest >= top_bit) {
          std::cout << "\n node " << (dest ^ top_bit) << ":\n";
          node += 1;
        } else {
          std::cout << "(" << dest - 1 << ", " << leftshift_tuple(elem)
                    << "), ";
        }
      });
      std::cout << "\n";
    }
  }
  template <int early_exit = 0, class F>
  static constexpr bool get_F_early_exit() {
    static_assert(early_exit >= -1 && early_exit <= 1);
    if constexpr (early_exit == 0) {
      if constexpr (requires(const F &f) { F::no_early_exit; }) {
        return F::no_early_exit;
      } else {
        return true;
      }
    } else if constexpr (early_exit == 1) {
      return false;
    } else if constexpr (early_exit == -1) {
      return true;
    }
  }
  [[nodiscard]] uint64_t get_size() const { return pma.get_size(); }
  template <int early_exit = 0, class F>
  void map_neighbors(uint64_t i, F f, [[maybe_unused]] std::nullptr_t unused,
                     bool run_parallel) const {
    constexpr bool no_early_exit = get_F_early_exit<early_exit, F>();
    if constexpr (binary) {
      auto f2 = [&](auto src, auto dest) {
        return f(src, element_or_first_element(dest) - 1);
      };
      if (run_parallel) {
        pma.template map_neighbors_pcsr<no_early_exit, true>(i, f2);
      } else {
        pma.template map_neighbors_pcsr<no_early_exit, false>(i, f2);
      }
    } else {
      auto f2 = [&](auto src, auto dest_and_val) {
        if constexpr (std::tuple_size_v<value_type> == 1) {
          return f(src, std::get<0>(dest_and_val) - 1,
                   std::get<1>(dest_and_val));
        } else {
          return f(src, std::get<0>(dest_and_val) - 1,
                   leftshift_tuple(dest_and_val));
        }
      };
      if (run_parallel) {
        pma.template map_neighbors_pcsr<no_early_exit, true>(i, f2);
      } else {
        pma.template map_neighbors_pcsr<no_early_exit, false>(i, f2);
      }
    }
  }
  [[nodiscard]] T num_nodes() const { return pma.num_nodes_pcsr(); }
  [[nodiscard]] size_t num_edges() const { return pma.num_edges_pcsr(); }
  [[nodiscard]] size_t
  get_degree(T node, [[maybe_unused]] std::nullptr_t unused = {}) const {
    return pma.degree_pcsr(node);
  }
  void write_adj_file(const std::string &filename) {
    std::ofstream myfile;
    myfile.open(filename);
    myfile << "AdjacencyGraph\n";

    myfile << num_nodes() << "\n";
    myfile << num_edges() << "\n";
    uint64_t running_edge_total = 0;
    for (uint64_t i = 0; i < num_nodes(); i++) {
      myfile << running_edge_total << "\n";
      running_edge_total += get_degree(i);
    }
    if constexpr (binary) {
      for (uint64_t i = 0; i < num_nodes(); i++) {
        map_neighbors(
            i,
            [&]([[maybe_unused]] auto src, auto dest) {
              myfile << dest << "\n";
            },
            nullptr, false);
      }
    } else {
      for (uint64_t i = 0; i < num_nodes(); i++) {
        map_neighbors(
            i,
            [&]([[maybe_unused]] auto src, auto dest,
                [[maybe_unused]] auto val) { myfile << dest << "\n"; },
            nullptr, false);
      }
      for (uint64_t i = 0; i < num_nodes(); i++) {
        map_neighbors(
            i,
            [&]([[maybe_unused]] auto src, [[maybe_unused]] auto dest,
                auto val) { myfile << val << "\n"; },
            nullptr, false);
      }
    }
    myfile.close();
  }
};