#pragma once

#include "PMA/CPMA.hpp"
#include "PMA/PMAkv.hpp"
#include "ParallelTools/parallel.h"
#include "ParallelTools/reducer.h"
#include <cstdint>

#include "PMA/internal/helpers.hpp"
#include "PMA/internal/io_util.hpp"
#include "PMA/internal/leaf.hpp"
#include "PMA/internal/rmat_util.hpp"
#include "PMA/internal/zipf.hpp"
#include "StructOfArrays/soa.hpp"

#include "EdgeMapVertexMap/algorithms/BC.h"
#include "EdgeMapVertexMap/algorithms/BFS.h"
#include "EdgeMapVertexMap/algorithms/Components.h"
#include "EdgeMapVertexMap/algorithms/PageRank.h"

#include <iomanip>
#include <limits>
#include <set>
#include <sys/time.h>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include <algorithm> // generate
#include <iostream>  // cout
#include <iterator>  // begin, end, and ostream_iterator
#include <random>    // mt19937 and uniform_int_distribution
#include <utility>
#include <vector> // vector

#include "tlx/container/btree_map.hpp"
#include "tlx/container/btree_set.hpp"
// just redefines some things in the tlx btree so I can get access to the size
// of the nodes
#include "btree_size_helpers.hpp"

template <class T> T get_random_value(std::mt19937_64 &eng) {
  static_assert(
      std::is_floating_point_v<T> || std::is_integral_v<T> ||
          std::is_same_v<T, uint24_t>,
      "only implemented random generation for integers, floats, and uint24_t");
  if constexpr (std::is_floating_point_v<T>) {
    std::uniform_real_distribution<T> dist(0, std::numeric_limits<T>::max());
    return dist(eng);
  } else {
    std::uniform_int_distribution<uint64_t> dist(0,
                                                 std::numeric_limits<T>::max());
    return dist(eng);
  }
}

template <typename... Values>
std::tuple<Values...> get_random_tuple(std::mt19937_64 &eng) {
  return std::make_tuple(get_random_value<Values>(eng)...);
}

template <typename... Values>
SOA<Values...> create_random_data_with_seed(size_t n, std::seed_seq &seed) {
  SOA<Values...> soa(n);
  std::mt19937_64 eng(seed);
  for (uint64_t i = 0; i < n; i++) {
    soa.get(i) = get_random_tuple<Values...>(eng);
  }
  // ensure that their is the zero element to test for that special case
  std::get<0>(soa.get(0)) = 0;
  return soa;
}

template <typename... Values>
SOA<Values...>
create_random_data_with_seed(size_t n, std::seed_seq &seed,
                             [[maybe_unused]] std::tuple<Values...> unused) {
  return create_random_data_with_seed<Values...>(n, seed);
}

template <typename... Values> SOA<Values...> create_random_data(size_t n) {

  std::random_device rd;
  auto seed = rd();
  std::seed_seq s{seed};
  return create_random_data_with_seed<Values...>(n, s);
}

template <typename... Values>
SOA<Values...>
create_random_data(size_t n, [[maybe_unused]] std::tuple<Values...> unused) {
  return create_random_data<Values...>(n);
}

template <typename... Values>
SOA<Values...> create_random_data_in_parallel(size_t n) {

  SOA<Values...> soa(n);
  uint64_t per_worker = (n / ParallelTools::getWorkers()) + 1;
  ParallelTools::parallel_for(0, ParallelTools::getWorkers(), [&](uint64_t i) {
    uint64_t start = i * per_worker;
    uint64_t end = (i + 1) * per_worker;
    if (end > n) {
      end = n;
    }
    std::mt19937_64 eng(i); // a source of random data

    for (size_t j = start; j < end; j++) {
      soa.get(j) = get_random_tuple<Values...>(eng);
    }
  });
  return soa;
}

template <typename... value_types>
std::tuple<value_types...> get_element_of_type(uint64_t i) {
  return {((value_types)i)...};
}

template <typename... value_types>
std::tuple<value_types...>
get_element_of_type(uint64_t i,
                    [[maybe_unused]] std::tuple<value_types...> unused) {
  return {((value_types)i)...};
}

template <typename... T, size_t... I>
std::tuple<T &...> makeReferencesHelper(std::tuple<T...> &t,
                                        std::index_sequence<I...>) {
  return std::tie(std::get<I>(t)...);
}

template <typename... T>
std::tuple<T &...> makeReferences(std::tuple<T...> &t) {
  return makeReferencesHelper<T...>(t,
                                    std::make_index_sequence<sizeof...(T)>{});
}

template <typename key_type, typename... value_types>
bool test_single_leaf(uint64_t elements_to_test) {
  static constexpr bool head_in_place = false;
  std::tuple<key_type, value_types...> head;
  uint64_t length = elements_to_test + 2;

  SOA<key_type, value_types...> soa(length);
  soa.zero();
  uncompressed_leaf<key_type, value_types...> leaf(
      makeReferences(head), soa.get_ptr(0), length * sizeof(key_type));
  for (uint64_t i = 1; i <= elements_to_test; i++) {
    leaf.template insert<head_in_place>(
        get_element_of_type<key_type, value_types...>(i));
  }
  // leaf.print();

  for (uint64_t i = 1; i <= elements_to_test; i += 2) {
    leaf.template remove<head_in_place>(i);
  }
  // leaf.print();
  return false;
}

template <typename key_type, typename... value_types>
bool test_leaf_merge(uint64_t elements_to_test, uint64_t leaves) {
  static constexpr bool head_in_place = false;
  using leaf_type = uncompressed_leaf<key_type, value_types...>;
  std::vector<std::tuple<key_type, value_types...>> heads(leaves);
  uint64_t length = elements_to_test + 2;

  SOA<key_type, value_types...> soa(length * leaves);
  soa.zero();
  for (uint64_t i = 0; i < leaves; i++) {
    leaf_type leaf(makeReferences(heads[i]), soa.get_ptr(i * length),
                   length * sizeof(key_type));

    for (uint64_t j = 1; j <= elements_to_test; j++) {
      leaf.template insert<head_in_place>(
          get_element_of_type<key_type, value_types...>(i * 1000 + j));
    }
    leaf.print();

    for (uint64_t j = 1; j <= elements_to_test; j += 2) {
      leaf.template remove<head_in_place>(i * 1000 + j);
    }
    leaf.print();
  }

  auto merged_data = leaf_type::template merge<head_in_place, false>(
      soa.get_ptr(0), leaves, length * sizeof(key_type), 0,
      [&heads](size_t idx) -> typename leaf_type::element_ref_type {
        return makeReferences(heads[idx]);
      },
      nullptr);
  merged_data.leaf.print();
  std::cout << "size = " << merged_data.size << "\n";

  merged_data.free();

  return false;
}

template <typename key_type, typename... value_types>
bool test_merge_into_leaf(uint64_t elements_to_test) {
  static constexpr bool head_in_place = false;
  std::tuple<key_type, value_types...> head;
  uint64_t length = 2 * elements_to_test + 2;

  SOA<key_type, value_types...> soa(length);
  soa.zero();

  SOA<key_type, value_types...> batch(elements_to_test);
  batch.zero();
  for (uint64_t i = 2; i <= elements_to_test; i += 2) {
    batch.get(i / 2 - 1) = get_element_of_type<key_type, value_types...>(i);
  }

  uncompressed_leaf<key_type, value_types...> leaf(
      makeReferences(head), soa.get_ptr(0), length * sizeof(key_type));
  for (uint64_t i = 1; i <= elements_to_test; i += 2) {
    leaf.template insert<head_in_place>(
        get_element_of_type<key_type, value_types...>(i));
  }
  leaf.print();

  leaf.template merge_into_leaf<head_in_place>(
      batch.get_ptr(0), batch.template get_ptr<0>(0) + elements_to_test,
      std::numeric_limits<uint64_t>::max());

  leaf.print();

  std::vector<key_type> batch2(elements_to_test / 2);
  for (uint64_t i = 0; i < elements_to_test / 2; i++) {
    batch2[i] = 1 + i * 3;
  }
  leaf.template strip_from_leaf<head_in_place>(
      batch2.data(), batch2.data() + batch2.size(),
      std::numeric_limits<uint64_t>::max());

  leaf.print();
  return false;
}

template <class T, typename... Values>
void test_tlx_btree_ordered_insert(uint64_t max_size) {
  static_assert(std::is_integral_v<T>);
  if (max_size > std::numeric_limits<T>::max()) {
    max_size = std::numeric_limits<T>::max();
  }
  uint64_t start = 0;
  uint64_t end = 0;
  tlx::btree_map<T, std::tuple<Values...>> m;

  start = get_usecs();
  for (uint32_t i = 1; i < max_size; i++) {
    m.insert(std::make_pair(i, get_element_of_type<Values...>(i)));
  }
  end = get_usecs();
  printf("insertion,\t %lu,\t", end - start);
  start = get_usecs();
  uint64_t sum = 0;
  for (auto el : m) {
    sum += el.first;
  }
  end = get_usecs();
  printf("key sum_time, \t%lu, \tsum_total, \t%lu\n", end - start, sum);
}

template <class T, typename... Values>
void test_tlx_btree_unordered_insert(uint64_t max_size, std::seed_seq &seed,
                                     uint64_t iters) {
  if (max_size > std::numeric_limits<T>::max()) {
    max_size = std::numeric_limits<T>::max();
  }
  auto data =
      create_random_data_with_seed<T, Values...>(max_size * iters, seed);
  uint64_t start = 0;
  uint64_t end = 0;
  std::vector<uint64_t> times;
  for (uint64_t it = 0; it < iters; it++) {
    tlx::btree_map<T, std::tuple<Values...>> m;
    start = get_usecs();
    for (uint64_t i = 0; i < max_size; i++) {
      m.insert(
          std::make_pair(std::get<0>(data.template get<0>(it * max_size + i)),
                         leftshift_tuple(data.get(it * max_size + i))));
    }
    end = get_usecs();
    printf("insertion, \t %lu,", end - start);
    times.push_back(end - start);
    uint64_t sum = 0;
    start = get_usecs();
    for (const auto e : m) {
      sum += e.first;
    }
    end = get_usecs();
    printf("sum, \t %lu, total = %lu\n", end - start, sum);
  }
  std::sort(times.begin(), times.end());
  printf("mean time %lu\n", times[iters / 2]);
}

template <typename traits> void test_cpma_ordered_insert(uint64_t max_size) {
  if (max_size > std::numeric_limits<typename traits::key_type>::max()) {
    max_size = std::numeric_limits<typename traits::key_type>::max();
  }
  uint64_t start = 0;
  uint64_t end = 0;
  CPMA<traits> m;
  start = get_usecs();
  for (uint32_t i = 0; i < max_size; i++) {
    m.insert(get_element_of_type(i, typename traits::element_type()));
  }
  end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  uint64_t sum = 0;
  start = get_usecs();
  sum = m.sum();
  end = get_usecs();
  printf("key sum_time, %lu, sum_total, %lu\n", end - start, sum);
}

template <typename traits>
void test_cpma_unordered_insert(uint64_t max_size, std::seed_seq &seed,
                                uint64_t iters) {
  if (max_size > std::numeric_limits<typename traits::key_type>::max()) {
    max_size = std::numeric_limits<typename traits::key_type>::max();
  }
  auto data = create_random_data_with_seed(max_size * iters, seed,
                                           typename traits::element_type());
  uint64_t start = 0;
  uint64_t end = 0;
  std::vector<uint64_t> times;
  for (uint64_t it = 0; it < iters; it++) {
    CPMA<traits> s;
    start = get_usecs();
    for (uint64_t i = 0; i < max_size; i++) {
      s.insert(data.get(it * max_size + i));
    }
    end = get_usecs();
    printf("insertion, \t %lu,", end - start);
    times.push_back(end - start);
    uint64_t sum = 0;
    start = get_usecs();
    sum = s.sum();
    end = get_usecs();
    printf("sum, \t %lu, total = %lu, size per element = %f\n", end - start,
           sum, ((double)s.get_size()) / s.get_element_count());
  }
  std::sort(times.begin(), times.end());
  printf("mean time %lu\n", times[iters / 2]);
}

template <class pma_type, class map_type>
bool pma_different_from_map(const pma_type &pma, const map_type &map) {
  // check that the right data is in the set
  uint64_t correct_sum = 0;
  for (auto e : map) {
    correct_sum += e.first;
    if (!pma.has(e.first)) {
      std::cout << "pma missing " << e << "\n";
      return true;
    }
    if (!approx_equal_tuple(pma.value(e.first), e.second)) {
      std::cout << "pma had the wrong value for the key " << e.first << "\n";
      std::cout << "got " << pma.value(e.first) << " expected " << e.second
                << "\n";
      return true;
    }
  }
  bool have_something_wrong = pma.template map<false>([&map](auto element) {
    uint64_t key = std::get<0>(element);
    auto value = leftshift_tuple(element);
    if (map.find(key) == map.end()) {
      printf("have something (%lu) that the set doesn't have\n", (uint64_t)key);

      return true;
    }
    if (!approx_equal_tuple((*map.find(key)).second, value)) {
      std::cout << "pma had the wrong value for the key " << key << "\n";
      std::cout << "got " << value << " expected " << (*map.find(key)).second
                << "\n";
      return true;
    }
    return false;
  });
  if (have_something_wrong) {
    printf("pma has something is shouldn't\n");
    return true;
  }
  if (correct_sum != pma.sum()) {
    printf("pma has bad key sum\n");
    return true;
  }
  return false;
}

template <typename traits>
bool verify_cpma(uint64_t number_elements, bool fast = false) {
  using key_type = typename traits::key_type;
  uint64_t max_num = std::numeric_limits<key_type>::max() - 1;
  if (max_num > std::numeric_limits<key_type>::max()) {
    max_num = std::numeric_limits<key_type>::max() - 1;
  }
  {
    CPMA<traits> t;
    uint64_t sum = 0;
    for (uint64_t i = 1; i < number_elements; i += 1) {
      // t.print_pma();
      // std::cout << "trying to insert " << i << "\n";
      t.insert(get_element_of_type(i, typename traits::element_type()));
      sum += i;
      if (sum != t.sum()) {
        std::cout << "bad sum after inserting sequntial numbers" << std::endl;
        std::cout << "got " << t.sum() << " expected " << sum << std::endl;
        std::cout << "just inserted " << i << std::endl;
        t.print_pma();
        return true;
      }
    }
  }
  tlx::btree_map<key_type, typename traits::value_type> correct;
  CPMA<traits> test;
  // test.print_pma();
  auto data =
      create_random_data(number_elements, typename traits::element_type());
  for (uint64_t i = 0; i < number_elements; i++) {
    // test.print_pma();
    // std::cout << "inserting: " << data.get(i) << "\n";
    if (correct.find(std::get<0>(data.template get<0>(i))) == correct.end()) {
      correct[std::get<0>(data.template get<0>(i))] =
          leftshift_tuple(data.get(i));
    } else {
      key_type key = std::get<0>(data.template get<0>(i));
      auto value = correct[key];
      typename traits::element_type el =
          std::tuple_cat(std::make_tuple(key), value);
      typename traits::element_ref_type el_ref = MakeTupleRef(el);

      typename traits::value_update()(el_ref, data.get(i));
      correct[key] = leftshift_tuple(el_ref);
    }

    test.insert(data.get(i));
    assert(test.check_nothing_full());
    if (!fast) {
      if (pma_different_from_map(test, correct)) {
        printf("issue during inserts\n");
        test.print_pma();
        return true;
      }
    }
  }
  if (pma_different_from_map(test, correct)) {
    printf("issue after inserts\n");
    return true;
  }
  {
    auto random_numbers = create_random_data<key_type>(number_elements);
    for (uint64_t i = 0; i < number_elements; i++) {
      key_type x = std::get<0>(random_numbers.get(i));
      // test.print_pma();
      // std::cout << "removing: " << x << std::endl;
      correct.erase(x);
      test.remove(x);
      if (!fast) {
        if (pma_different_from_map(test, correct)) {
          printf("issue during deletes\n");
          return true;
        }
      }
    }
  }

  if (pma_different_from_map(test, correct)) {
    printf("issue after deletes\n");
    return true;
  }
  // test.print_pma();
  uint64_t num_rounds = 10;
  for (uint64_t round = 0; round < num_rounds; round++) {
    // put stuff into the pma

    auto batch = create_random_data(number_elements / num_rounds,
                                    typename traits::element_type());

    // printf("before insert\n");
    // test.print_pma();
    // // test.print_array();

    // std::sort(batch.begin(), batch.end());
    // printf("\n*** BATCH %lu ***\n", round);
    // for (auto elt : batch) {
    //   std::cout << elt.get() << ", ";
    // }
    // std::cout << std::endl;

    // try inserting batch
    test.insert_batch(batch.get_ptr(0), batch.size());

    for (uint64_t i = 0; i < number_elements / num_rounds; i++) {
      if (correct.find(std::get<0>(batch.template get<0>(i))) ==
          correct.end()) {
        correct[std::get<0>(batch.template get<0>(i))] =
            leftshift_tuple(batch.get(i));
      } else {
        key_type key = std::get<0>(batch.template get<0>(i));
        auto value = correct[key];
        typename traits::element_type el =
            std::tuple_cat(std::make_tuple(key), value);
        typename traits::element_ref_type el_ref = MakeTupleRef(el);
        typename traits::value_update()(el_ref, batch.get(i));
        correct[key] = leftshift_tuple(el_ref);
      }
    }

    // everything in batch has to be in test
    for (uint64_t i = 0; i < batch.size(); i++) {
      typename traits::element_type e = batch.get(i);
      key_type key = std::get<0>(e);
      if (!test.has(key)) {
        std::cout << "missing something in batch " << e << std::endl;
        test.print_pma();

        printf("\n*** BATCH ***\n");
        std::sort(batch.begin(), batch.end());
        for (typename traits::element_type elt : batch) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;

        printf("\n*** CORRECT ***\n");
        for (auto elt : correct) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;
        return true;
      }
      // the elements don't have to match the batch in the case where a single
      // key is in the batch multiple times, just make sure we match the correct
      // afterwords if (i > 0 && std::get<0>(batch.get(i)) ==
      // std::get<0>(batch.get(i - 1))) {
      //   continue;
      // }
      // if (test.value(key) != leftshift_tuple(e)) {

      //   std::cout << "bad value from batch " << e << std::endl;
      //   std::cout << "got " << test.value(key) << std::endl;
      //   test.print_pma();

      //   printf("\n*** BATCH ***\n");
      //   // std::sort(batch.begin(), batch.end());
      //   for (typename traits::element_type elt : batch) {
      //     std::cout << elt << ", ";
      //   }
      //   std::cout << std::endl;

      //   printf("\n*** CORRECT ***\n");
      //   for (auto elt : correct) {
      //     std::cout << elt << ", ";
      //   }
      //   std::cout << std::endl;
      //   return true;
      // }
    }

    if (pma_different_from_map(test, correct)) {
      printf("issue after batch insert\n");
      test.print_pma();

      printf("\n*** BATCH ***\n");
      // std::sort(batch.begin(), batch.end());
      for (typename traits::element_type elt : batch) {
        std::cout << elt << ", ";
      }
      std::cout << std::endl;

      printf("\n*** CORRECT ***\n");
      for (auto elt : correct) {
        std::cout << elt << ", ";
      }
      std::cout << std::endl;
      return true;
      return true;
    }
  }

  // random batch
  for (uint64_t round = 0; round < num_rounds; round++) {
    // put stuff into the pma
    std::vector<key_type> batch;
    auto random_numbers =
        create_random_data<key_type>(number_elements / num_rounds);
    for (uint64_t i = 0; i < number_elements / num_rounds; i++) {
      key_type x = std::get<0>(random_numbers.get(i));
      batch.push_back(x);
      correct.erase(x);
    }

    // printf("before remove\n");
    // test.print_pma();

    // std::sort(batch.begin(), batch.end());
    // printf("\n*** BATCH %lu ***\n", round);
    // for (auto elt : batch) {
    //   std::cout << elt << ", ";
    // }
    // std::cout << std::endl;

    // try removing batch
    test.remove_batch(batch.data(), batch.size());

    // everything in batch has to be in test
    for (key_type e : batch) {
      if (test.has(e)) {
        std::cout << "has something in random batch " << e << std::endl;
        test.print_pma();

        printf("\n*** BATCH ***\n");
        std::sort(batch.begin(), batch.end());
        for (key_type elt : batch) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;

        printf("\n*** CORRECT ***\n");
        for (auto elt : correct) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;
        return true;
      }
    }

    if (pma_different_from_map(test, correct)) {
      printf("issue after random batch delete\n");
      return true;
    }
  }
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<uint64_t> dist(0, correct.size());
  // batch of elements
  for (uint64_t round = 0; round < num_rounds; round++) {
    // put stuff into the pma
    std::vector<key_type> batch;
    for (uint64_t i = 0; i < number_elements / num_rounds; i++) {
      if (correct.empty()) {
        break;
      }
      // this is quite slow, but will generate random elements from the set
      uint32_t steps = dist(eng);
      auto it = std::begin(correct);
      std::advance(it, steps);
      if (it == std::end(correct)) {
        continue;
      }
      batch.push_back((*it).first);
      correct.erase((*it).first);
    }

    // printf("before insert\n");
    // test.print_pma();
    // test.print_array();

    // std::sort(batch.begin(), batch.end());
    // printf("\n*** BATCH %lu ***\n", round);
    // for (auto elt : batch) {
    //   std::cout << elt << ", ";
    // }
    // std::cout << std::endl;

    // try inserting batch
    test.remove_batch(batch.data(), batch.size());

    // everything in batch can't be in test
    for (key_type e : batch) {
      if (test.has(e)) {
        std::cout << "has something in batch " << e << std::endl;
        test.print_pma();

        printf("\n*** BATCH ***\n");
        std::sort(batch.begin(), batch.end());
        for (key_type elt : batch) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;

        printf("\n*** CORRECT ***\n");
        for (auto elt : correct) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;
        return true;
      }
    }

    if (pma_different_from_map(test, correct)) {
      printf("issue after batch delete of items\n");
      return true;
    }
  }

  return false;
}

template <typename traits>
bool verify_cpma_different_sizes(
    const std::vector<std::pair<uint64_t, bool>> &args) {
  for (auto arg : args) {
    printf("testing size %lu, fast = %d\n", std::get<0>(arg), std::get<1>(arg));
    if (verify_cpma<traits>(std::get<0>(arg), std::get<1>(arg))) {
      return true;
    }
  }
  return false;
}

/*

template <class traits>
bool real_graph(const std::string &filename, int iters = 20,
                uint32_t start_node = 0, uint32_t max_batch_size = 100000) {
  uint32_t num_nodes = 0;
  uint64_t num_edges = 0;
  auto edges = get_edges_from_file_adj_sym(filename, &num_edges, &num_nodes);

  printf("done reading in the file, n = %u, m = %lu\n", num_nodes, num_edges);
  CPMA<traits> g;

  auto start = get_usecs();
  // for (auto edge : edges) {
  //   g.insert(edge);
  // }
  g.insert_batch(edges.data(), edges.size());
  auto end = get_usecs();
  printf("inserting the edges took %lums\n", (end - start) / 1000);
  num_nodes = g.num_nodes();
  int64_t size = g.get_size();
  printf("size = %lu bytes, num_edges = %lu, num_nodes = %u\n", size,
         g.get_element_count(), num_nodes);

  int32_t parallel_bfs_result2_ = 0;
  uint64_t parallel_bfs_time2 = 0;

  for (int i = 0; i < iters; i++) {
    start = get_usecs();
    int32_t *parallel_bfs_result = EdgeMapVertexMap::BFS(g, start_node);
    end = get_usecs();
    parallel_bfs_result2_ += parallel_bfs_result[0];
    if (i == 0 && parallel_bfs_result != nullptr) {
      uint64_t reached = 0;
      for (uint32_t j = 0; j < num_nodes; j++) {
        reached += parallel_bfs_result[j] != -1;
      }
      printf("the bfs from source %u, reached %lu vertices\n", start_node,
             reached);
      std::vector<uint32_t> depths(num_nodes, UINT32_MAX);
      ParallelTools::parallel_for(0, num_nodes, [&](uint32_t j) {
        uint32_t current_depth = 0;
        int32_t current_parent = j;
        if (parallel_bfs_result[j] < 0) {
          return;
        }
        while (current_parent != parallel_bfs_result[current_parent]) {
          current_depth += 1;
          current_parent = parallel_bfs_result[current_parent];
        }
        depths[j] = current_depth;
      });
      std::ofstream myfile;
      myfile.open("bfs.out");
      for (unsigned int i = 0; i < num_nodes; i++) {
        myfile << depths[i] << "\n";
      }
      myfile.close();
    }

    free(parallel_bfs_result);
    parallel_bfs_time2 += (end - start);
  }
  // printf("bfs took %lums, parent of 0 = %d\n", (bfs_time)/(1000*iters),
  // bfs_result_/iters);
  printf("parallel_bfs with edge_map took %lums, parent of 0 = %d\n",
         parallel_bfs_time2 / (1000 * iters), parallel_bfs_result2_ / iters);
  printf("F-Graph, %d, BFS, %u, %s, ##, %f\n", iters, start_node,
         filename.c_str(), (double)parallel_bfs_time2 / (iters * 1000000));
  double pagerank_value = 0;
  uint64_t pagerank_time = 0;
  double *values3 = nullptr;
  for (int i = 0; i < iters; i++) {
    if (values3 != nullptr) {
      free(values3);
    }
    start = get_usecs();
    values3 = EdgeMapVertexMap::PR_S<double>(g, 10);
    end = get_usecs();
    pagerank_value += values3[0];
    pagerank_time += end - start;
  }
  printf("pagerank with MAPS took %f microsecond, value of 0 = %f, for %d "
         "iters, trash=%f\n",
         (double)pagerank_time / iters, values3[0], iters, pagerank_value);
  printf("F-Graph, %d, PageRank, %u, %s, ##, %f\n", iters, start_node,
         filename.c_str(), (double)pagerank_time / (iters * 1000000));
  std::ofstream myfile;
  myfile.open("pr.out");
  for (unsigned int i = 0; i < num_nodes; i++) {
    myfile << values3[i] << "\n";
  }
  myfile.close();
  free(values3);

  double *values4 = nullptr;
  double dep_0 = 0;
  uint64_t bc_time = 0;
  for (int i = 0; i < iters; i++) {
    if (values4 != nullptr) {
      free(values4);
    }
    start = get_usecs();
    values4 = EdgeMapVertexMap::BC(g, start_node);
    end = get_usecs();
    bc_time += end - start;
    dep_0 += values4[0];
  }

  printf("BC took %lums, value of 0 = %f\n", bc_time / (1000 * iters),
         dep_0 / iters);

  printf("F-Graph, %d, BC, %u, %s, ##, %f\n", iters, start_node,
         filename.c_str(), (double)bc_time / (iters * 1000000));
  if (values4 != nullptr) {
    std::ofstream myfile;
    myfile.open("bc.out");
    for (uint32_t i = 0; i < num_nodes; i++) {
      myfile << values4[i] << "\n";
    }
    myfile.close();
    free(values4);
  }

  uint32_t *values5 = nullptr;
  uint32_t id_0 = 0;
  uint64_t cc_time = 0;
  for (int i = 0; i < iters; i++) {
    if (values5) {
      free(values5);
    }
    start = get_usecs();
    values5 = EdgeMapVertexMap::CC(g);
    end = get_usecs();
    cc_time += end - start;
    id_0 += values5[0];
  }

  printf("CC took %lums, value of 0 = %u\n", cc_time / (1000 * iters),
         id_0 / iters);
  printf("F-Graph, %d, Components, %u, %s, ##, %f\n", iters, start_node,
         filename.c_str(), (double)cc_time / (iters * 1000000));
  if (values5 != nullptr) {
    std::unordered_map<uint32_t, uint32_t> components;
    for (uint32_t i = 0; i < num_nodes; i++) {
      components[values5[i]] += 1;
    }
    printf("there are %zu components\n", components.size());
    uint32_t curent_max = 0;
    uint32_t curent_max_key = 0;
    for (auto p : components) {
      if (p.second > curent_max) {
        curent_max = p.second;
        curent_max_key = p.first;
      }
    }
    printf("the element with the biggest component is %u, it has %u members "
           "to its component\n",
           curent_max_key, curent_max);
    std::ofstream myfile;
    myfile.open("cc.out");
    for (uint32_t i = 0; i < num_nodes; i++) {
      myfile << values5[i] << "\n";
    }
    myfile.close();
  }

  free(values5);

  if (true) {
    for (uint32_t b_size = 10; b_size <= max_batch_size; b_size *= 10) {
      auto r = random_aspen(b_size);
      double batch_insert_time = 0;
      double batch_remove_time = 0;
      for (int it = 0; it < iters + 1; it++) {
        // uint64_t size = g.get_memory_size();
        // printf("size start = %lu\n", size);
        double a = 0.5;
        double b = 0.1;
        double c = 0.1;
        size_t nn = 1UL << (log2_up(num_nodes) - 1);
        auto rmat = rMat<uint32_t>(nn, r.ith_rand(it), a, b, c);
        std::vector<uint64_t> es(b_size);
        ParallelTools::parallel_for(0, b_size, [&](uint32_t i) {
          std::pair<uint32_t, uint32_t> edge = rmat(i);
          es[i] = (static_cast<uint64_t>(edge.first) << 32U) | edge.second;
        });

        start = get_usecs();
        g.insert_batch(es.data(), b_size);
        end = get_usecs();
        // printf("%lu\n", end - start);
        if (it > 0) {
          batch_insert_time += end - start;
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(es.begin(), es.end(), gen);
        // size = g.get_memory_size();
        // printf("size end = %lu\n", size);
        start = get_usecs();
        g.remove_batch(es.data(), b_size);
        end = get_usecs();
        if (it > 0) {
          batch_remove_time += end - start;
        }
      }
      batch_insert_time /= (1000000 * iters);
      batch_remove_time /= (1000000 * iters);
      // printf("batch_size = %d, time to insert = %f seconds, throughput =
      // %4.2e "
      //        "updates/second\n",
      //        b_size, batch_insert_time, b_size / (batch_insert_time));
      // printf("batch_size = %d, time to remove = %f seconds, throughput =
      // %4.2e "
      //        "updates/second\n",
      //        b_size, batch_remove_time, b_size / (batch_remove_time));
      printf("%u, %f, %f\n", b_size, batch_insert_time, batch_remove_time);
    }
  }
  return true;
}



template <typename traits>
bool scan_bench(size_t num_elements, size_t num_bits, size_t iters = 5) {
  std::vector<uint64_t> times;
  for (size_t i = 0; i < iters; i++) {
    uint64_t start = 0;
    uint64_t end = 0;
    uint64_t sum = 0;
    auto data = create_random_data_in_parallel<typename traits::key_type>(
        num_elements, 1UL << num_bits);

    {
      CPMA<traits> uncompressed_pma;
      uncompressed_pma.insert_batch(data.data(), data.size());
      start = get_usecs();
      sum = uncompressed_pma.sum();
      end = get_usecs();
      printf("uncompressed sum took %lu microseconds was %lu: size = %lu "
             "head_size = %lu "
             "num_unique = %lu\n",
             end - start, sum, uncompressed_pma.get_size(),
             uncompressed_pma.get_head_structure_size(),
             uncompressed_pma.get_element_count());
      uint64_t sum2 = 0;
      uncompressed_pma.template map<true>([&sum2](auto el) { sum2 += el; });
      printf("sum2 = %lu\n", sum2);
      times.push_back((end - start));
    }
  }
  std::sort(times.begin(), times.end());
  printf("median after %zu iters was: %zu\n", iters, times[iters / 2]);

  return true;
}


bool scan_bench_btree(size_t num_elements, size_t num_bits, size_t iters = 5) {
  std::vector<uint64_t> btree_times;
  for (size_t i = 0; i < iters; i++) {
    std::seed_seq seed;
    uint64_t start = 0;
    uint64_t end = 0;
    uint64_t sum = 0;
    auto data = create_random_data<uint64_t>(num_elements, 1UL << num_bits);

    {

      tlx::btree_set<uint64_t> btree_set(data.begin(), data.end());

      uint64_t num_partitions = ParallelTools::getWorkers() * 10;
      uint64_t per_partition = num_elements / num_partitions;
      std::vector<tlx::btree_set<uint64_t>::const_iterator> iterators(
          num_partitions + 1);
      uint64_t position = 0;
      uint64_t partion_num = 0;
      for (auto it = btree_set.begin(); it != btree_set.end(); it++) {
        if (position % per_partition == 0) {
          iterators[partion_num] = it;
          partion_num += 1;
          if (partion_num == num_partitions) {
            break;
          }
        }
        position += 1;
      }
      uint64_t correct_sum = 0;
      uint64_t serial_start = get_usecs();
      for (auto it = btree_set.begin(); it != btree_set.end(); it++) {
        correct_sum += *it;
      }
      uint64_t serial_end = get_usecs();
      iterators[num_partitions] = btree_set.end();
      ParallelTools::Reducer_sum<uint64_t> partial_sums;
      start = get_usecs();
      ParallelTools::parallel_for(0, num_partitions, [&](uint64_t i) {
        uint64_t local_sum = 0;
        auto start = iterators[i];
        auto end = iterators[i + 1];
        for (auto it = start; it != end; it++) {
          local_sum += *it;
        }
        partial_sums.add(local_sum);
      });
      end = get_usecs();
      sum = partial_sums.get();
      struct key_of_value {
        static const uint64_t &get(const uint64_t &v) { return v; }
      };
      using btree_type = BTree_size_helper<uint64_t, uint64_t, key_of_value>;
      size_t btree_size =
          btree_set.get_stats().inner_nodes * sizeof(btree_type::InnerNode) +
          btree_set.get_stats().leaves * sizeof(btree_type::LeafNode);
      printf("parallel sum took %lu microseconds was %lu: serial sum took "
             "%lu, "
             "got "
             "%lu size was %lu\n",
             end - start, sum, serial_end - serial_start, correct_sum,
             btree_size);
      btree_times.push_back(std::min((end - start), serial_end - serial_start));
    }
  }
  std::sort(btree_times.begin(), btree_times.end());
  printf("average after %zu iters was: \n %zu\n", iters,
         btree_times[iters / 2]);

  return true;
}

template <typename traits>
std::tuple<uint64_t, uint64_t>
batch_test(size_t num_elements_start, size_t batch_size, size_t num_bits = 40,
           size_t iters = 5, bool verify = false) {
  uint64_t insert_total = 0;
  uint64_t delete_total = 0;
  for (size_t i = 0; i < iters + 1; i++) {

    uint64_t start = 0;
    uint64_t end = 0;
    auto data = create_random_data_in_parallel<typename traits::key_type>(
        num_elements_start * 2, 1UL << num_bits);
    tlx::btree_set<typename traits::key_type> correct;
    if (verify) {
      correct.insert(data.begin(), data.end());
    }
    {
      CPMA<traits> pma;
      pma.insert_batch(data.data(), data.size() / 2);
      split_cnt.reset();
      size_cnt.reset();
      search_cnt.reset();
      search_steps_cnt.reset();
      start = get_usecs();
      for (auto *batch_start = data.data() + data.size() / 2;
           batch_start < data.data() + data.size(); batch_start += batch_size) {
        pma.insert_batch(batch_start, batch_size);
      }
      end = get_usecs();
      split_cnt.report();
      size_cnt.report();
      search_cnt.report();
      search_steps_cnt.report();
      if (i > 0) {
        insert_total += end - start;
      }

      if (!verify) {
        printf("batch_size = %lu, total sum = %lu, time = %lu\n", batch_size,
               pma.sum(), end - start);
      }
      if (verify) {
        if (pma_different_from_set(pma, correct)) {
          printf("bad pma\n");
          return {-1UL, -1UL};
        }
      }
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(data.data() + data.size() / 2, data.data() + data.size(), g);

      start = get_usecs();
      for (auto *batch_start = data.data() + data.size() / 2;
           batch_start < data.data() + data.size(); batch_start += batch_size) {
        pma.remove_batch(batch_start, batch_size);
      }
      end = get_usecs();
      if (i > 0) {
        delete_total += end - start;
      }
    }
  }
  return {insert_total / iters, delete_total / iters};
}

// make a structure by adding elements in batches
// random numbers are not generated before hand to make it nicer to the cache
// do not time this function, only use it for cache numbers
template <typename traits>
uint64_t build_by_batch_for_cache_count(size_t total_elements,
                                        size_t batch_size,
                                        size_t num_bits = 40) {

  CPMA<traits> pma;
  for (size_t i = 0; i < total_elements; i += batch_size) {
    auto data = create_random_data<typename traits::key_type>(batch_size,
                                                              1UL << num_bits);
    pma.insert_batch(data.data(), data.size());
  }
  return pma.get_size();
}

template <typename traits>
bool batch_bench(size_t num_elements_start, size_t num_bits = 40,
                 size_t iters = 5, bool verify = false) {
  std::map<size_t, uint64_t> insert_total;
  std::map<size_t, uint64_t> delete_total;

  for (size_t batch_size = 1; batch_size < num_elements_start;
       batch_size *= 10) {
    auto results = batch_test<traits>(num_elements_start, batch_size, num_bits,
                                      iters, verify);
    insert_total[batch_size] = std::get<0>(results);
    delete_total[batch_size] = std::get<1>(results);
  }

  for (size_t batch_size = 1; batch_size < num_elements_start;
       batch_size *= 10) {
    printf("%lu, %lu, %lu\n", batch_size, insert_total[batch_size] / iters,
           delete_total[batch_size] / iters);
  }
  return false;
}

template <class Set, class F, class key_type>
void map_range(const Set &set, F f, key_type s, key_type t) {
  constexpr bool has_map_range = requires(const Set &g) {
    g.map_range(f, s, t);
  };
  if constexpr (has_map_range) {
    set.map_range(f, s, t);
  } else {

    auto it = set.lower_bound(s);
    while (it != set.end() && *it < t) {
      f(*it);
      it++;
    }
  }
}

template <class Set>
bool map_range_bench(size_t num_elements_start, uint64_t num_ranges,
                     uint64_t num_sizes, size_t num_bits, size_t iters,
                     std::seed_seq &seed1, std::seed_seq &seed2) {
  using T = typename Set::key_type;
  std::vector<std::vector<uint64_t>> times(num_sizes);
  auto data_to_insert = create_random_data_with_seed<T>(
      num_elements_start * iters, 1UL << num_bits, seed1);

  for (size_t it = 0; it < iters; it++) {

    uint64_t start = 0;
    uint64_t end = 0;
    Set s(data_to_insert.data() + (it * num_elements_start),
          data_to_insert.data() + ((it + 1) * num_elements_start));
    uint64_t log_size_of_range = 0;
    while (log_size_of_range < num_sizes) {
      uint64_t size_of_range = 1UL << log_size_of_range;
      auto range_starts = create_random_data_with_seed<uint64_t>(
          num_ranges, (1UL << num_bits) - size_of_range, seed2);
      ParallelTools::Reducer_sum<uint64_t> sum_of_counts;
      ParallelTools::Reducer_sum<uint64_t> sum_of_vals;
      start = get_usecs();
      ParallelTools::parallel_for(0, num_ranges, [&](uint64_t i) {
        uint64_t num_in_range = 0;
        uint64_t sum_in_range = 0;
        map_range(
            s,
            [&num_in_range, &sum_in_range]([[maybe_unused]] auto el) {
              num_in_range += 1;
              sum_in_range += el;
            },
            range_starts[i], range_starts[i] + size_of_range);
        sum_of_counts.add(num_in_range);
        sum_of_vals.add(sum_in_range);
      });
      end = get_usecs();
      printf("the sum of the counts from the ranges was %lu took %lu "
             "range_size was %lu, sum of elements was %lu\n",
             sum_of_counts.get(), end - start, size_of_range,
             sum_of_vals.get());
      times[log_size_of_range].push_back(end - start);
      log_size_of_range += 1UL;
    }
  }
  for (uint64_t i = 0; i < num_sizes; i++) {
    std::sort(times[i].begin(), times[i].end());
    printf("for size %lu, median time was %lu\n", 1UL << i,
           times[i][iters / 2]);
  }
  return true;
}

template <typename traits>
bool find_bench(size_t num_elements_start, uint64_t num_searches,
                size_t num_bits = 40, size_t iters = 5, bool verify = false) {
  std::vector<size_t> find_times;
  for (size_t i = 0; i < iters; i++) {

    uint64_t start = 0;
    uint64_t end = 0;
    std::random_device rd1;
    std::seed_seq seed1{rd1()};
    // auto data_to_insert = create_random_data<uint64_t>(num_elements_start,
    //                                                    1UL << num_bits,
    //                                                    seed1);
    auto data_to_insert =
        create_random_data_in_parallel<typename traits::key_type>(
            num_elements_start, 1UL << num_bits);

    // std::random_device rd2;
    // std::seed_seq seed2{rd2()};
    // auto data_to_search =
    //     create_random_data<uint64_t>(num_searches, 1UL << num_bits, seed2);
    auto data_to_search =
        create_random_data_in_parallel<typename traits::key_type>(
            num_searches, 1UL << num_bits);
    std::unordered_set<uint64_t> correct;
    uint64_t correct_num_contains = 0;
    if (verify) {
      correct.insert(data_to_insert.begin(), data_to_insert.end());
      ParallelTools::Reducer_sum<uint64_t> number_contains;
      ParallelTools::parallel_for(0, data_to_search.size(), [&](uint64_t i) {
        number_contains.add(correct.contains(data_to_search[i]));
      });
      correct_num_contains = number_contains.get();
    }
    CPMA<traits> pma;
    pma.insert_batch(data_to_insert.data(), data_to_insert.size());
    ParallelTools::Reducer_sum<uint64_t> number_contains;
    start = get_usecs();
    ParallelTools::parallel_for(0, data_to_search.size(), [&](uint64_t i) {
      number_contains.add(pma.has(data_to_search[i]));
    });

    end = get_usecs();
    uint64_t num_contains = number_contains.get();
    find_times.push_back(end - start);

    if (!verify) {
      printf("found %lu elements, pma had %lu elements, the heads took %lu "
             "bytes, total took %lu bytes, %lu\n",
             num_contains, pma.get_element_count(),
             pma.get_head_structure_size(), pma.get_size(), end - start);
    }
    if (verify) {
      if (correct_num_contains != num_contains) {
        printf("something wrong with the finds, we found %lu, while the "
               "correct found %lu\n",
               num_contains, correct_num_contains);
      }
    }
  }
  std::sort(find_times.begin(), find_times.end());
  size_t find_time = 0;
  find_time = find_times[iters / 2];
  printf("median_find_time = %lu\n", find_time);
  return false;
}

bool find_bench_tlx_btree(size_t num_elements_start, uint64_t num_searches,
                          size_t num_bits = 40, size_t iters = 5,
                          bool verify = false) {
  std::vector<size_t> uncompressed_find_times;
  for (size_t i = 0; i < iters; i++) {

    uint64_t start;
    uint64_t end;
    std::random_device rd1;
    std::seed_seq seed1{rd1()};
    // auto data_to_insert = create_random_data<uint64_t>(num_elements_start,
    //                                                    1UL << num_bits,
    //                                                    seed1);
    auto data_to_insert = create_random_data_in_parallel<uint64_t>(
        num_elements_start, 1UL << num_bits);

    // std::random_device rd2;
    // std::seed_seq seed2{rd2()};
    // auto data_to_search =
    //     create_random_data<uint64_t>(num_searches, 1UL << num_bits, seed2);
    auto data_to_search =
        create_random_data_in_parallel<uint64_t>(num_searches, 1UL << num_bits);
    std::unordered_set<uint64_t> correct;
    uint64_t correct_num_contains = 0;
    if (verify) {
      correct.insert(data_to_insert.begin(), data_to_insert.end());
      ParallelTools::Reducer_sum<uint64_t> number_contains;
      ParallelTools::parallel_for(0, data_to_search.size(), [&](uint64_t i) {
        number_contains.add(correct.contains(data_to_search[i]));
      });
      correct_num_contains = number_contains.get();
    }
    {
      // std::sort(data_to_insert.begin(), data_to_insert.end());

      tlx::btree_set<uint64_t> btree_set;

      // btree_set.bulk_load(data_to_insert.begin(), data_to_insert.end());
      for (const auto &el : data_to_insert) {
        btree_set.insert(el);
      }

      ParallelTools::Reducer_sum<uint64_t> number_contains;
      start = get_usecs();
      ParallelTools::parallel_for(0, data_to_search.size(), [&](uint64_t i) {
        number_contains.add(btree_set.exists(data_to_search[i]));
      });

      end = get_usecs();
      uint64_t num_contains = number_contains.get();
      uncompressed_find_times.push_back(end - start);

      if (!verify) {
        struct key_of_value {
          static const uint64_t &get(const uint64_t &v) { return v; }
        };
        using btree_type = BTree_size_helper<uint64_t, uint64_t, key_of_value>;
        size_t btree_size =
            btree_set.get_stats().inner_nodes * sizeof(btree_type::InnerNode) +
            btree_set.get_stats().leaves * sizeof(btree_type::LeafNode);
        printf("found %lu elements, memory size was %lu, %lu\n", num_contains,
               btree_size, end - start);
      }
      if (verify) {
        if (correct_num_contains != num_contains) {
          printf("something wrong with the finds, we found %lu, while the "
                 "correct found %lu\n",
                 num_contains, correct_num_contains);
        }
      }
    }
  }
  std::sort(uncompressed_find_times.begin(), uncompressed_find_times.end());
  size_t uncompressed_time = uncompressed_find_times[iters / 2];
  printf("mean_uncompressed_find_time = %lu, compressed_find_time = 0\n",
         uncompressed_time);
  return false;
}

template <class Set, class F, class key_type>
void map_range_length(const Set &set, F f, key_type s, uint64_t length) {
  constexpr bool has_map_range_length = requires(const Set &g) {
    g.map_range_length(f, s, length);
  };
  if constexpr (has_map_range_length) {
    set.map_range_length(f, s, length);
  } else {
    auto it = set.lower_bound(s);
    uint64_t count = 0;
    while (it != set.end() && count < length) {
      f(*it);
      it++;
      count += 1;
    }
  }
}

template <class Set>
void ycsb_insert_bench(std::vector<bool> &operations,
                       std::vector<typename Set::key_type> &values) {

  Set s(values.data(), values.data() + values.size() / 2);
  uint64_t count_found = 0;
  uint64_t start = get_usecs();
  for (uint64_t i = values.size() / 2; i < values.size(); i++) {
    if (operations[i]) {
      s.insert(values[i]);
    } else {
      count_found += s.exists(values[i]);
    }
  }
  uint64_t end = get_usecs();
  printf("found %lu elemnts in %lu microseconds\n", count_found, end - start);
}

template <class Set>
void ycsb_scan_bench(std::vector<typename Set::key_type> values,
                     std::vector<typename Set::key_type> ranges) {

  Set s(values.data(), values.data() + values.size() / 2);
  uint64_t total_sum = 0;
  uint64_t start = get_usecs();
  for (uint64_t i = values.size() / 2; i < values.size(); i++) {
    if (ranges[i] == 0) {
      s.insert(values[i]);
    } else {
      map_range_length(
          s, [&total_sum](auto el) { total_sum += el; }, values[i], ranges[i]);
    }
  }
  uint64_t end = get_usecs();
  printf("total sum was %lu in %lu microseconds\n", total_sum, end - start);
}

template <class Set>
void simple_rank_test(uint64_t num_elements, uint64_t batch_size = 1) {
  std::vector<typename Set::key_type> values(num_elements);
  for (uint64_t i = 0; i < num_elements; i++) {
    values[i] = i + 1;
  }
  std::random_device rd;
  std::mt19937 g(0);
  // std::mt19937 g(rd());
  std::shuffle(values.begin(), values.end(), g);
  Set s;
  uint64_t start = get_usecs();
  if (batch_size == 1) {
    for (auto v : values) {
      s.insert(v);
    }
  } else {
    uint64_t index = 0;
    while (index + batch_size < values.size()) {
      s.insert_batch(values.data() + index, batch_size);
      index += batch_size;
    }
    s.insert_batch(values.data() + index, values.size() - index);
  }
  uint64_t end = get_usecs();
  printf("inserting the elements keeping track of the ranks took %lu\n",
         end - start);
  start = get_usecs();
  for (auto v : values) {
    uint64_t rank = s.rank(v);
    if (rank != v - 1) {
      printf("got the wrong rank, element %lu, got rank %lu\n", (uint64_t)v,
             rank);
    }
  }
  end = get_usecs();
  printf("getting the rank of each element took %lu\n", end - start);
  // s.print_pma();

  start = get_usecs();
  for (auto v : values) {
    auto el = s.select(v - 1);
    if (el != v) {
      printf("got the wrong element, index %lu, got element %lu\n",
             (uint64_t)v - 1, (uint64_t)el);
    }
  }
  end = get_usecs();
  printf("selecting the elements by rank took %lu\n", end - start);
  // s.print_pma();
}

template <class Set> void simple_rank_insert_test(uint64_t num_elements) {
  std::vector<typename Set::key_type> values(num_elements);
  for (uint64_t i = 0; i < num_elements; i++) {
    values[i] = i + 1;
  }
  std::random_device rd;
  std::mt19937 g(0);
  // std::mt19937 g(rd());
  std::shuffle(values.begin(), values.end(), g);
  Set pma;
  std::vector<uint64_t> correct;

  uint64_t start = get_usecs();
  for (auto v : values) {
    uint64_t rank = g() % (correct.size() + 1);
    // printf("inserting %lu into position %lu\n", v, rank);
    pma.insert_by_rank(v, rank);
    correct.insert(correct.begin() + rank, v);
    // pma.print_pma();
  }
  uint64_t end = get_usecs();
  printf("inserting the elements by ranks took %lu\n", end - start);

  uint64_t counter = 0;
  pma.template map<true>([&counter, &correct](uint64_t element) {
    if (element != correct[counter++]) {
      printf("something is in the wrong position\n");
    }
  });

  // pma.print_pma();
  // std::cout << "correct vector:";
  // for (auto el : correct) {
  //   std::cout << el << ",";
  // }
  // std::cout << "\n";
}

void simple_key_value_test(uint64_t num_elements) {
  auto keys = create_random_data<uint64_t>(
      num_elements, std::numeric_limits<uint64_t>::max() - 1);
  auto values = create_random_data<uint32_t>(
      num_elements, std::numeric_limits<uint32_t>::max() - 1);

  PMAkv<uint64_t, uint32_t> test;
  uint64_t unique = 0;
  uint64_t start = get_usecs();
  for (uint64_t i = 0; i < num_elements; i++) {
    unique += test.insert_or_update(keys[i], values[i]);
  }
  uint64_t end = get_usecs();
  printf("inserting the elements took %lu, %lu unique\n", end - start, unique);
  uint64_t found = 0;
  start = get_usecs();

  for (uint64_t i = 0; i < num_elements; i++) {
    auto [had, val] = test.get(keys[i]);
    if (had) {
      found += val == values[i];
    }
  }
  end = get_usecs();
  printf("finding the elements took %lu, %lu found\n", end - start, found);
}

void simple_key_value_test_map(uint64_t num_elements) {
  auto keys = create_random_data<uint64_t>(
      num_elements, std::numeric_limits<uint64_t>::max() - 1);
  auto values = create_random_data<uint32_t>(
      num_elements, std::numeric_limits<uint32_t>::max() - 1);

  tlx::btree_map<uint64_t, uint32_t> test;
  uint64_t unique = 0;
  uint64_t start = get_usecs();
  for (uint64_t i = 0; i < num_elements; i++) {
    // auto p = test.insert_or_assign(keys[i], values[i]);
    // unique += p.second;
    test[keys[i]] = values[i];
  }
  uint64_t end = get_usecs();
  printf("inserting the elements took %lu, %lu unique\n", end - start, unique);
  uint64_t found = 0;
  start = get_usecs();

  for (uint64_t i = 0; i < num_elements; i++) {
    auto it = test.find(keys[i]);
    if (it != test.end()) {
      found += (*it).second == values[i];
    }
  }
  end = get_usecs();
  printf("finding the elements took %lu, %lu found\n", end - start, found);
}
*/
