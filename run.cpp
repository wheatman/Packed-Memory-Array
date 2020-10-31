#include "CPMA.hpp"
#include "leaf.hpp"
#include "test.hpp"

#include <iostream>

#if !defined(KEY_TYPE)
#define KEY_TYPE uint64_t
#endif
using key_type = KEY_TYPE;

#if !defined(LEAFFORM)
#define LEAFFORM delta_compressed
#endif
#define LEAFFORM2(form) form##_leaf<key_type>
#define LEAFFORM3(form) LEAFFORM2(form)
using leaf = LEAFFORM3(LEAFFORM);

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

using traits = PMA_traits<leaf, head_form, B_size, store_density>;
using PMA_Type = CPMA<traits>;

int main(int32_t argc, char *argv[]) {

  if (std::string("cpma_helper") == argv[1]) {
    std::seed_seq seed{0};
    uint64_t max_size = std::strtol(argv[2], nullptr, 10);
    uint64_t start_batch_size = std::strtol(argv[3], nullptr, 10);
    uint64_t end_batch_size = std::strtol(argv[4], nullptr, 10);
    uint64_t trials = std::strtol(argv[5], nullptr, 10);

    timing_cpma_helper<traits>(max_size, start_batch_size, end_batch_size,
                               trials, seed);
  }
  if (std::string("v_leaf") == argv[1]) {
    // bool verify_leaf(uint32_t size, uint32_t num_ops, uint32_t range_start,
    // uint32_t range_end)
    if (verify_leaf<leaf>(sizeof(key_type) * 4 * 32, 10, 1, 10)) {
      verify_leaf<leaf>(sizeof(key_type) * 4 * 32, 10, 1, 10, 2);
      return 1;
    }
    if (verify_leaf<leaf>(sizeof(key_type) * 64 * 32, 200, 1, 100)) {
      verify_leaf<leaf>(sizeof(key_type) * 64 * 32, 200, 1, 100, 2);
      return 1;
    }
  }
  if (std::string("verify") == argv[1]) {

    if (verify_cpma_different_sizes<traits>(
            {{100, false}, {1000, false}, {10000, false}, {20000, true}})) {
      return 1;
    }
  }

  if (std::string("v_batch") == argv[1]) {

    uint64_t num_elements_start = 100000;
    if (argc == 3) {
      num_elements_start = atoll(argv[2]);
    }

    std::cout << "batch bench test\n";
    if (batch_bench<traits>(num_elements_start, 40, 5, true)) {
      return 1;
    }
  }
  if (std::string("profile") == argv[1]) {
    if (argc != 3) {
      std::cout << "specify 2 arguments for the profile" << std::endl;
    }

    std::seed_seq s;
    test_cpma_unordered_insert<traits>(atoll(argv[2]), s, 7);

    test_tlx_btree_ordered_insert<key_type>(atoll(argv[2]));
  }

  if (std::string("sizes") == argv[1]) {
    std::cout << "sizeof(PMA_Type) == " << sizeof(PMA_Type) << std::endl;
    if (argc == 3) {
      std::random_device r;
      std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
      std::cout << "############ sizes ############" << std::endl;
      test_cpma_size<traits>(std::strtol(argv[2], nullptr, 10), seed);
    }
  }

  if (std::string("sizes_file") == argv[1]) {
    if (argc < 3) {
      std::cout << "specify at least 2 arguments for the test" << std::endl;
    }
    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};

    test_cpma_size_file_out_simple<traits>(std::strtol(argv[2], nullptr, 10),
                                           seed, "uncompressed_sizes_1.1.csv");
  }
  if constexpr (std::is_same_v<key_type, uint64_t>) {
    if (std::string("graph") == argv[1]) {
      real_graph<traits>(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
    }
  }
  if (std::string("scan") == argv[1]) {
    uint64_t trials = 11;
    if (argc == 5) {
      trials = atoll(argv[4]);
    }
    scan_bench<traits>(atoll(argv[2]), atoll(argv[3]), trials);
  }

  if (std::string("batch_bench") == argv[1]) {
    batch_bench<traits>(atoll(argv[2]), 40, 10);
  }
  if (std::string("batch") == argv[1]) {

    uint64_t trials = 10;
    if (argc == 5) {
      trials = atoll(argv[4]);
    }
    auto results =
        batch_test<traits>(atoll(argv[2]), atoll(argv[3]), 40, trials);
    std::cout << (double)std::get<0>(results) / 1000000 << ", "
              << (double)std::get<1>(results) / 1000000 << "\n";
  }
  if (std::string("single") == argv[1]) {
    std::seed_seq s;
    uint64_t max_size = atoll(argv[2]);
    uint64_t trials = 5;
    if (argc == 4) {
      trials = atoll(argv[3]);
    }

    test_cpma_unordered_insert<traits>(max_size, s, trials);

    // test_tlx_btree_unordered_insert<key_type>(atoll(argv[2]), s, trials);
  }

  if (std::string("single_seq") == argv[1]) {
    std::seed_seq s;
    test_cpma_ordered_insert<traits>(atoll(argv[2]));
    test_tlx_btree_ordered_insert<key_type>(atoll(argv[2]));
  }
  if (std::string("single_alt") == argv[1]) {
    std::seed_seq s;
    uint64_t num_items = atoll(argv[2]);
    uint64_t percent_ordered = atoll(argv[3]);

    test_cpma_ordered_and_unordered_insert<traits>(num_items, percent_ordered,
                                                   s, 5);

    test_btree_ordered_and_unordered_insert<key_type>(num_items,
                                                      percent_ordered, s, 5);
  }

  if (std::string("multi_seq") == argv[1]) {
    uint64_t num_items = atoll(argv[2]);
    uint64_t groups = atoll(argv[3]);

    test_cpma_multi_seq_insert<traits>(num_items, groups);

    test_btree_multi_seq_insert<key_type>(num_items, groups);
  }

  if (std::string("bulk") == argv[1]) {
    uint64_t num_items = atoll(argv[2]);
    uint64_t num_per = atoll(argv[3]);

    test_cpma_bulk_insert<traits>(num_items, num_per);

    test_btree_bulk_insert<key_type>(num_items, num_per);
  }
  if (std::string("find") == argv[1]) {
    find_bench<traits>(atoll(argv[2]), atoll(argv[3]), 40, 5, false);

    find_bench_tlx_btree(atoll(argv[2]), atoll(argv[3]), 40, 5, false);
  }
  if (std::string("map_range") == argv[1]) {
    uint64_t num_elements_start = atoll(argv[2]);
    uint64_t num_ranges = atoll(argv[3]);
    uint64_t max_log_range_size = atoll(argv[4]);

    uint64_t trials = 5;
    if (argc == 6) {
      trials = atoll(argv[5]);
    }

    std::random_device r;
    std::seed_seq seed1{0};
    std::seed_seq seed2{1};

    map_range_bench<CPMA<traits>>(num_elements_start, num_ranges,
                                  max_log_range_size, 40, trials, seed1, seed2);

    // map_range_bench<tlx::btree_set<uint64_t>>(num_elements_start, num_ranges,
    //                                           max_log_range_size, 40, trials,
    //                                           seed1, seed2);
  }

  if (std::string("map_range_single") == argv[1]) {
    uint64_t num_elements_start = atoll(argv[2]);
    uint64_t num_ranges = atoll(argv[3]);
    uint64_t range_size = atoll(argv[4]);

    uint64_t trials = 5;
    if (argc == 6) {
      trials = atoll(argv[5]);
    }

    std::random_device r;
    std::seed_seq seed1{0};
    std::seed_seq seed2{1};

    map_range_single<CPMA<traits>>(num_elements_start, num_ranges, range_size,
                                   40, trials, seed1, seed2);

    // map_range_bench<tlx::btree_set<uint64_t>>(num_elements_start, num_ranges,
    //                                           max_log_range_size, 40, trials,
    //                                           seed1, seed2);
  }

  if (std::string("map_range_cache") == argv[1]) {
    uint64_t num_elements_start = atoll(argv[2]);
    uint64_t num_ranges = atoll(argv[3]);
    uint64_t range_size = atoll(argv[4]);

    std::random_device r;
    std::seed_seq seed1{0};
    std::seed_seq seed2{1};

    map_range_test<CPMA<traits>>(num_elements_start, num_ranges, range_size, 40,
                                 seed1, seed2);
  }

  if (std::string("ycsb_a") == argv[1]) {
    long length;
    char *S = readStringFromFile(argv[2], &length);
    words W = stringToWords(S, length);
    printf("got %ld words\n", W.m);
    std::vector<bool> operations(W.m / 2);
    std::vector<typename traits::key_type> values(W.m / 2);
    uint64_t insert_count = 0;
    uint64_t read_count = 0;
    for (int64_t i = 0; i < W.m / 2; i++) {
      if (W.Strings[2 * i] == std::string("INSERT")) {
        insert_count += 1;
        operations[i] = true;
        values[i] = atoll(W.Strings[2 * i + 1]);
      } else if (W.Strings[2 * i] == std::string("READ")) {
        read_count += 1;
        operations[i] = false;
        values[i] = atoll(W.Strings[2 * i + 1]);
      } else {
        printf("something is wrong\n");
      }
    }
    printf("got %lu inserts and %lu reads\n", insert_count, read_count);

    ycsb_insert_bench<CPMA<traits>>(operations, values);
    ycsb_insert_bench<tlx::btree_set<typename traits::key_type>>(operations,
                                                                 values);
  }

  if (std::string("ycsb_e") == argv[1]) {
    long length;
    char *S = readStringFromFile(argv[2], &length);
    words W = stringToWords(S, length);
    printf("got %ld words\n", W.m);
    std::vector<typename traits::key_type> values(W.m / 3);
    std::vector<typename traits::key_type> ranges(W.m / 3);
    uint64_t insert_count = 0;
    uint64_t scan_count = 0;
    for (int64_t i = 0; i < W.m / 3; i++) {
      if (W.Strings[3 * i] == std::string("INSERT")) {
        insert_count += 1;
        ranges[i] = 0;
        values[i] = atoll(W.Strings[3 * i + 1]);
      } else if (W.Strings[3 * i] == std::string("SCAN")) {
        scan_count += 1;
        values[i] = atoll(W.Strings[3 * i + 1]);
        ranges[i] = atoll(W.Strings[3 * i + 2]);
      } else {
        printf("something is wrong\n");
      }
    }
    printf("got %lu inserts and %lu scans\n", insert_count, scan_count);

    ycsb_scan_bench<CPMA<traits>>(values, ranges);
    ycsb_scan_bench<tlx::btree_set<typename traits::key_type>>(values, ranges);
  }

  if (std::string("batch_build") == argv[1]) {
    uint64_t size =
        build_by_batch_for_cache_count<traits>(atoll(argv[2]), atoll(argv[3]));
    std::cout << "end size was " << size << "\n";
  }

  if (std::string("batch_time") == argv[1]) {
    double seconds =
        build_by_batch_for_time<traits>(atoll(argv[2]), atoll(argv[3]), 5);
    std::cout << "mean time " << seconds << "\n";
  }
}
