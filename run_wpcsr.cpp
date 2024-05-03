
#include <span>

#include "PMA/CPMA.hpp"
#include "PMA/PCSR.hpp"

#include "EdgeMapVertexMap/algorithms/BellmanFord.h"
#include "EdgeMapVertexMap/include/EdgeMapVertexMap/internal/io_util.hpp"

#include "include/PMA/internal/rmat_util.hpp"

#include "parlay/internal/group_by.h"
#include "parlay/primitives.h"

#if !defined(KEY_TYPE)
#define KEY_TYPE uint32_t
#endif
using key_type = KEY_TYPE;

#if !defined(WEIGHT_TYPE)
#define WEIGHT_TYPE uint32_t
#endif
using weight_type = WEIGHT_TYPE;

#if !defined(LEAFFORM)
#define LEAFFORM uncompressed
#endif
#define LEAFFORM2(form) form##_leaf<key_type, weight_type>
#define LEAFFORM3(form) LEAFFORM2(form)
using leaf = LEAFFORM3(LEAFFORM);

#if !defined(HEADFORM)
#define HEADFORM InPlace
#endif
static constexpr HeadForm head_form = HEADFORM;
static constexpr uint64_t B_size = (head_form == BNary) ? 17 : 0;

static constexpr bool store_density = false;
static constexpr bool support_rank = false;

using traits = PMA_traits<leaf, head_form, B_size, store_density, support_rank,
                          false, 0, true, true>;
using PCSR_Type = PCSR<traits>;

bool real_graph(const std::string &filename, int iters = 20,
                uint32_t start_node = 0) {
  PCSR_Type::node_t num_nodes = 0;
  uint64_t num_edges = 0;
  auto edges = EdgeMapVertexMap::get_edges_from_file_adj<key_type, weight_type>(
      filename, &num_edges, &num_nodes, true);

  printf("done reading in the file, n = %lu, m = %lu\n", (uint64_t)num_nodes,
         num_edges);
  if (start_node >= num_nodes) {
    std::cerr << "start node is greater than the number of nodes\n";
    std::cerr << "start node is " << start_node << "\n";
    return false;
  }

  auto start = get_usecs();
  PCSR_Type g(num_nodes, edges);
  // for (auto edge : edges) {
  //   g.insert(edge.first, edge.second);
  // }
  auto end = get_usecs();
  // g.print();
  g.write_adj_file("del.adj");
  // return true;
  printf("inserting the edges took %lums\n", (end - start) / 1000);
  num_nodes = g.num_nodes();
  int64_t size = g.get_size();
  printf("size = %lu bytes, num_edges = %lu, num_nodes = %lu\n", size,
         g.num_edges(), (uint64_t)g.num_nodes());

  int32_t bf_result2_ = 0;
  uint64_t parallel_bf_time2 = 0;

  for (int i = 0; i < iters; i++) {
    start = get_usecs();
    auto *bf_out = EdgeMapVertexMap::BF(g, start_node);
    end = get_usecs();
    if (i == 0) {
      std::ofstream myfile;
      myfile.open("bf.out");
      for (unsigned int j = 0; j < num_nodes; j++) {
        myfile << bf_out[j] << "\n";
      }
      myfile.close();
    }
    bf_result2_ += bf_out[0];

    free(bf_out);
    parallel_bf_time2 += (end - start);
  }
  // printf("bfs took %lums, parent of 0 = %d\n", (bfs_time)/(1000*iters),
  // bfs_result_/iters);
  printf("BellmanFord took %lums, distance of 0 = %d\n",
         parallel_bf_time2 / (1000 * iters), bf_result2_ / iters);
  printf("F-Graph, %d, BF, %u, %s, ##, %f\n", iters, start_node,
         filename.c_str(), (double)parallel_bf_time2 / (iters * 1000000));

  if (true) {
    for (uint64_t b_size = 10; b_size <= 1000000; b_size *= 10) {
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
        auto rmat = rMat<PCSR_Type::node_t>(
            (PCSR_Type::node_t)nn, (PCSR_Type::node_t)r.ith_rand(it), a, b, c);
        std::vector<
            std::tuple<PCSR_Type::node_t, PCSR_Type::node_t, weight_type>>
            es(b_size);
        ParallelTools::parallel_for(0, b_size, [&](uint64_t i) {
          // rmat breaks if the batch is bigger than a uint32_t
          assert(i < std::numeric_limits<uint32_t>::max());
          std::pair<PCSR_Type::node_t, PCSR_Type::node_t> edge =
              rmat((uint32_t)i);
          es[i] = {edge.first, edge.second, i % 256};
        });
        auto delete_batch = es;

        start = get_usecs();
        g.insert_batch(es);
        end = get_usecs();
        static constexpr PCSR_Type::node_t top_bit =
            (std::numeric_limits<PCSR_Type::node_t>::max() >> 1) + 1;
        ParallelTools::parallel_for(0, b_size, [&](uint64_t i) {
          auto src = std::get<0>(es[i]);
          if (src >= top_bit) {
            src ^= top_bit;
          }
          if (!g.contains(src, std::get<1>(es[i]))) {
            std::cout << "missing something after insert\n";
          }
        });
        // printf("%lu\n", end - start);
        if (it > 0) {
          batch_insert_time += end - start;
        }
        // size = g.get_memory_size();
        // printf("size end = %lu\n", size);
        start = get_usecs();
        // for (const auto &[src, dest] : delete_batch) {
        //   g.remove(src, dest);
        // }
        g.remove_batch(delete_batch);
        end = get_usecs();
        ParallelTools::parallel_for(0, b_size, [&](uint64_t i) {
          auto src = std::get<0>(delete_batch[i]);
          if (src >= top_bit) {
            src ^= top_bit;
          }
          if (g.contains(src, std::get<1>(delete_batch[i]))) {
            std::cout << "have something after deletes\n";
          }
        });
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
      printf("%lu, %f, %f\n", b_size, batch_insert_time, batch_remove_time);
    }
  }

  return true;
}

int main([[maybe_unused]] int32_t argc, char *argv[]) {
  real_graph(argv[1], atoi(argv[2]), atoi(argv[3]));
}