
#include <limits>
#include <span>

#include "PMA/CPMA.hpp"
#include "PMA/PCSR.hpp"

#include "EdgeMapVertexMap/algorithms/BC.h"
#include "EdgeMapVertexMap/algorithms/BFS.h"
#include "EdgeMapVertexMap/algorithms/Components.h"
#include "EdgeMapVertexMap/algorithms/PageRank.h"
#include "EdgeMapVertexMap/include/EdgeMapVertexMap/internal/io_util.hpp"
#include "EdgeMapVertexMap/include/EdgeMapVertexMap/internal/utils.hpp"

#include "include/PMA/internal/rmat_util.hpp"

#include "parlay/internal/group_by.h"
#include "parlay/primitives.h"

#if !defined(KEY_TYPE)
#define KEY_TYPE uint32_t
#endif
using key_type = KEY_TYPE;

#if !defined(LEAFFORM)
#define LEAFFORM uncompressed
#endif
#define LEAFFORM2(form) form##_leaf<key_type>
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
                PCSR_Type::node_t start_node = 0) {
  PCSR_Type::node_t num_nodes = 0;
  uint64_t num_edges = 0;
  auto edges = EdgeMapVertexMap::get_edges_from_file_adj<PCSR_Type::node_t>(
      filename, &num_edges, &num_nodes, true);

  printf("done reading in the file, n = %lu, m = %lu\n",
         static_cast<uint64_t>(num_nodes), num_edges);
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

  int32_t parallel_bfs_result2_ = 0;
  uint64_t parallel_bfs_time2 = 0;

  for (int i = 0; i < iters; i++) {
    start = get_usecs();
    auto *parallel_bfs_result = EdgeMapVertexMap::BFS(g, start_node);
    end = get_usecs();
    parallel_bfs_result2_ += parallel_bfs_result[0];
    if (i == 0 && parallel_bfs_result != nullptr) {
      uint64_t reached = 0;
      for (PCSR_Type::node_t j = 0; j < num_nodes; j++) {
        reached += parallel_bfs_result[j] != -1;
      }
      printf("the bfs from source %lu, reached %lu vertices\n",
             static_cast<uint64_t>(start_node), reached);
      std::vector<PCSR_Type::node_t> depths(
          num_nodes, std::numeric_limits<PCSR_Type::node_t>::max());
      ParallelTools::parallel_for(0, num_nodes, [&](PCSR_Type::node_t j) {
        PCSR_Type::node_t current_depth = 0;
        int64_t current_parent = j;
        if (parallel_bfs_result[j] < 0) {
          return;
        }
        while (current_parent != parallel_bfs_result[current_parent]) {
          current_depth += 1;
          current_parent = parallel_bfs_result[current_parent];
        }
        depths[j] = current_depth;
      });
      EdgeMapVertexMap::write_array_to_file("bfs.out", depths.data(),
                                            num_nodes);
    }

    free(parallel_bfs_result);
    parallel_bfs_time2 += (end - start);
  }
  // printf("bfs took %lums, parent of 0 = %d\n", (bfs_time)/(1000*iters),
  // bfs_result_/iters);
  printf("parallel_bfs with edge_map took %lums, parent of 0 = %d\n",
         parallel_bfs_time2 / (1000 * iters), parallel_bfs_result2_ / iters);
  printf("F-Graph, %d, BFS, %lu, %s, ##, %f\n", iters,
         static_cast<uint64_t>(start_node), filename.c_str(),
         (double)parallel_bfs_time2 / (iters * 1000000));
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
  printf("F-Graph, %d, PageRank, %lu, %s, ##, %f\n", iters,
         static_cast<uint64_t>(start_node), filename.c_str(),
         (double)pagerank_time / (iters * 1000000));
  EdgeMapVertexMap::write_array_to_file("pr.out", values3, num_nodes);
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

  printf("F-Graph, %d, BC, %lu, %s, ##, %f\n", iters,
         static_cast<uint64_t>(start_node), filename.c_str(),
         (double)bc_time / (iters * 1000000));
  if (values4 != nullptr) {
    EdgeMapVertexMap::write_array_to_file("bc.out", values4, num_nodes);
    free(values4);
  }

  PCSR_Type::node_t *values5 = nullptr;
  PCSR_Type::node_t id_0 = 0;
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

  printf("CC took %lums, value of 0 = %lu\n", cc_time / (1000 * iters),
         static_cast<uint64_t>(id_0) / iters);
  printf("F-Graph, %d, Components, %lu, %s, ##, %f\n", iters,
         static_cast<uint64_t>(start_node), filename.c_str(),
         (double)cc_time / (iters * 1000000));
  if (values5 != nullptr) {
    auto counts =
        parlay::histogram_by_key(parlay::slice(values5, values5 + num_nodes));

    printf("there are %zu components\n", counts.size());
    auto max_it = parlay::max_element(counts, [](const auto &l, const auto &r) {
      return l.second < r.second;
    });

    printf("the element with the biggest component is %lu, it has %lu members "
           "to its component\n",
           (uint64_t)max_it->first, max_it->second);
    EdgeMapVertexMap::write_array_to_file("cc.out", values5, num_nodes);
  }

  free(values5);

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
        std::vector<std::tuple<PCSR_Type::node_t, PCSR_Type::node_t>> es(
            b_size);
        ParallelTools::parallel_for(0, b_size, [&](uint64_t i) {
          // rmat breaks if the batch is bigger than a uint32_t
          assert(i < std::numeric_limits<uint32_t>::max());
          std::pair<PCSR_Type::node_t, PCSR_Type::node_t> edge =
              rmat((uint32_t)i);
          es[i] = {edge.first, edge.second};
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