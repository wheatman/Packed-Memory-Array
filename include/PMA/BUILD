cc_library(
    name = "cpma",
    hdrs = ["CPMA.hpp"],
    deps = [
        "//PMA/internal:helpers",
        "//PMA/internal:timers",
        "//PMA/internal:leaf",
        "@ParallelTools//ParallelTools:parallel",
        "@ParallelTools//ParallelTools:reducer",
        "@ParallelTools//ParallelTools:concurrent_hash_map",
        "@ParallelTools//ParallelTools:flat_hash_map",
        "@ParallelTools//ParallelTools:sort",
        "@soa//StructOfArrays:soa",
        "@soa//StructOfArrays:multipointer",
        "@parlaylib//parlay:primitives",
    ]
)

cc_library(
    name = "pcsr",
    hdrs = ["PCSR.hpp"],
    deps = [
        "//PMA/internal:helpers",
        ":cpma",
        "//PMA/internal:leaf",
        "@ParallelTools//ParallelTools:parallel",
        "@parlaylib//parlay:primitives",
    ]
)

package(
    default_visibility = ["//visibility:public"],
)