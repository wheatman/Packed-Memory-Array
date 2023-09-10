cc_library(
    name = "helpers",
    hdrs = ["helpers.hpp"],
)

cc_library(
    name = "timers",
    hdrs = ["timers.hpp"],
)

cc_library(
    name = "leaf",
    hdrs = ["leaf.hpp"],
    deps = [
        "helpers",
        "timers",
        "@ParallelTools//ParallelTools:parallel",
        "@ParallelTools//ParallelTools:reducer",
        "@soa//StructOfArrays:soa",
        "@soa//StructOfArrays:multipointer",
    ]
)

package(
    default_visibility = ["//visibility:public"],
)