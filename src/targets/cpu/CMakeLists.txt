find_package(OpenBLAS)

add_library(migraph_cpu
    cpu_target.cpp
    cpu_lowering.cpp
)
rocm_clang_tidy_check(migraph_cpu)
target_link_libraries(migraph_cpu PUBLIC migraph)
target_link_libraries(migraph_cpu PRIVATE OpenBLAS::OpenBLAS)
target_include_directories(migraph_cpu PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
