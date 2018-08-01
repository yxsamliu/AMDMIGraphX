#find_package(OpenBLAS)

find_library(OPENBLAS_LIB 
    NAMES openblas
    HINTS "/opt/OpenBLAS/lib")

add_library(migraph_cpu
    cpu_target.cpp
    cpu_lowering.cpp
)
rocm_clang_tidy_check(migraph_cpu)
# target_link_libraries(migraph_cpu migraph OpenBLAS::OpenBLAS)
include_directories(SYSTEM /opt/OpenBLAS/include)
target_link_libraries(migraph_cpu migraph ${OPENBLAS_LIB} pthread)
target_include_directories(migraph_cpu PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
