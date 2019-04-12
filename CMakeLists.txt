cmake_minimum_required(VERSION 3.5)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "The binary and source directroy cannot be the same")
endif()

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT MSVC_IDE AND NOT CMAKE_BUILD_TYPE )
    set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

# Default installation path
if(WIN32)
    set(CMAKE_INSTALL_PREFIX "/opt/rocm/x86_64-w64-mingw32" CACHE PATH "")
else()
    set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "")
endif()

project(migraphx)
find_package(ROCM REQUIRED)

include(ROCMSetupVersion)

rocm_setup_version(VERSION 0.2)

option( BUILD_SHARED_LIBS "Build as a shared library" ON )

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.4")
        message(FATAL_ERROR "MIGraph requires at least gcc 5.4")
    endif()
endif()

if(CMAKE_CXX_COMPILER MATCHES ".*hcc")
    message(STATUS "Enable miopen backend")
    set(MIGRAPHX_ENABLE_GPU On CACHE BOOL "")
else()
    set(MIGRAPHX_ENABLE_GPU Off CACHE BOOL "")
endif()

set(MIGRAPHX_ENABLE_TF Off CACHE BOOL "")

add_compile_options(-std=c++14)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(EnableCompilerWarnings)
include(ROCMClangTidy)
rocm_enable_clang_tidy(
    CHECKS
        *
    ERRORS
        *
    HEADER_FILTER
        ".*hpp"
    EXTRA_ARGS
        -DMIGRAPHX_USE_CLANG_TIDY
        "-Dmain\\\\(...\\\\)=main\\\\(__VA_ARGS__\\\\) // NOLINT"

)
include(ROCMCppCheck)
rocm_enable_cppcheck(
    CHECKS
        warning
        style
        performance
        portability
    SUPPRESS 
        ConfigurationNotChecked
        unmatchedSuppression
        unusedFunction
    FORCE
    INCONCLUSIVE
    RULE_FILE
        ${CMAKE_CURRENT_SOURCE_DIR}/cppcheck.rules
    SOURCES
        src/
        test/
    INCLUDE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/targets/cpu/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/targets/miopen/include
        ${CMAKE_CURRENT_SOURCE_DIR}/test/include
    DEFINE
        CPPCHECK=1
)

enable_testing()

include(ROCMCreatePackage)
rocm_create_package(
    NAME MIGraphX
    DESCRIPTION "AMD's graph optimizer"
    MAINTAINER "Paul Fultz II <paul.fultz@amd.com>"
    LDCONFIG
    PTH
    DEPENDS miopen-hip rocblas hip_hcc half
)

add_subdirectory(src)
add_subdirectory(doc)
add_subdirectory(test)
add_subdirectory(tools)
