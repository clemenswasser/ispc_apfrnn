cmake_minimum_required(VERSION 3.31)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  include(FetchContent)
  FetchContent_Declare(vcpkg
    URL [[https://github.com/microsoft/vcpkg/archive/refs/tags/2026.10.16.tar.gz]]
    CONFIGURE_COMMAND [[]])
  FetchContent_MakeAvailable(vcpkg)
  set(CMAKE_TOOLCHAIN_FILE "${vcpkg_SOURCE_DIR}/scripts/buildsystems/vcpkg.cmake")
endif()

project(ispc_apfrnn LANGUAGES CXX ISPC)

option(APFRNN_BUILD_TESTS "Build the doctest suite" ${PROJECT_IS_TOP_LEVEL})
option(APFRNN_BUILD_BENCHMARKS "Build the Google Benchmark targets" ${PROJECT_IS_TOP_LEVEL})
option(APFRNN_BUILD_BASELINES
  "Build the CompactNSearch/TreeNSearch comparison benchmarks" OFF)
# The SYCL backend needs the whole tree to be configured with icpx: one CMake
# tree has one C++ compiler, and CPU and GPU code are linked statically.
option(APFRNN_ENABLE_SYCL "Build the optional static SYCL GPU backend (requires icpx)" OFF)

find_package(TBB CONFIG REQUIRED)

# The ISPC kernels are a separate object library because CMake exports
# ISPC_HEADER_DIRECTORY as a relative include path to anything that links the
# target, which breaks consuming this project with add_subdirectory. The
# property is resolved against the top-level binary directory.
file(RELATIVE_PATH APFRNN_ISPC_HEADERS "${CMAKE_BINARY_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/ispc")
add_library(apfrnn_ispc_objects OBJECT
  src/kernels/cells.ispc
  src/kernels/rows.ispc
  src/kernels/csr.ispc
  src/kernels/sync.ispc)
# src/kernels_ispc.h includes each component's generated header.
set_target_properties(apfrnn_ispc_objects PROPERTIES
  ISPC_HEADER_DIRECTORY "${APFRNN_ISPC_HEADERS}")
target_compile_options(apfrnn_ispc_objects PRIVATE
  $<$<NOT:$<CONFIG:Debug>>:-O3>
  $<$<NOT:$<CONFIG:Release>>:-g>
  --opt=fast-math
  --addressing=64
  # store_compacted calls llvm.experimental.vector.compress on AVX-512.
  --enable-llvm-intrinsics)

add_library(ispc_apfrnn_kernels
  src/apfrnn.cpp
  src/neighbor_search.cpp
  src/plan.cpp
  src/sweep.cpp
  $<TARGET_OBJECTS:apfrnn_ispc_objects>)
add_dependencies(ispc_apfrnn_kernels apfrnn_ispc_objects)
target_include_directories(ispc_apfrnn_kernels PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/ispc")
target_sources(ispc_apfrnn_kernels PUBLIC
  FILE_SET HEADERS BASE_DIRS include FILES include/apfrnn/neighbor_search.hpp)
target_compile_features(ispc_apfrnn_kernels PUBLIC cxx_std_17)
target_link_libraries(ispc_apfrnn_kernels PUBLIC
  TBB::tbb TBB::tbbmalloc $<TARGET_NAME_IF_EXISTS:TBB::tbbmalloc_proxy>)

if(APFRNN_ENABLE_SYCL)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    message(FATAL_ERROR
      "APFRNN_ENABLE_SYCL=ON requires the Intel oneAPI C++ compiler. "
      "Configure with -DCMAKE_CXX_COMPILER=icpx after sourcing setvars.sh.")
  endif()
  find_package(IntelSYCL REQUIRED)

  add_library(apfrnn_sycl STATIC src/sycl/apfrnn_sycl.cpp)
  target_include_directories(apfrnn_sycl PUBLIC src/sycl)
  target_link_libraries(apfrnn_sycl PUBLIC IntelSYCL::SYCL_CXX)
  target_compile_features(apfrnn_sycl PRIVATE cxx_std_17)
  set_target_properties(apfrnn_sycl PROPERTIES POSITION_INDEPENDENT_CODE ON)

  target_compile_definitions(ispc_apfrnn_kernels PUBLIC APFRNN_HAVE_SYCL=1)
  # Link-only: the SYCL runtime is needed at the final link, but its -fsycl
  # usage requirements must not reach the ISPC kernel source.
  target_link_libraries(ispc_apfrnn_kernels PRIVATE $<LINK_ONLY:apfrnn_sycl>)
endif()

if(APFRNN_BUILD_TESTS OR APFRNN_BUILD_BENCHMARKS OR APFRNN_BUILD_BASELINES)
  # Synthetic scenes and VTK loading shared by tests and benchmarks, which also
  # reach the internal headers in src/.
  add_library(apfrnn_support INTERFACE)
  target_include_directories(apfrnn_support INTERFACE support src)
  target_link_libraries(apfrnn_support INTERFACE ispc_apfrnn_kernels)
endif()

if(APFRNN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(APFRNN_BUILD_BENCHMARKS OR APFRNN_BUILD_BASELINES)
  add_subdirectory(bench)
endif()
