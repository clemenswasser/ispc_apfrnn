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

find_package(TBB CONFIG REQUIRED)

add_library(ispc_apfrnn_kernels
  src/apfrnn.cpp
  src/neighbor_search.cpp
  src/plan.cpp
  src/sweep.cpp
  src/kernel.ispc)
target_sources(ispc_apfrnn_kernels PUBLIC
  FILE_SET HEADERS BASE_DIRS include FILES include/apfrnn/neighbor_search.hpp)
target_compile_features(ispc_apfrnn_kernels PUBLIC cxx_std_17)
target_compile_options(ispc_apfrnn_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:ISPC>:$<$<NOT:$<CONFIG:Debug>>:-O3>>
  $<$<COMPILE_LANGUAGE:ISPC>:$<$<NOT:$<CONFIG:Release>>:-g>>
  $<$<COMPILE_LANGUAGE:ISPC>:--opt=fast-math>)
target_link_libraries(ispc_apfrnn_kernels PUBLIC
  TBB::tbb TBB::tbbmalloc $<TARGET_NAME_IF_EXISTS:TBB::tbbmalloc_proxy>)

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
