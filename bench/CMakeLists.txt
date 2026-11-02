find_package(benchmark CONFIG REQUIRED)

if(APFRNN_BUILD_BENCHMARKS)
  add_executable(ispc_apfrnn_bench search_bench.cpp api_bench.cpp)
  target_link_libraries(ispc_apfrnn_bench PRIVATE
    apfrnn_support benchmark::benchmark)
endif()

if(APFRNN_BUILD_BASELINES)
  if(NOT APFRNN_BUILD_BENCHMARKS)
    message(FATAL_ERROR "APFRNN_BUILD_BASELINES requires APFRNN_BUILD_BENCHMARKS")
  endif()
  find_package(OpenMP REQUIRED)
  set(USE_DOUBLE_PRECISION OFF CACHE BOOL "CompactNSearch float build" FORCE)

  include(FetchContent)
  FetchContent_Declare(CompactNSearch
    URL [[https://github.com/InteractiveComputerGraphics/CompactNSearch/archive/b40afcf47fe1963b363eba2371f04b42720fcb1d.tar.gz]]
    EXCLUDE_FROM_ALL)
  FetchContent_Declare(TreeNSearch
    URL [[https://github.com/InteractiveComputerGraphics/TreeNSearch/archive/61cf111f65c01f1d71fb5023ac72ce6bb0e8d6f9.tar.gz]]
    EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(CompactNSearch TreeNSearch)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(TreeNSearch PRIVATE -mavx2)
  endif()

  target_compile_definitions(ispc_apfrnn_bench PRIVATE
    APFRNN_BENCH_BASELINES=1)
  target_link_libraries(ispc_apfrnn_bench PRIVATE
    CompactNSearch TreeNSearch OpenMP::OpenMP_CXX)
endif()
