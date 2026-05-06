cmake_minimum_required(VERSION 3.31)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    include(FetchContent)
    FetchContent_Declare(vcpkg URL [[https://github.com/microsoft/vcpkg/archive/refs/tags/2026.04.27.tar.gz]])
    FetchContent_MakeAvailable(vcpkg)
    set(CMAKE_TOOLCHAIN_FILE "${vcpkg_SOURCE_DIR}/scripts/buildsystems/vcpkg.cmake")
endif()

project(ispc_apfrnn LANGUAGES CXX ISPC)

add_library(ispc_apfrnn_kernels kernel.ispc apfrnn.cpp)
target_compile_options(ispc_apfrnn_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:ISPC>:$<$<NOT:$<CONFIG:Debug>>:-O3>>
    $<$<COMPILE_LANGUAGE:ISPC>:$<$<NOT:$<CONFIG:Release>>:-g>>
    $<$<COMPILE_LANGUAGE:ISPC>:--opt=fast-math>)
target_sources(ispc_apfrnn_kernels PUBLIC FILE_SET HEADERS FILES apfrnn.hpp)
target_link_libraries(ispc_apfrnn_kernels PUBLIC TBB::tbb TBB::tbbmalloc $<TARGET_NAME_IF_EXISTS:TBB::tbbmalloc_proxy>)

add_executable(ispc_apfrnn main.cpp)
find_package(TBB CONFIG REQUIRED)
find_package(doctest CONFIG REQUIRED)
target_link_libraries(ispc_apfrnn PRIVATE ispc_apfrnn_kernels doctest::doctest)

add_executable(ispc_apfrnn_bench bench.cpp)
find_package(benchmark CONFIG REQUIRED)
target_link_libraries(ispc_apfrnn_bench PRIVATE benchmark::benchmark benchmark::benchmark_main ispc_apfrnn_kernels)
