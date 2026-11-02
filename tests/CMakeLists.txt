find_package(doctest CONFIG REQUIRED)

add_executable(ispc_apfrnn_tests
  apfrnn_test.cpp neighbor_search_test.cpp plan_test.cpp sync_points_test.cpp)
target_link_libraries(ispc_apfrnn_tests PRIVATE apfrnn_support doctest::doctest)

add_test(NAME ispc_apfrnn_tests COMMAND ispc_apfrnn_tests)
