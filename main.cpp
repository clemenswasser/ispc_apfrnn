#include "apfrnn.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

int main() {
  constexpr int N = 1000000;
  constexpr float R = 2.0f;

  std::cout << "Allocating and generating " << N << " points...\n";
  std::vector<float> X(N), Y(N), Z(N);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.0f, 100.0f);
  for (int i = 0; i < N; ++i) {
    X[i] = dist(gen);
    Y[i] = dist(gen);
    Z[i] = dist(gen);
  }
  X[0] = 1e8;
  Y[0] = -1e8;
  Z[0] = 5e8;

  auto t_start = std::chrono::steady_clock::now();
  auto neighbor_data = apfrnn::build_neighbor_search_data(X, Y, Z, R);
  apfrnn::write_neighbors_parallel(neighbor_data);
  auto t_end = std::chrono::steady_clock::now();
  std::cout
      << "Pipeline completed in: "
      << std::chrono::duration<double, std::milli>(t_end - t_start).count()
      << " ms\n";
  std::cout << "Total neighbors found: " << neighbor_data.row_ptr[N] << "\n";

  int test_idx = 42;
  std::vector<int> exact_neighbors;
  for (int i = 0; i < N; ++i) {
    if (i == test_idx)
      continue;
    float dx = X[test_idx] - X[i];
    float dy = Y[test_idx] - Y[i];
    float dz = Z[test_idx] - Z[i];
    if (dx * dx + dy * dy + dz * dz <= R * R) {
      exact_neighbors.push_back(i);
    }
  }

  std::vector<int> ispc_neighbors;
  auto sorted_it = std::find(neighbor_data.original_index.begin(),
                             neighbor_data.original_index.end(), test_idx);
  int sorted_idx = sorted_it - neighbor_data.original_index.begin();
  for (int i = neighbor_data.row_ptr[sorted_idx];
       i < neighbor_data.row_ptr[sorted_idx + 1]; ++i) {
    ispc_neighbors.push_back(neighbor_data.col_idx[i]);
  }

  std::sort(exact_neighbors.begin(), exact_neighbors.end());
  std::sort(ispc_neighbors.begin(), ispc_neighbors.end());

  std::cout << "\nValidation for Particle " << test_idx << ":\n";
  std::cout << "Expected count : " << exact_neighbors.size() << "\n";
  std::cout << "ISPC count     : " << ispc_neighbors.size() << "\n";

  bool valid = (exact_neighbors == ispc_neighbors);
  std::cout << "Data Matched?  : " << (valid ? "YES [SUCCESS]" : "NO [FAILED]")
            << "\n";
  std::cout << "Neighbors of extreme outlier [0] : " << neighbor_data.counts[0]
            << "\n";

  return 0;
}
