#pragma once

#include "apfrnn.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

namespace apfrnn::support {

struct PointCloud {
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
};

inline PointCloud make_point_cloud(int num_points, float side, uint32_t seed) {
  PointCloud cloud;
  cloud.x.resize(num_points);
  cloud.y.resize(num_points);
  cloud.z.resize(num_points);

  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(0.0f, side);
  for (int index = 0; index < num_points; ++index) {
    cloud.x[index] = distribution(generator);
    cloud.y[index] = distribution(generator);
    cloud.z[index] = distribution(generator);
  }

  return cloud;
}

inline void set_extreme_outlier(PointCloud &cloud, int point_index = 0) {
  cloud.x[point_index] = 1e8f;
  cloud.y[point_index] = -1e8f;
  cloud.z[point_index] = 5e8f;
}

inline std::vector<int> exact_neighbors_for(const PointCloud &cloud,
                                            int test_index, float radius) {
  std::vector<int> neighbors;
  for (int index = 0; index < static_cast<int>(cloud.x.size()); ++index) {
    if (index == test_index) {
      continue;
    }

    float dx = cloud.x[test_index] - cloud.x[index];
    float dy = cloud.y[test_index] - cloud.y[index];
    float dz = cloud.z[test_index] - cloud.z[index];
    if (dx * dx + dy * dy + dz * dz <= radius * radius) {
      neighbors.push_back(index);
    }
  }

  std::sort(neighbors.begin(), neighbors.end());
  return neighbors;
}

inline std::vector<int>
exact_cross_neighbors_for(const PointCloud &query_cloud, int query_index,
                          const PointCloud &target_cloud, float radius) {
  std::vector<int> neighbors;
  for (int index = 0; index < static_cast<int>(target_cloud.x.size());
       ++index) {
    float dx = query_cloud.x[query_index] - target_cloud.x[index];
    float dy = query_cloud.y[query_index] - target_cloud.y[index];
    float dz = query_cloud.z[query_index] - target_cloud.z[index];
    if (dx * dx + dy * dy + dz * dz <= radius * radius) {
      neighbors.push_back(index);
    }
  }

  std::sort(neighbors.begin(), neighbors.end());
  return neighbors;
}

inline int sorted_index_for(const NeighborSearchData &neighbor_data,
                            int original_index) {
  auto sorted_it =
      std::find(neighbor_data.original_index.begin(),
                neighbor_data.original_index.end(), original_index);
  assert(sorted_it != neighbor_data.original_index.end());
  return static_cast<int>(sorted_it - neighbor_data.original_index.begin());
}

inline std::vector<int>
ispc_neighbors_for(const NeighborSearchData &neighbor_data,
                   int original_index) {
  std::vector<int> neighbors;
  int sorted_index = sorted_index_for(neighbor_data, original_index);
  for (int index = neighbor_data.row_ptr[sorted_index];
       index < neighbor_data.row_ptr[sorted_index + 1]; ++index) {
    neighbors.push_back(neighbor_data.col_idx[index]);
  }

  std::sort(neighbors.begin(), neighbors.end());
  return neighbors;
}

inline std::vector<int>
ispc_neighbors_for(const CrossNeighborSearchData &neighbor_data,
                   int original_index) {
  std::vector<int> neighbors;
  auto sorted_it =
      std::find(neighbor_data.original_index.begin(),
                neighbor_data.original_index.end(), original_index);
  assert(sorted_it != neighbor_data.original_index.end());
  int sorted_index =
      static_cast<int>(sorted_it - neighbor_data.original_index.begin());
  for (int index = neighbor_data.row_ptr[sorted_index];
       index < neighbor_data.row_ptr[sorted_index + 1]; ++index) {
    neighbors.push_back(neighbor_data.col_idx[index]);
  }

  std::sort(neighbors.begin(), neighbors.end());
  return neighbors;
}

inline bool is_strictly_sorted_unique(const std::vector<int> &values) {
  return std::adjacent_find(
             values.begin(), values.end(),
             [](int left, int right) { return left >= right; }) == values.end();
}

template <typename SearchData>
inline bool has_valid_row_ptr(const SearchData &neighbor_data) {
  if (neighbor_data.row_ptr.empty()) {
    return neighbor_data.num_points == 0;
  }

  if (neighbor_data.row_ptr.size() !=
      static_cast<std::size_t>(neighbor_data.num_points + 1)) {
    return false;
  }

  if (neighbor_data.row_ptr.front() != 0) {
    return false;
  }

  for (std::size_t index = 1; index < neighbor_data.row_ptr.size(); ++index) {
    if (neighbor_data.row_ptr[index] < neighbor_data.row_ptr[index - 1]) {
      return false;
    }
  }

  return neighbor_data.row_ptr.back() ==
         static_cast<int>(neighbor_data.col_idx.size());
}

inline std::size_t
collect_concat_cross_neighbors(const NeighborSearchData &neighbor_data,
                               int query_size) {
  std::vector<int> original_to_sorted(neighbor_data.num_points);
  for (int sorted_index = 0; sorted_index < neighbor_data.num_points;
       ++sorted_index) {
    original_to_sorted[neighbor_data.original_index[sorted_index]] =
        sorted_index;
  }

  std::size_t total_neighbors = 0;
  for (int point_index = 0; point_index < query_size; ++point_index) {
    int sorted_index = original_to_sorted[point_index];
    int begin = neighbor_data.row_ptr[sorted_index];
    int end = neighbor_data.row_ptr[sorted_index + 1];
    for (int row_index = begin; row_index < end; ++row_index) {
      if (neighbor_data.col_idx[row_index] >= query_size) {
        ++total_neighbors;
      }
    }
  }

  return total_neighbors;
}

} // namespace apfrnn::support