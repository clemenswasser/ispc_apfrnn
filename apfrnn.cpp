#include "apfrnn.hpp"

#include "kernel_ispc.h"

#include <atomic>
#include <cmath>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>

namespace apfrnn {

uint32_t spatial_hash(float x, float y, float z, float radius,
                      uint32_t hash_mask) {
  uint32_t ux = static_cast<uint32_t>(static_cast<int>(std::floor(x / radius)));
  uint32_t uy = static_cast<uint32_t>(static_cast<int>(std::floor(y / radius)));
  uint32_t uz = static_cast<uint32_t>(static_cast<int>(std::floor(z / radius)));
  return ((ux * 73856093) ^ (uy * 19349663) ^ (uz * 83492791)) & hash_mask;
}

NeighborSearchData build_neighbor_search_data(const std::vector<float> &x,
                                              const std::vector<float> &y,
                                              const std::vector<float> &z,
                                              float radius,
                                              uint32_t hash_size) {
  NeighborSearchData data;
  data.num_points = static_cast<int>(x.size());
  data.radius = radius;
  data.hash_size = hash_size;
  data.hash_mask = hash_size - 1;

  std::vector<uint32_t> hashes(data.num_points);
  std::vector<std::atomic<int>> bin_counts(hash_size);
  for (auto &count : bin_counts) {
    count.store(0, std::memory_order_relaxed);
  }

  tbb::parallel_for(0, data.num_points, [&](int i) {
    hashes[i] = spatial_hash(x[i], y[i], z[i], radius, data.hash_mask);
    bin_counts[hashes[i]].fetch_add(1, std::memory_order_relaxed);
  });

  data.bin_start.assign(hash_size, 0);
  data.bin_end.assign(hash_size, 0);
  std::vector<std::atomic<int>> bin_write_pos(hash_size);
  int current_offset = 0;
  for (uint32_t i = 0; i < hash_size; ++i) {
    data.bin_start[i] = current_offset;
    bin_write_pos[i].store(current_offset, std::memory_order_relaxed);
    current_offset += bin_counts[i].load(std::memory_order_relaxed);
    data.bin_end[i] = current_offset;
  }

  data.sorted_x.resize(data.num_points);
  data.sorted_y.resize(data.num_points);
  data.sorted_z.resize(data.num_points);
  data.original_index.resize(data.num_points);
  tbb::parallel_for(0, data.num_points, [&](int i) {
    uint32_t hash = hashes[i];
    int pos = bin_write_pos[hash].fetch_add(1, std::memory_order_relaxed);
    data.sorted_x[pos] = x[i];
    data.sorted_y[pos] = y[i];
    data.sorted_z[pos] = z[i];
    data.original_index[pos] = i;
  });

  data.counts.assign(data.num_points, 0);
  tbb::parallel_for(
      tbb::blocked_range<int>(0, data.num_points, 1024),
      [&](const tbb::blocked_range<int> &range) {
        ispc::count_neighbors_ispc(
            range.begin(), range.end(), data.radius, data.radius * data.radius,
            data.sorted_x.data(), data.sorted_y.data(), data.sorted_z.data(),
            data.original_index.data(), data.bin_start.data(),
            data.bin_end.data(), data.hash_mask, data.counts.data());
      });

  data.row_ptr.assign(data.num_points + 1, 0);
  int total_neighbors = tbb::parallel_scan(
      tbb::blocked_range<int>(0, data.num_points), 0,
      [&](const tbb::blocked_range<int> &range, int sum, bool is_final_scan) {
        int temp = sum;
        for (int i = range.begin(); i < range.end(); ++i) {
          if (is_final_scan) {
            data.row_ptr[i] = temp;
          }
          temp += data.counts[i];
        }
        return temp;
      },
      [](int left, int right) { return left + right; });
  data.row_ptr[data.num_points] = total_neighbors;
  data.col_idx.assign(total_neighbors, 0);

  return data;
}

void write_neighbors_range(NeighborSearchData &data, int start_query,
                           int end_query) {
  ispc::write_neighbors_ispc(
      start_query, end_query, data.radius, data.radius * data.radius,
      data.sorted_x.data(), data.sorted_y.data(), data.sorted_z.data(),
      data.original_index.data(), data.bin_start.data(), data.bin_end.data(),
      data.hash_mask, data.row_ptr.data(), data.col_idx.data());
}

void write_neighbors_parallel(NeighborSearchData &data) {
  tbb::parallel_for(tbb::blocked_range<int>(0, data.num_points, 1024),
                    [&](const tbb::blocked_range<int> &range) {
                      write_neighbors_range(data, range.begin(), range.end());
                    });
}

} // namespace apfrnn