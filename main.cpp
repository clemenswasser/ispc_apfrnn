
// main.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>
#include <tbb/blocked_range.h>

// ISPC function bindings
extern "C" {
    void count_neighbors_ispc(
        int start_query, int end_query, float R, float R2,
        const float* X, const float* Y, const float* Z, const int* orig_idx,
        const int* bin_start, const int* bin_end, uint32_t hash_mask, int* counts);

    void write_neighbors_ispc(
        int start_query, int end_query, float R, float R2,
        const float* X, const float* Y, const float* Z, const int* orig_idx,
        const int* bin_start, const int* bin_end, uint32_t hash_mask,
        const int* row_ptr, int* col_idx);
}

// Ensure C++ logic perfectly mirrors ISPC bitwise hash
inline uint32_t spatial_hash(float x, float y, float z, float R, uint32_t hash_mask) {
    uint32_t ux = (uint32_t)(int)std::floor(x / R);
    uint32_t uy = (uint32_t)(int)std::floor(y / R);
    uint32_t uz = (uint32_t)(int)std::floor(z / R);
    return ((ux * 73856093) ^ (uy * 19349663) ^ (uz * 83492791)) & hash_mask;
}

int main() {
    int N = 1000000;
    float R = 2.0f;
    uint32_t HASH_SIZE = 1 << 22; // ~4 Million buckets for insane distance handling
    uint32_t hash_mask = HASH_SIZE - 1;

    std::cout << "Allocating and generating " << N << " points...\n";
    std::vector<float> X(N), Y(N), Z(N);
    
    // Generate Random Data + Add an "Insane Distance" particle
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);
    for (int i = 0; i < N; ++i) {
        X[i] = dist(gen); Y[i] = dist(gen); Z[i] = dist(gen);
    }
    X[0] = 1e8; Y[0] = -1e8; Z[0] = 5e8; // Extreme spatial outlier proof

    auto t_start = std::chrono::high_resolution_clock::now();

    // 1. Compute Hashes & Bin Counts
    std::vector<uint32_t> hashes(N);
    std::vector<std::atomic<int>> bin_counts(HASH_SIZE);
    tbb::parallel_for(0, N, [&](int i) {
        hashes[i] = spatial_hash(X[i], Y[i], Z[i], R, hash_mask);
        bin_counts[hashes[i]].fetch_add(1, std::memory_order_relaxed);
    });

    // 2. Prefix Sum to finalize Grid Setup
    std::vector<int> bin_start(HASH_SIZE, 0), bin_end(HASH_SIZE, 0);
    std::vector<std::atomic<int>> bin_write_pos(HASH_SIZE);
    int current_offset = 0;
    for (uint32_t i = 0; i < HASH_SIZE; ++i) {
        bin_start[i] = current_offset;
        bin_write_pos[i].store(current_offset, std::memory_order_relaxed);
        current_offset += bin_counts[i].load(std::memory_order_relaxed);
        bin_end[i] = current_offset;
    }

    // 3. Scatter to sorted SoA Arrays O(N) Cache Optimizer
    std::vector<float> sX(N), sY(N), sZ(N);
    std::vector<int> sOrig(N);
    tbb::parallel_for(0, N, [&](int i) {
        uint32_t h = hashes[i];
        int pos = bin_write_pos[h].fetch_add(1, std::memory_order_relaxed);
        sX[pos] = X[i]; sY[pos] = Y[i]; sZ[pos] = Z[i]; sOrig[pos] = i;
    });

    // 4. ISPC Pass 1: Count Neighbors
    std::vector<int> counts(N, 0);
    tbb::parallel_for(tbb::blocked_range<int>(0, N, 1024), [&](const tbb::blocked_range<int>& r) {
        count_neighbors_ispc(r.begin(), r.end(), R, R*R, sX.data(), sY.data(), sZ.data(), sOrig.data(),
                             bin_start.data(), bin_end.data(), hash_mask, counts.data());
    });

    // 5. TBB Exclusive Prefix Scan (Compute exact CSR row limits)
    std::vector<int> row_ptr(N + 1, 0);
    int total_neighbors = tbb::parallel_scan(
        tbb::blocked_range<int>(0, N), 0,
        [&](const tbb::blocked_range<int>& r, int sum, bool is_final_scan) {
            int temp = sum;
            for (int i = r.begin(); i < r.end(); ++i) {
                if (is_final_scan) row_ptr[i] = temp;
                temp += counts[i];
            }
            return temp;
        },[](int left, int right) { return left + right; }
    );
    row_ptr[N] = total_neighbors;

    // 6. Allocate CSR column memory
    std::vector<int> col_idx(total_neighbors);

    // 7. ISPC Pass 2: Write exact neighbor relations
    tbb::parallel_for(tbb::blocked_range<int>(0, N, 1024), [&](const tbb::blocked_range<int>& r) {
        write_neighbors_ispc(r.begin(), r.end(), R, R*R, sX.data(), sY.data(), sZ.data(), sOrig.data(),
                             bin_start.data(), bin_end.data(), hash_mask, row_ptr.data(), col_idx.data());
    });

    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Pipeline completed in: " 
              << std::chrono::duration<double, std::milli>(t_end - t_start).count() << " ms\n";
    std::cout << "Total neighbors found: " << total_neighbors << "\n";

    // ============================================
    // VALIDATION: Proof of absolute correctness
    // ============================================
    int test_idx = 42; // Choose random original index to verify
    std::vector<int> exact_neighbors;
    for (int i = 0; i < N; ++i) {
        if (i == test_idx) continue;
        float dx = X[test_idx] - X[i];
        float dy = Y[test_idx] - Y[i];
        float dz = Z[test_idx] - Z[i];
        if (dx*dx + dy*dy + dz*dz <= R*R) {
            exact_neighbors.push_back(i);
        }
    }
    
    // Gather results produced by ISPC
    std::vector<int> ispc_neighbors;
    for (int i = row_ptr[test_idx]; i < row_ptr[test_idx + 1]; ++i) {
        ispc_neighbors.push_back(col_idx[i]);
    }

    std::sort(exact_neighbors.begin(), exact_neighbors.end());
    std::sort(ispc_neighbors.begin(), ispc_neighbors.end());

    std::cout << "\nValidation for Particle " << test_idx << ":\n";
    std::cout << "Expected count : " << exact_neighbors.size() << "\n";
    std::cout << "ISPC count     : " << ispc_neighbors.size() << "\n";
    
    bool valid = (exact_neighbors == ispc_neighbors);
    std::cout << "Data Matched?  : " << (valid ? "YES [SUCCESS]" : "NO [FAILED]") << "\n";
    std::cout << "Neighbors of extreme outlier [0] : " << counts[0] << "\n";

    return 0;
}
