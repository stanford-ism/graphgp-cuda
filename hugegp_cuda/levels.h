// levels.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// levels are equivalent to the longest path from root to node in the graph
__global__ void compute_levels(
    const uint32_t* neighbors, // (N, k) in topological order
    uint32_t* levels, // (N,)
    size_t n_points,
    size_t n_initial,
    int k
) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid > 0) return; // this loop must be done serially

    for (size_t i = 0; i < n_initial; ++i) {
        levels[i] = 0; // initial points have level 0
    }

    for (size_t i = n_initial; i < n_points; ++i) {
        uint32_t neighbor_level = levels[neighbors[i * k]];
        uint32_t max_level = neighbor_level;
        for (int j = 1; j < k; ++j) {
            neighbor_level = levels[neighbors[i * k + j]];
            if (neighbor_level > max_level) {
                max_level = neighbor_level;
            }
        }
        levels[i] = max_level + 1;
    }
}