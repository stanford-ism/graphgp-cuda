// depth.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// depths are the longest path from root to node in the graph
__global__ void compute_depths(
    const int* neighbors, // (N - n0, k) in topological order
    int* depths, // (N,)
    int n_points,
    int n0,
    int k
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid > 0) return; // this loop must be done serially

    for (int i = 0; i < n0; ++i) {
        depths[i] = 0; // initial points have level 0
    }

    for (int i = n0; i < n_points; ++i) {
        int neighbor_level = depths[neighbors[(i - n0) * k]];
        int max_level = neighbor_level;
        for (int j = 1; j < k; ++j) {
            neighbor_level = depths[neighbors[(i - n0) * k + j]];
            if (neighbor_level > max_level) {
                max_level = neighbor_level;
            }
        }
        depths[i] = max_level + 1;
    }
}