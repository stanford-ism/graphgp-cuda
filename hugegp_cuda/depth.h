// depth.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "common.h"
#include "sort.h"

// depths are the longest path from root to node in the graph
__global__ void compute_depths(
    const int* neighbors, // (N - n0, k) in topological order
    int* depths, // (N,)
    int n_points,
    int n0,
    int k,
    int n_threads // only 1 thread will be allowed to run
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= 1) return; // this loop must be done serially

    for (int i = 0; i < n0; ++i) {
        depths[i] = 0; // initial points have level 0
    }
    int neighbor_depth;
    int max_depth;

    for (int i = n0; i < n_points; ++i) {
        max_depth = 0;
        for (int j = 0; j < k; ++j) {
            neighbor_depth = depths[neighbors[(i - n0) * k + j]];
            if (neighbor_depth > max_depth) {
                max_depth = neighbor_depth;
            }
        }
        depths[i] = max_depth + 1;
    }
}

// alternative approach where we just update depths a bunch of times and hope for the best
__global__ void update_depths(
    const int* neighbors,
    const int* depths_src,
    int* depths_dst,
    int n0,
    int k,
    int n_threads
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_threads) return;

    int neighbor_depth;
    int max_depth = 0;
    for (int j = 0; j < k; ++j) {
        neighbor_depth = depths_src[neighbors[tid * k + j]];
        if (neighbor_depth > max_depth) {
            max_depth = neighbor_depth;
        }
    }
    depths_dst[tid + n0] = max_depth + 1;
}

__global__ void reindex_neighbors(int* neighbors, int* inv_permutation, int n_threads) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_threads) return;
    neighbors[tid] = inv_permutation[neighbors[tid]];
}

__host__ void order_by_depth(
    cudaStream_t stream,
    float* points, // (N, d)
    int* indices, // (N,)
    int* neighbors, // (N - n0, k)
    int* depths, // (N,)
    int* permutation, // (N,)
    int* inverse_permutation, // (N,)
    int n0,
    int k,
    int n_points,
    int n_dim
) {
    CUDA_LAUNCH(arange_kernel, n_points, stream, permutation);
    sort_by_depth(stream, points + n0 * n_dim, indices + n0, neighbors, depths + n0, permutation + n0, n_dim, k, n_points - n0);
    CUDA_LAUNCH(compute_inverse_permutation, n_points, stream, permutation, inverse_permutation);
    CUDA_LAUNCH(reindex_neighbors, (n_points - n0) * k, stream, neighbors, inverse_permutation);
}