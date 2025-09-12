// depth.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "common.h"
#include "sort.h"
#include "cubit/cubit.h"

// alternative approach where we just update depths a bunch of times and hope for the best
__global__ void update_depths_parallel(
    const int* neighbors,
    const int* old_depths,
    int* new_depths,
    int* changed, // single flag 0 or 1
    int n0,
    int k,
    int n_threads
) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_threads) return;

    int neighbor_depth;
    int max_depth = 0;
    for (int j = 0; j < k; ++j) {
        neighbor_depth = old_depths[neighbors[tid * k + j]];
        if (neighbor_depth > max_depth) {
            max_depth = neighbor_depth;
        }
    }
    if (max_depth + 1 != old_depths[tid + n0]) {
        atomicOr(changed, 1); // reduction better in principle but this isn't the bottleneck
    }
    new_depths[tid + n0] = max_depth + 1;
}

// Compute longest path from root by repeatedly updating all nodes until the depth doesn't change.
// This is much faster than the serial approach for shallow graphs on large GPUs.
__host__ void compute_depths_parallel(
    cudaStream_t stream,
    const int* neighbors,
    int* depths,
    int* temp,
    int n0,
    int k,
    int n_points
) {
    // initialize depth arrays
    int* new_depths = depths;
    int* old_depths = temp;
    cudaMemsetAsync(new_depths, 0, n_points * sizeof(int), stream);
    cudaMemsetAsync(old_depths, -1, n_points * sizeof(int), stream);
    cudaMemsetAsync(old_depths, 0, n0 * sizeof(int), stream);

    // flag for when depths change
    int changed = 1;
    int* d_changed;
    cudaMallocAsync(&d_changed, sizeof(int), stream);

    // loop until depths don't change
    while (changed == 1) {
        cudaMemsetAsync(d_changed, 0, sizeof(int), stream);
        int* tmp = new_depths; new_depths = old_depths; old_depths = tmp; // swap buffers
        CUDA_LAUNCH(update_depths_parallel, n_points - n0, stream, neighbors, old_depths, new_depths, d_changed, n0, k);
        cudaMemcpyAsync(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    if (new_depths != depths) {
        cudaMemcpyAsync(depths, new_depths, n_points * sizeof(int), cudaMemcpyDeviceToHost, stream);
    }
}

// Compute longest path from root with a single kernel that traverses the whole array.
__global__ void compute_depths_serial(
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
            neighbor_depth = depths[neighbors[(i - n0) * (size_t)k + j]];
            if (neighbor_depth > max_depth) {
                max_depth = neighbor_depth;
            }
        }
        depths[i] = max_depth + 1;
    }
}

__global__ void reindex_neighbors(int* neighbors, const int* inv_permutation, int n_threads) {
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
    int* temp, // (N,)
    int n0,
    int k,
    int n_points,
    int n_dim
) {

    // sort by depth, tracking permutation
    CUDA_LAUNCH(arange_kernel, n_points, stream, permutation);
    sort(depths + n0, permutation + n0, n_points - n0, stream);

    // permute arrays one-by-one
    float* temp_float = reinterpret_cast<float*>(temp);
    permute(stream, points, temp_float, permutation, n_dim, 0, n_points);
    permute(stream, indices, temp, permutation, 1, 0, n_points);
    permute(stream, neighbors, temp, permutation + n0, k, n0, n_points - n0); // tricky, first n0 don't exist

    // update neighbors to point to new indices
    CUDA_LAUNCH(compute_inverse_permutation, n_points, stream, permutation, temp);
    CUDA_LAUNCH(reindex_neighbors, (n_points - n0) * k, stream, neighbors, temp);
}