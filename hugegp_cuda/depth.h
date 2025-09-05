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

// template <int N_DIM>
// __host__ void order_by_depth(
//     cudaStream_t stream,
//     const float* points, // (N, d)
//     const int* indices, // (N,)
//     const int* neighbors, // (N - n0, k)
//     const int* depths, // (N,)
//     float* points_out, // (N, d)
//     int* indices_out, // (N,)
//     int* neighbors_out, // (N - n0, k)
//     int* depths_out, // (N,)
//     int* depths_buffer, // (N,)
//     int* order_buffer, // (N,)
//     int n0,
//     int n_points
// ) {
//     int n_threads = n_points;
//     int threads_per_block = 256;
//     int n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;

//     arange_kernel<<<n_blocks, threads_per_block, 0, stream>>>(order_buffer, n_points);

//     cub::DeviceRadixSort::SortPairs(
//         order_buffer, n_points,
//         points_out, n_points,
//         indices_out, n_points,
//         neighbors_out, n_points - n0,
//         depths_out, n_points,
//         stream
//     );
// }