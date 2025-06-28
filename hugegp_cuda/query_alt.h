// query.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "common.h"
#include "query.h"


__forceinline__ __device__ uint32_t bit_reverse(uint32_t n, uint32_t b) {
    n = ((n >> 1) & 0x55555555) | ((n & 0x55555555) << 1);
    n = ((n >> 2) & 0x33333333) | ((n & 0x33333333) << 2);
    n = ((n >> 4) & 0x0F0F0F0F) | ((n & 0x0F0F0F0F) << 4);
    n = ((n >> 8) & 0x00FF00FF) | ((n & 0x00FF00FF) << 8);
    n = (n >> 16) | (n << 16);
    return n >> (32 - b);
}

__forceinline__ __device__ uint32_t alt_index(uint32_t index) {
    uint32_t level = floored_log2(index + 1);
    uint32_t alt_index = index - ((1 << level) - 1);
    return bit_reverse(alt_index, level) + ((1 << level) - 1);
}

template <int MAX_K, int N_DIM>
__forceinline__ __device__ void query_preceding_neighbors_alt_impl(
    const float* points, // (N, d) in k-d tree order
    const int8_t* split_dims, // (N,) in k-d tree order
    uint32_t* neighbors_out, // (N, k) output buffer
    size_t query_index, // index of the query point in the tree
    size_t n_points, // total number of points in the tree
    int k // number of neighbors to find
) {
    // search all points before query alt index
    uint32_t query_alt_index = alt_index(query_index);

    // load query point
    float query[N_DIM];
    for (int i = 0; i < N_DIM; ++i) {
        query[i] = points[query_index * N_DIM + i];
    }

    // initialize neighbor arrays
    uint32_t neighbors[MAX_K];
    float distances[MAX_K];
    float max_distance = INFINITY;
    for (int i = 0; i < k; ++i) {
        distances[i] = max_distance;
        neighbors[i] = 0;
    }

    // set up traversal variables
    uint32_t current = 0;
    uint32_t root_parent = (current - uint32_t(1)) / 2;
    uint32_t previous = root_parent;
    uint32_t next = 0;

    // traverse until we return to root
    while (current != root_parent) {
        uint32_t parent = (current - uint32_t(1)) / 2;

        // update neighbor array if necessary
        if (previous == parent) {
            float current_distance = compute_square_distance<N_DIM>(points + current * N_DIM, query);
            if (current_distance < max_distance) {
                insert_neighbor(neighbors, distances, current, current_distance, k);
                max_distance = distances[k - 1];
            }
        }

        // locate children and determine if far child in range
        int8_t split_dim = split_dims[current];
        float split_distance = query[split_dim] - points[current * N_DIM + split_dim];
        uint32_t near_side = (split_distance >= 0.0f);
        uint32_t near_child = (2 * current + 1) + near_side;
        uint32_t far_child = (2 * current + 2) - near_side;
        uint32_t far_in_range = (far_child < n_points) && (alt_index(far_child) < query_alt_index) && (split_distance * split_distance <= max_distance);

        // determine next node to traverse
        if (previous == parent) {
            if ((near_child < n_points) && (alt_index(near_child) < query_alt_index)) next = near_child;
            else if (far_in_range) next = far_child;
            else next = parent;
        } else if (previous == near_child) {
            if (far_in_range) next = far_child;
            else next = parent;
        } else {
            next = parent;
        }
        previous = current;
        current = next;
    }

    // write neighbors to output
    for (int i = 0; i < k; ++i) {
        neighbors_out[query_index * k + i] = neighbors[i];
    }
}

template <int MAX_K, int N_DIM>
__global__ void query_preceding_neighbors_alt_kernel(
    const float* points,
    const int8_t* split_dims,
    uint32_t* neighbors,
    size_t n_points,
    int k
) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_points) return;
    query_preceding_neighbors_alt_impl<MAX_K, N_DIM>(points, split_dims, neighbors, tid, n_points, k);
}

template <int MAX_K, int N_DIM>
__host__ void query_preceding_neighbors_alt(
    cudaStream_t stream,
    const float* points,
    const int8_t* split_dims,
    uint32_t* neighbors,
    size_t n_points,
    int k
) {
    size_t n_threads = n_points;
    size_t threads_per_block = 256;
    size_t n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
    query_preceding_neighbors_alt_kernel<MAX_K, N_DIM><<<n_blocks, threads_per_block, 0, stream>>>(
        points, split_dims, neighbors, n_points, k
    );
}