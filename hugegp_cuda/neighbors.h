// neighbors.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "common.h"

__forceinline__ __device__ void insert_neighbor(
    uint32_t* neighbors,
    float* distances,
    uint32_t current_index,
    float current_distance,
    int k
) {
    int i = k - 1;
    while ((i > 0) && (current_distance < distances[i-1])) {
        neighbors[i] = neighbors[i-1];
        distances[i] = distances[i-1];
        --i;
    }
    neighbors[i] = current_index;
    distances[i] = current_distance;
}

template <int MAX_K, int N_DIM>
__forceinline__ __device__ void query_coarse_neighbors_impl(
    const float* points, // (N, d)
    const uint32_t* indices, // (N,)
    const int8_t* split_dims, // (N,)
    uint32_t* neighbors_out, // (N, k) output buffer
    int k, // number of neighbors to find
    size_t query_index // index of the query point
) {

    // compute number of points in levels above query
    size_t n_points = (1 << floored_log2(query_index + 1)) - 1;

    // load query point
    float query[N_DIM];
    #pragma unroll
    for (int i = 0; i < N_DIM; ++i) {
        query[i] = points[indices[query_index] * N_DIM + i];
    }

    // initialize neighbor arrays
    uint32_t neighbors[MAX_K];
    float distances[MAX_K];
    float max_distance = INFINITY;
    #pragma unroll
    for (int i = 0; i < k; ++i) {
        distances[i] = max_distance;
        neighbors[i] = 0;
    }

    // set up traversal variables
    uint32_t current = 0;
    uint32_t root_parent = (current - uint32_t(1)) / 2;
    uint32_t previous = root_parent;
    uint32_t next = 0;
    float current_point[N_DIM];

    // traverse until we return to root
    while (current != root_parent) {
        uint32_t parent = (current - uint32_t(1)) / 2;
        uint32_t current_index = indices[current];
        int8_t split_dim = split_dims[current];
        #pragma unroll
        for (int i = 0; i < N_DIM; ++i) {
            current_point[i] = points[current_index * N_DIM + i];
        }

        // update neighbor array if necessary
        if (previous == parent) {
            float current_distance = compute_square_distance<N_DIM>(points + current_index * N_DIM, query);
            if (current_distance < max_distance) {
                insert_neighbor(neighbors, distances, current, current_distance, k);
                max_distance = distances[k - 1];
            }
        }

        // locate children and determine if far child in range
        float split_distance = query[split_dim] - current_point[split_dim];
        uint32_t near_side = (split_distance >= 0.0f);
        uint32_t near_child = (2 * current + 1) + near_side;
        uint32_t far_child = (2 * current + 2) - near_side;
        uint32_t far_in_range = (far_child < n_points) && (split_distance * split_distance <= max_distance);

        // determine next node to traverse
        if (previous == parent) {
            if (near_child < n_points) next = near_child;
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
    #pragma unroll
    for (int i = 0; i < k; ++i) {
        neighbors_out[query_index * k + i] = neighbors[i];
    }
}

template <int MAX_K, int N_DIM>
__global__ void query_coarse_neighbors_kernel(
    const float* points,
    const uint32_t* indices,
    const int8_t* split_dims,
    uint32_t* neighbors,
    int k,
    size_t n_points
) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_points) return;
    query_coarse_neighbors_impl<MAX_K, N_DIM>(
        points,
        indices,
        split_dims,
        neighbors,
        k,
        tid
    );
}

template <int MAX_K, int N_DIM>
__host__ void query_coarse_neighbors(
    cudaStream_t stream,
    const float* points,
    const uint32_t* indices,
    const int8_t* split_dims,
    uint32_t* neighbors,
    int k,
    size_t n_points
) {
    size_t n_threads = n_points;
    size_t threads_per_block = 256;
    size_t n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
    query_coarse_neighbors_kernel<MAX_K, N_DIM><<<n_blocks, threads_per_block, 0, stream>>>(
        points,
        indices,
        split_dims,
        neighbors,
        k,
        n_points
    );
}