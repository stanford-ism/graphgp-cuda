// query.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "common.h"
#include "tree.h"

__forceinline__ __device__ void insert_neighbor(
    int* neighbors,
    float* distances,
    int current_index,
    float current_distance,
    int k
) {
    int i = k - 1;
    // ensure well-defined ordering by putting earlier indices first
    while ((i > 0) && ((current_distance < distances[i-1]) || (current_distance == distances[i-1] && current_index < neighbors[i-1]))) {
        neighbors[i] = neighbors[i-1];
        distances[i] = distances[i-1];
        --i;
    }
    neighbors[i] = current_index;
    distances[i] = current_distance;
}


template <int MAX_K, int N_DIM>
__global__ void query_neighbors_kernel(
    const float* points, // (N, d)
    const int* split_dims, // (N,)
    const int* query_indices, // (Q,)
    const int* max_indices, // (Q,)
    int* neighbors_out, // (Q, k)
    int k,
    int n_points,
    int n_queries
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_queries) return;
    int query_index = query_indices[tid];
    int max_index = max_indices[tid];

    // load query point
    float query[N_DIM];
    for (int i = 0; i < N_DIM; ++i) {
        query[i] = points[query_index * N_DIM + i];
    }

    // initialize neighbor arrays
    int neighbors[MAX_K];
    float distances[MAX_K];
    float max_distance = INFINITY;
    for (int i = 0; i < k; ++i) {
        distances[i] = max_distance;
        neighbors[i] = 0;
    }

    // set up traversal variables
    int current = 0;
    int root_parent = compute_parent(current);
    int previous = root_parent;
    int next = 0;

    // traverse until we return to root
    while (current != root_parent) {
        int parent = compute_parent(current);

        // update neighbor array if necessary
        if (previous == parent) {
            float current_distance = compute_square_distance<N_DIM>(points + current * N_DIM, query);
            if (current_distance < max_distance) {
                insert_neighbor(neighbors, distances, current, current_distance, k);
                max_distance = distances[k - 1];
            }
        }

        // locate children and determine if far child in range
        int split_dim = split_dims[current];
        float split_distance = query[split_dim] - points[current * N_DIM + split_dim];
        int near_child = (split_distance < 0) ? compute_left(current) : compute_right(current);
        int far_child = (split_distance < 0) ? compute_right(current) : compute_left(current);
        bool far_in_range = (far_child < max_index) & (split_distance * split_distance <= max_distance);

        // determine next node to traverse
        if (previous == parent) {
            if (near_child < max_index) next = near_child;
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
        neighbors_out[tid * k + i] = neighbors[i];
    }
}

template <int MAX_K, int N_DIM>
__host__ void query_neighbors(
    cudaStream_t stream,
    const float* points,
    const int* split_dims,
    const int* query_indices,
    const int* max_indices,
    int* neighbors,
    int k,
    int n_points,
    int n_queries
) {
    int n_threads = n_queries;
    int threads_per_block = 256;
    int n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
    query_neighbors_kernel<MAX_K, N_DIM><<<n_blocks, threads_per_block, 0, stream>>>(
        points, split_dims, query_indices, max_indices, neighbors, k, n_points, n_queries
    );
}

template <int MAX_K, int N_DIM>
__global__ void query_preceding_neighbors_kernel(
    const float* points, // (N, d)
    const int* split_dims, // (N,)
    int* neighbors_out, // (Q, k)
    int n0,
    int k,
    int n_points
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_points - n0) return;
    int query_idx = idx + n0;

    // load query point
    float query[N_DIM];
    for (int i = 0; i < N_DIM; ++i) {
        query[i] = points[query_idx * N_DIM + i];
    }

    // initialize neighbor arrays
    int neighbors[MAX_K];
    float distances[MAX_K];
    float max_distance = INFINITY;
    for (int i = 0; i < k; ++i) {
        distances[i] = max_distance;
        neighbors[i] = 0;
    }

    // set up traversal variables
    int current = 0;
    int root_parent = compute_parent(current);
    int previous = root_parent;
    int next = 0;

    // traverse until we return to root
    while (current != root_parent) {
        int parent = compute_parent(current);

        // update neighbor array if necessary
        if (previous == parent) {
            float current_distance = compute_square_distance<N_DIM>(points + current * N_DIM, query);
            if (current_distance < max_distance) {
                insert_neighbor(neighbors, distances, current, current_distance, k);
                max_distance = distances[k - 1];
            }
        }

        // locate children and determine if far child in range
        int split_dim = split_dims[current];
        float split_distance = query[split_dim] - points[current * N_DIM + split_dim];
        int near_child = (split_distance < 0) ? compute_left(current) : compute_right(current);
        int far_child = (split_distance < 0) ? compute_right(current) : compute_left(current);
        bool far_in_range = (far_child < query_idx) & (split_distance * split_distance <= max_distance);

        // determine next node to traverse
        if (previous == parent) {
            if (near_child < query_idx) next = near_child;
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
        neighbors_out[idx * k + i] = neighbors[i];
    }
}

template <int MAX_K, int N_DIM>
__host__ void query_preceding_neighbors(
    cudaStream_t stream,
    const float* points,
    const int* split_dims,
    int* neighbors,
    int n0,
    int k,
    int n_points
) {
    int n_threads = n_points - n0;
    int threads_per_block = 256;
    int n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
    query_preceding_neighbors_kernel<MAX_K, N_DIM><<<n_blocks, threads_per_block, 0, stream>>>(
        points, split_dims, neighbors, n0, k, n_points
    );
}