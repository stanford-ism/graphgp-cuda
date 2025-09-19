// query.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "common.h"
#include "tree.h"

template <typename i_t, typename f_t>
__forceinline__ __device__ void insert_neighbor(
    i_t* neighbors,
    f_t* distances,
    size_t current_index,
    f_t current_distance,
    size_t k
) {
    size_t i = k - 1;
    // ensure well-defined ordering by putting earlier indices first
    while ((i > 0) && ((current_distance < distances[i-1]) || (current_distance == distances[i-1] && current_index < neighbors[i-1]))) {
        neighbors[i] = neighbors[i-1];
        distances[i] = distances[i-1];
        --i;
    }
    neighbors[i] = current_index;
    distances[i] = current_distance;
}


template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__global__ void query_preceding_neighbors_kernel(
    const f_t* points, // (N, d)
    const i_t* split_dims, // (N,)
    i_t* neighbors_out, // (Q, k)
    size_t n0,
    size_t k,
    size_t n_threads
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_threads) return;
    size_t query_idx = idx + n0;

    // load query point
    f_t query[N_DIM];
    for (size_t i = 0; i < N_DIM; ++i) {
        query[i] = points[query_idx * N_DIM + i];
    }

    // initialize neighbor arrays
    i_t neighbors[MAX_K];
    f_t distances[MAX_K];
    f_t max_distance = INFINITY;
    for (size_t i = 0; i < k; ++i) {
        distances[i] = max_distance;
        neighbors[i] = 0;
    }

    // set up traversal variables
    size_t current = 0;
    size_t root_parent = compute_parent(current);
    size_t previous = root_parent;
    size_t next = 0;

    // traverse until we return to root
    while (current != root_parent) {
        size_t parent = compute_parent(current);

        // update neighbor array if necessary
        if (previous == parent) {
            f_t current_distance = compute_square_distance(points + current * N_DIM, query, N_DIM);
            if (current_distance < max_distance) {
                insert_neighbor(neighbors, distances, current, current_distance, k);
                max_distance = distances[k - 1];
            }
        }

        // locate children and determine if far child in range
        i_t split_dim = split_dims[current];
        f_t split_distance = query[split_dim] - points[current * N_DIM + split_dim];
        size_t near_child = (split_distance < 0) ? compute_left(current) : compute_right(current);
        size_t far_child = (split_distance < 0) ? compute_right(current) : compute_left(current);
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
    for (size_t i = 0; i < k; ++i) {
        neighbors_out[idx * k + i] = neighbors[i];
    }
}

template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__global__ void query_neighbors_kernel(
    const f_t* points, // (N, d)
    const i_t* split_dims, // (N,)
    const i_t* query_indices, // (Q,)
    const i_t* max_indices, // (Q,)
    i_t* neighbors_out, // (Q, k)
    size_t k,
    size_t n_points, 
    size_t n_queries
) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_queries) return;
    size_t query_index = query_indices[tid];
    i_t max_index = max_indices[tid];

    // load query point
    f_t query[N_DIM];
    for (size_t i = 0; i < N_DIM; ++i) {
        query[i] = points[query_index * N_DIM + i];
    }

    // initialize neighbor arrays
    i_t neighbors[MAX_K];
    f_t distances[MAX_K];
    f_t max_distance = INFINITY;
    for (size_t i = 0; i < k; ++i) {
        distances[i] = max_distance;
        neighbors[i] = 0;
    }

    // set up traversal variables
    size_t current = 0;
    size_t root_parent = compute_parent(current);
    size_t previous = root_parent;
    size_t next = 0;

    // traverse until we return to root
    while (current != root_parent) {
        size_t parent = compute_parent(current);

        // update neighbor array if necessary
        if (previous == parent) {
            f_t current_distance = compute_square_distance(points + current * N_DIM, query, N_DIM);
            if (current_distance < max_distance) {
                insert_neighbor(neighbors, distances, current, current_distance, k);
                max_distance = distances[k - 1];
            }
        }

        // locate children and determine if far child in range
        i_t split_dim = split_dims[current];
        f_t split_distance = query[split_dim] - points[current * N_DIM + split_dim];
        size_t near_child = (split_distance < 0) ? compute_left(current) : compute_right(current);
        size_t far_child = (split_distance < 0) ? compute_right(current) : compute_left(current);
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
    for (size_t i = 0; i < k; ++i) {
        neighbors_out[tid * k + i] = neighbors[i];
    }
}

// template <int MAX_K, int N_DIM>
// __host__ void query_neighbors(
//     cudaStream_t stream,
//     const float* points,
//     const int* split_dims,
//     const int* query_indices,
//     const int* max_indices,
//     int* neighbors,
//     int k,
//     int n_points,
//     int n_queries
// ) {
//     CUDA_LAUNCH(query_neighbors_kernel<MAX_K, N_DIM>, n_queries, stream, points, split_dims, query_indices, max_indices, neighbors, k, n_points);
// }

// template <int MAX_K, int N_DIM>
// __host__ void query_preceding_neighbors(
//     cudaStream_t stream,
//     const float* points,
//     const int* split_dims,
//     int* neighbors,
//     int n0,
//     int k,
//     int n_points
// ) {
//     CUDA_LAUNCH(query_preceding_neighbors_kernel<MAX_K, N_DIM>, n_points - n0, stream, points, split_dims, neighbors, n0, k, n_points);
// }