// build_tree.h
#pragma once

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include "common.h"


template <int N_DIM>
__host__ void build_tree(
    cudaStream_t stream,
    const float* original_points, // (N, d)
    float* points, // (N, d)
    int* split_dims, // (N,)
    int* indices, // (N,)
    float* points_buffer, // (N, d)
    int* indices_buffer, // (N,)
    size_t n_points
) {
    size_t n_levels = floored_log2(n_points) + 1;
    size_t n_threads;
    size_t threads_per_block = 256;
    size_t n_blocks;

    

    //
}

__forceinline__ __device__ uint32_t compute_left(uint32_t node) {
    int level = floored_log2(node + 1);
    int n_level = 1 << level;
    return node + n_level;
}

__forceinline__ __device__ uint32_t compute_right(uint32_t node) {
    int level = floored_log2(node + 1);
    int n_level = 1 << level;
    return node + 2 * n_level;
}

__forceinline__ __device__ uint32_t compute_parent(uint32_t node) {
    int level = floored_log2(node + 1);
    int n_above = (1 << level) - 1;
    int n_parent_level = 1 << (level - 1);
    int parent = (node < n_above + n_parent_level) ? (node - n_parent_level) : (node - 2 * n_parent_level);
    parent = (node == 0) ? UINT32_MAX : parent;  // root has no parent
    return parent;
}

__forceinline__ __device__ int segment_left(int node, )