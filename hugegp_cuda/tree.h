// tree.h
#pragma once

#include <cuda_runtime.h>
#include "common.h"
#include "sort.h"
// #include "cubit.h"

// This implementation is optimized for minimal memory overhead and simplicity of implementation.
// Therefore, we do not use CUB sorts or segmented reductions. Instead, we express that algorithm
// as a sequence of in-place sorts using cubit. Unlike cudaKDTree, we split along the dimension with
// the largest range. This requires a sort for each dimension. More than an order of magnitude
// performance improvement is likely possible, but for typical applications the graph is constructed once
// and used many times so speed is not critical. Also, building the tree is not even the performance
// bottleneck for graph construction, since querying and depth computation take significant time.

__forceinline__ __device__ int compute_left(int current) {
    int level = floored_log2(current + 1);
    int n_level = 1 << level;
    return current + n_level;
}

__forceinline__ __device__ int compute_right(int current) {
    int level = floored_log2(current + 1);
    int n_level = 1 << level;
    return current + 2 * n_level;
}

__forceinline__ __device__ int compute_parent(int current) {
    int level = floored_log2(current + 1);
    int n_above = (1 << level) - 1;
    int n_parent_level = 1 << (level - 1);
    int parent = (current < n_above + n_parent_level) ? (current - n_parent_level) : (current - 2 * n_parent_level);
    return (current == 0) ? -1 : parent;  // root has no parent
}

__forceinline__ __device__ int compute_segment_start(int tag, int n_above, int n_remaining) {
    int n_level = n_above + 1;
    int q = n_remaining / n_level;
    int r = n_remaining % n_level;
    int i = tag - n_above;
    int start = (i < r) ? i * (q + 1) : (r * (q + 1) + (i - r) * q);
    return start + n_above;
}

__forceinline__ __device__ int compute_segment_end(int tag, int n_above, int n_remaining) {
    int n_level = n_above + 1;
    int q = n_remaining / n_level;
    int r = n_remaining % n_level;
    int i = tag - n_above;
    int end = (i < r) ? (i + 1) * (q + 1) : (r * (q + 1) + (i - r + 1) * q);
    return end + n_above;
}

template <int N_DIM>
__global__ void update_ranges(const int* tags, const float* points, float* ranges, int* split_dims, int dim, int n_above, int n_remaining) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_remaining) return;
    int i = n_above + idx;

    int tag = tags[i];
    int start = compute_segment_start(tag, n_above, n_remaining);
    int end = compute_segment_end(tag, n_above, n_remaining);
    float start_val = points[start * N_DIM + dim];
    float end_val = points[(end - 1) * N_DIM + dim];
    float dim_range = abs(end_val - start_val);
    if (dim_range > ranges[i]) {
        ranges[i] = dim_range;
        split_dims[i] = dim;
    }
}

__global__ void update_tags(int* tags, int* split_dims, float* ranges, int n_above, int n_remaining) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_remaining) return;
    int i = n_above + idx;

    int tag = tags[i];
    int start = compute_segment_start(tag, n_above, n_remaining);
    int end = compute_segment_end(tag, n_above, n_remaining);
    int midpoint = (start + end) / 2;
    if (i == midpoint) return;
    if (i < midpoint) tags[i] = compute_left(tag);
    else if (i > midpoint) tags[i] = compute_right(tag);
    split_dims[i] = 0; // reset for next iteration
    ranges[i] = 0.0f;
}

template <int N_DIM>
__host__ void build_tree(
    cudaStream_t stream,
    float* points, // (N, d)
    int* split_dims, // (N,)
    int* indices, // (N,)
    int* tags, // (N,)
    float* ranges, // (N,)
    int n_points
) {
    int n_levels = floored_log2(n_points) + 1;
    int n_threads;
    int threads_per_block = 256;
    int n_blocks;

    // initialize tags, split_dims, ranges to zero and indices to arange
    cudaMemsetAsync(tags, 0, n_points * sizeof(int), stream);
    cudaMemsetAsync(split_dims, 0, n_points * sizeof(int), stream);
    cudaMemsetAsync(ranges, 0, n_points * sizeof(float), stream);
    n_threads = n_points;
    n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
    arange_kernel<<<n_blocks, threads_per_block, 0, stream>>>(indices, n_points);

    for (int level = 0; level < n_levels; ++level) {
        int n_above = (1 << level) - 1;
        int n_remaining = n_points - n_above;
        n_threads = n_remaining;
        n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;

        // sort segments along each dimension and store the dimension with largest range in split_dims
        for (int dim = 0; dim < N_DIM; ++ dim) {
            sort_points<N_DIM>(stream, tags, points, split_dims, indices, ranges, dim, n_points);
            update_ranges<N_DIM><<<n_blocks, threads_per_block, 0, stream>>>(
                tags, points, ranges, split_dims, dim, n_above, n_remaining
            );
        }

        // sort along split dimension and update tags for next level
        sort_points<N_DIM>(stream, tags, points, split_dims, indices, ranges, int(-1), n_points);
        update_tags<<<n_blocks, threads_per_block, 0, stream>>>(
            tags, split_dims, ranges, n_above, n_remaining
        );
    }
}



