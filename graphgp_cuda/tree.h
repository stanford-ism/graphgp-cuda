// tree.h
#pragma once

#include <cuda_runtime.h>
#include "common.h"
#include "sort.h"
// #include "cubit.h"

// This implementation is optimized for minimal memory overhead and ease of maintenance.
// Therefore, we do not use CUB sorts or segmented reductions. Instead, we express the algorithm
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

__global__ void update_ranges(
    const int* tags,
    const float* points_1d,
    float* ranges,
    int* split_dims,
    int dim,
    int n_above,
    int n_threads
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_threads) return;
    int i = n_above + idx;

    int tag = tags[i];
    int start = compute_segment_start(tag, n_above, n_threads);
    int end = compute_segment_end(tag, n_above, n_threads);
    float start_val = points_1d[start];
    float end_val = points_1d[end - 1];
    float dim_range = abs(end_val - start_val);
    if (dim_range > ranges[i]) {
        ranges[i] = dim_range;
        split_dims[i] = dim;
    }
}

__global__ void update_tags(int* tags, int n_above, int n_threads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_threads) return;
    int i = n_above + idx;

    int tag = tags[i];
    int start = compute_segment_start(tag, n_above, n_threads);
    int end = compute_segment_end(tag, n_above, n_threads);
    int midpoint = (start + end) / 2;
    if (i == midpoint) return;
    if (i < midpoint) tags[i] = compute_left(tag);
    else if (i > midpoint) tags[i] = compute_right(tag);
}

__host__ void build_tree(
    cudaStream_t stream,
    const float* points_in, // (N, d)
    float* points, // (N, d)
    int* split_dims, // (N,)
    int* indices, // (N,)
    int* tags, // (N,)
    float* ranges, // (N,)
    int n_dim,
    int n_points
) {
    int n_levels = floored_log2(n_points) + 1;

    // initialize tags, split_dims, ranges to zero and indices to arange
    cudaMemsetAsync(tags, 0, n_points * sizeof(int), stream);
    cudaMemsetAsync(split_dims, 0, n_points * sizeof(int), stream);
    cudaMemsetAsync(ranges, 0, n_points * sizeof(float), stream);
    CUDA_LAUNCH(arange_kernel<int>, n_points, stream, indices);

    for (int level = 0; level < n_levels; ++level) {
        int n_above = (1 << level) - 1;
        int n_remaining = n_points - n_above;

        // compute split_dim with the largest range
        for (int dim = 0; dim < n_dim; ++dim) {
            CUDA_LAUNCH(copy_row_indices, n_points, stream, points_in, indices, points, n_dim, dim);
            if (dim == 0) {
                sort(tags, points, indices, split_dims, n_points, stream); // tags move, must move split_dims and track points
                cudaMemsetAsync(split_dims + n_above, 0, n_remaining * sizeof(int), stream);
                cudaMemsetAsync(ranges + n_above, 0, n_remaining * sizeof(float), stream);
            } else {
                sort(tags, points, n_points, stream); // doesn't move tags, don't need to move split_dims or indices
            }
            CUDA_LAUNCH(update_ranges, n_remaining, stream, tags, points, ranges, split_dims, dim, n_above);
        }

        // sort along split_dim and update tags
        CUDA_LAUNCH(copy_row_indices_split_dims, n_points, stream, points_in, indices, split_dims, points, n_dim);
        sort(tags, points, indices, n_points, stream); // doesn't move tags, split_dims is same, must track points
        CUDA_LAUNCH(update_tags, n_remaining, stream, tags, n_above);
    }
    
    // final permutation of points
    CUDA_LAUNCH(permute_rows, n_points * n_dim, stream, points_in, points, indices, n_dim); 
}