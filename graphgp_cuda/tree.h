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
// and used many times so speed is not critical.

__forceinline__ __device__ size_t compute_left(size_t current) {
    size_t level = floored_log2(current + 1);
    size_t n_level = 1 << level;
    return current + n_level;
}

__forceinline__ __device__ size_t compute_right(size_t current) {
    size_t level = floored_log2(current + 1);
    size_t n_level = 1 << level;
    return current + 2 * n_level;
}

__forceinline__ __device__ size_t compute_parent(size_t current) {
    size_t level = floored_log2(current + 1);
    size_t n_above = (1 << level) - 1;
    size_t n_parent_level = 1 << (level - 1);
    size_t parent = (current < n_above + n_parent_level) ? (current - n_parent_level) : (current - 2 * n_parent_level);
    return (current == 0) ? SIZE_MAX : parent;  // use SIZE_MAX as sentinel, almost certainly safe
}

__forceinline__ __device__ size_t compute_segment_start(size_t tag, size_t n_above, size_t n_remaining) {
    size_t n_level = n_above + 1;
    size_t q = n_remaining / n_level;
    size_t r = n_remaining % n_level;
    size_t i = tag - n_above;
    size_t start = (i < r) ? i * (q + 1) : (r * (q + 1) + (i - r) * q);
    return start + n_above;
}

__forceinline__ __device__ size_t compute_segment_end(size_t tag, size_t n_above, size_t n_remaining) {
    size_t n_level = n_above + 1;
    size_t q = n_remaining / n_level;
    size_t r = n_remaining % n_level;
    size_t i = tag - n_above;
    size_t end = (i < r) ? (i + 1) * (q + 1) : (r * (q + 1) + (i - r + 1) * q);
    return end + n_above;
}

template <typename i_t, typename f_t>
__global__ void update_ranges(
    const i_t* tags,
    const f_t* points_1d,
    f_t* ranges,
    i_t* split_dims,
    size_t dim,
    size_t n_above,
    size_t n_remaining
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_remaining) return;
    size_t i = n_above + idx;

    size_t tag = tags[i];
    size_t start = compute_segment_start(tag, n_above, n_remaining);
    size_t end = compute_segment_end(tag, n_above, n_remaining);
    f_t start_val = points_1d[start];
    f_t end_val = points_1d[end - 1];
    f_t dim_range = abs(end_val - start_val);
    if (dim_range > ranges[i]) {
        ranges[i] = dim_range;
        split_dims[i] = dim;
    }
}

template <typename i_t>
__global__ void update_tags(i_t* tags, size_t n_above, size_t n_remaining) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_remaining) return;
    size_t i = n_above + idx;

    size_t tag = tags[i];
    size_t start = compute_segment_start(tag, n_above, n_remaining);
    size_t end = compute_segment_end(tag, n_above, n_remaining);
    size_t midpoint = (start + end) / 2;
    if (i == midpoint) return;
    if (i < midpoint) tags[i] = compute_left(tag);
    else if (i > midpoint) tags[i] = compute_right(tag);
}

template <typename i_t, typename f_t>
__host__ void build_tree(
    cudaStream_t stream,
    const f_t* points_in, // (N, d)
    f_t* points, // (N, d)
    i_t* split_dims, // (N,)
    i_t* indices, // (N,)
    i_t* tags, // (N,)
    f_t* ranges, // (N,)
    size_t n_dim,
    size_t n_points
) {
    size_t n_levels = floored_log2(n_points) + 1;

    // initialize tags, split_dims, ranges to zero and indices to arange
    cudaMemsetAsync(tags, 0, n_points * sizeof(i_t), stream);
    cudaMemsetAsync(split_dims, 0, n_points * sizeof(i_t), stream);
    cudaMemsetAsync(ranges, 0, n_points * sizeof(f_t), stream);
    arange_kernel<i_t><<<cld(n_points, 256), 256, 0, stream>>>(indices, n_points);

    for (size_t level = 0; level < n_levels; ++level) {
        size_t n_above = (1 << level) - 1;
        size_t n_remaining = n_points - n_above;

        // compute split_dim with the largest range
        for (size_t dim = 0; dim < n_dim; ++dim) {
            permute_row<<<cld(n_points, 256), 256, 0, stream>>>(points_in, indices, points, n_dim, dim, 0, n_points);
            if (dim == 0) {
                sort(tags, points, indices, split_dims, n_points, stream); // tags move, must move split_dims and track points
                cudaMemsetAsync(split_dims + n_above, 0, n_remaining * sizeof(i_t), stream);
                cudaMemsetAsync(ranges + n_above, 0, n_remaining * sizeof(f_t), stream);
            } else {
                sort(tags, points, n_points, stream); // doesn't move tags, don't need to move split_dims or indices
            }
            update_ranges<<<cld(n_remaining, 256), 256, 0, stream>>>(
                tags, points, ranges, split_dims, dim, n_above, n_remaining);
        }

        // sort along split_dim and update tags
        permute_row_with_dims<<<cld(n_points, 256), 256, 0, stream>>>(points_in, indices, split_dims, points, n_dim, 0, n_points);
        sort(tags, points, indices, n_points, stream); // doesn't move tags, split_dims is same, must track points
        update_tags<<<cld(n_remaining, 256), 256, 0, stream>>>(tags, n_above, n_remaining);
    }
    
    // final permutation of points
    permute_rows<<<cld(n_points * n_dim, 256), 256, 0, stream>>>(points_in, indices, points, n_dim, n_points);
}