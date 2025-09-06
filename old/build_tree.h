// build_tree.h
#pragma once

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include "common.h"


__forceinline__ __device__ int compute_left(int node) {
    int level = floored_log2(node + 1);
    int n_level = 1 << level;
    return node + n_level;
}

__forceinline__ __device__ int compute_right(int node) {
    int level = floored_log2(node + 1);
    int n_level = 1 << level;
    return node + 2 * n_level;
}

__forceinline__ __device__ int compute_parent(int node) {
    int level = floored_log2(node + 1);
    int n_above = (1 << level) - 1;
    int n_parent_level = 1 << (level - 1);
    int parent = (node < n_above + n_parent_level) ? (node - n_parent_level) : (node - 2 * n_parent_level);
    parent = (node == 0) ? UINT32_MAX : parent;  // root has no parent
    return parent;
}

__forceinline__ __device__ int segment_left(int tag) {
    return tag;
}

__forceinline__ __device__ int segment_right(int tag) {
    return tag + 1;
}

__forceinline__ __device__ int update_tag(int tag) {
    return tag + 2;
}

template <int N_DIM>
__host__ void build_tree(
    cudaStream_t stream,
    const float* original_points, // (N, d)
    float* points, // (N, d)
    int* split_dims, // (N,)
    int* indices, // (N,)

    int* tags,
    int* tags_alt,
    float* points_alt,
    int* indices_alt,
    float* spreads,
    int* perm,
    int* perm_alt,
    int n_points
) {
    int n_levels = floored_log2(n_points) + 1;
    int n_threads;
    int threads_per_block = 256;
    int n_blocks;

    
    // Initialize tags to zero, points to original points, and indices to arange
    n_threads = n_points;
    n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
    cudaMemsetAsync(tags, 0, n_points * sizeof(int), stream);
    cudaMemcpyAsync(points, original_points, n_points * sizeof(float) * N_DIM, stream);
    arange_kernel<<<n_blocks, threads_per_block, 0, stream>>>(indices, n_points);

    // allocate temporary memory for CUB ops
    // a bit risky, not 100% sure this is always enough since docs are unclear, but it works
    void *temp_storage;
    size_t temp_storage_bytes = 4 * n_points * sizeof(int);
    cudaMalloc(&temp_storage, temp_storage_bytes);

    // NOTE: This is clearly not the most efficient implementation. We would ideally use
    // segmented ops for everything, as is done in jaxkd-cuda. However, this turned out to be
    // a pain and actually requires a lot of memory overhead. Since tree construction speed is
    // not critical for GraphGP, we use a simpler approach which requires only repeated sorts.
    // In the future these could be converted to in-place sorts for memory efficiency.
    for (int i = 0; i < n_levels; ++i) {
        for (int dim = 0; dim < N_DIM; ++dim) {

            // Arrays to keep throughout loop
            //  - points (N, d)
            //  - split_dims (N,)
            //  - indices (N,)

            // Additional buffers
            //  - f (max(3, d), N) float, with (f1, f2) a double buffer
            //  - i (2, N) int, with (i1, i2) a double buffer
            //  - t (2, N) int, with (t1, t2) a double buffer

            // for each dim
            //     0. initialize f3 as zero, to be used as spread
            //     1. copy (points[dim], tags) -> (f1, i1)
            //     2. sort (f1, i1) using (f2, i2) buffer
            //     3. sort (i1, f1) using (i2, f2) buffer
            //     4. update spread into f3 and split_dims into split_dims

            // copy (points[split_dim], arange) -> (f1, i1)
            // sort (f1, i1) using (f2, i2) buffer ("sort by value along split dimension")
            // gather t1 -> t2 using i1, swap t2 <-> t1
            // sort (t1, i1) using (t2, i2) buffer ("sort by tag")
            // gather (points, indices) -> (f123, t2) using t1
            // copy (f123, t2) -> (points, indices)
            // update tags

            // summary:
            // We need to pack a tag + point value struct, maybe we compute the tag on the fly?
            // For each dimension we will construct and sort, using an extra float buffer for spread and then compute split_dims
            // Then we need to pack tag + value along dim for key and points, indices for the value, then sort
            // Then we need to update tags
            // Then we need to sort again with the split_dims


        }
    }

}

