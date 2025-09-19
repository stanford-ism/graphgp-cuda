// depth.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "common.h"
#include "sort.h"
#include "cubit/cubit.h"

// alternative approach where we just update depths a bunch of times and hope for the best
template <typename i_t>
__global__ void update_depths_parallel(
    const i_t* neighbors,
    const i_t* old_depths,
    i_t* new_depths,
    i_t* changed, // single flag 0 or 1
    size_t n0,
    size_t k,
    size_t n_threads
) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;

    size_t neighbor_depth;
    size_t max_depth = 0;
    for (size_t j = 0; j < k; ++j) {
        neighbor_depth = old_depths[neighbors[tid * k + j]];
        if (neighbor_depth > max_depth) {
            max_depth = neighbor_depth;
        }
    }
    if (max_depth + 1 != old_depths[tid + n0]) {
        atomicOr(changed, 1); // reduction better in principle but this isn't the bottleneck
    }
    new_depths[tid + n0] = max_depth + 1;
}

// Compute longest path from root by repeatedly updating all nodes until the depth doesn't change.
// This is much faster than the serial approach for shallow graphs on large GPUs.
template <typename i_t>
__host__ void compute_depths_parallel(
    cudaStream_t stream,
    const i_t* neighbors,
    i_t* depths,
    i_t* temp,
    size_t n0,
    size_t k,
    size_t n_points
) {
    // initialize depth arrays
    i_t* new_depths = depths;
    i_t* old_depths = temp;
    cudaMemsetAsync(new_depths, 0, n_points * sizeof(i_t), stream);
    cudaMemsetAsync(old_depths, -1, n_points * sizeof(i_t), stream);
    cudaMemsetAsync(old_depths, 0, n0 * sizeof(i_t), stream);

    // flag for when depths change
    i_t changed = 1;
    i_t* d_changed;
    cudaMallocAsync(&d_changed, sizeof(i_t), stream);

    // loop until depths don't change
    while (changed == 1) {
        cudaMemsetAsync(d_changed, 0, sizeof(i_t), stream);
        i_t* tmp = new_depths; new_depths = old_depths; old_depths = tmp; // swap buffers
        update_depths_parallel<<<cld(n_points - n0, 256), 256, 0, stream>>>(
            neighbors, old_depths, new_depths, d_changed, n0, k, n_points - n0
        );
        cudaMemcpyAsync(&changed, d_changed, sizeof(i_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    if (new_depths != depths) {
        cudaMemcpyAsync(depths, new_depths, n_points * sizeof(i_t), cudaMemcpyDeviceToDevice, stream);
    }
}


template <typename i_t>
__global__ void reindex_neighbors(i_t* neighbors, const i_t* inv_permutation, size_t n_threads) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;
    neighbors[tid] = inv_permutation[neighbors[tid]];
}

template <typename i_t, typename f_t>
__host__ void order_by_depth(
    cudaStream_t stream,
    f_t* points, // (N, d)
    i_t* indices, // (N,)
    i_t* neighbors, // (N - n0, k)
    i_t* depths, // (N,)
    i_t* permutation, // (N,)
    i_t* temp_int, // (N,)
    f_t* temp_float, // (N,) can alias with temp_int
    size_t n0,
    size_t k,
    size_t n_points,
    size_t n_dim
) {

    // sort by depth, tracking permutation
    arange_kernel<i_t><<<cld(n_points, 256), 256, 0, stream>>>(permutation, n_points);
    sort(depths + n0, permutation + n0, n_points - n0, stream);

    // permute arrays one-by-one
    permute(stream, points, temp_float, permutation, n_dim, 0, n_points);
    permute(stream, indices, temp_int, permutation, 1, 0, n_points);
    permute(stream, neighbors, temp_int, permutation + n0, k, n0, n_points - n0); // tricky, first n0 don't exist

    // update neighbors to point to new indices
    compute_inverse_permutation<<<cld(n_points, 256), 256, 0, stream>>>(permutation, temp_int, n_points);
    reindex_neighbors<<<cld((n_points - n0) * k, 256), 256, 0, stream>>>(neighbors, temp_int, (n_points - n0) * k);
}