// refine_logdet.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "common.h"
#include "linalg.h"
#include "covariance.h"

template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__global__ void refine_logdet_kernel(
    const f_t* points, // (N, d)
    const i_t* neighbors, // (N - n0, k)
    const f_t* cov_bins, // (R,)
    const f_t* cov_vals, // (B, R)
    f_t* logdet, // (B,)
    size_t n0,
    size_t k,
    size_t n_points,
    size_t n_cov,
    size_t n_batches, // number of batches, affects cov_vals, xi, and values
    size_t start_idx, // = offsets[level]
    size_t n_threads // = (end_idx - start_idx) * n_batches
) {
    // compute global index of the point to refine
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;
    size_t n_points_per_batch = n_threads / n_batches;
    size_t b = tid / n_points_per_batch; // batch index
    size_t idx = start_idx + (tid % n_points_per_batch); // point index within batch

    // batched memory access
    const f_t *b_cov_vals = cov_vals + b * n_cov;
    f_t *b_logdet = logdet + b;

    // define working variables, these should fit on register
    f_t pts[(MAX_K + 1) * N_DIM]; // fine point + K coarse points
    f_t mat[((MAX_K + 1) * (MAX_K + 2)) / 2]; // lower triangular matrix for joint covariance

    // load neighbor points and values
    for (size_t i = 0; i < k; ++i) {
        size_t neighbor_idx = neighbors[(idx - n0) * k + i];
        for (size_t j = 0; j < N_DIM; ++j) {
            pts[i * N_DIM + j] = points[neighbor_idx * N_DIM + j];
        }
    }

    // load current point
    for (size_t j = 0; j < N_DIM; ++j) {
        pts[k * N_DIM + j] = points[idx * N_DIM + j];
    }

    // refinement operation
    cov_lookup_matrix(pts, cov_bins, b_cov_vals, mat, k + 1, N_DIM, n_cov); // joint covariance
    cholesky(mat, k + 1); // factorize
    atomicAdd(b_logdet, log(mat[tri(k, k)])); // logdet += log(std)
}



template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__host__ void refine_logdet(
    cudaStream_t stream,
    const f_t* points,
    const i_t* neighbors,
    const i_t* offsets,
    const f_t* cov_bins,
    const f_t* cov_vals,
    f_t* logdet, // (B,)
    size_t n0,
    size_t k,
    size_t n_points,
    size_t n_levels,
    size_t n_cov,
    size_t n_batches // batch dim only affects cov_vals, initial_values, xi, and output values
) {
    // copy offsets to host
    i_t *offsets_host;
    offsets_host = (i_t*)malloc(n_levels * sizeof(i_t));
    if (offsets_host == nullptr) throw std::runtime_error("Failed to allocate memory for offsets on host");
    cudaMemcpy(offsets_host, offsets, n_levels * sizeof(i_t), cudaMemcpyDeviceToHost);

    // initialize logdet to zero
    cudaMemsetAsync(logdet, 0, n_batches * sizeof(f_t), stream);

    // iteratively refine levels
    for (int level = 1; level < n_levels; ++level) {
        size_t start_idx = offsets_host[level - 1];
        size_t end_idx = offsets_host[level];
        size_t n_threads = (end_idx - start_idx) * n_batches;
        refine_logdet_kernel<MAX_K, N_DIM, i_t, f_t><<<cld(n_threads, 256), 256, 0, stream>>>(
            points,
            neighbors,
            cov_bins,
            cov_vals,
            logdet,
            n0,
            k,
            n_points,
            n_cov,
            n_batches,
            start_idx,
            n_threads);
    }

    free(offsets_host);
}