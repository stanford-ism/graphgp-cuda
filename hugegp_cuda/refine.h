// refine.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "common.h"
#include "linalg.h"
#include "covariance.h"

template <int MAX_K, int N_DIM>
__global__ void refine_kernel(
    const float* points, // (N, d)
    const uint32_t* neighbors, // (N - N0, k)
    const float* cov_bins, // (R,)
    const float* cov_vals, // (B, R)
    const float* xi, // (B, R)
    float* values, // (B, R)
    int k,
    size_t n_points,
    size_t n_initial,
    size_t n_cov,
    size_t n_batches, // number of batches, affects cov_vals, xi, and values
    size_t start_idx, // = offsets[level]
    size_t n_threads // = (end_idx - start_idx) * n_batches
) {
    // compute global index of the point to refine
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_threads) return;
    size_t n_points_per_batch = n_threads / n_batches;
    size_t b = tid / n_points_per_batch; // batch index
    size_t idx = start_idx + (tid % n_points_per_batch); // point index within batch

    // batched memory access
    const float *b_cov_vals = cov_vals + b * n_cov;
    const float *b_xi = xi + b * n_points;
    float *b_values = values + b * n_points;

    // define working variables, these should fit on register
    float pts[(MAX_K + 1) * N_DIM]; // fine point + K coarse points
    float vec[MAX_K + 1];
    float mat[((MAX_K + 1) * (MAX_K + 2)) / 2]; // lower triangular matrix for joint covariance

    // load neighbor points and either values or xi depending on if coarse or fine
    size_t k_coarse = 0;
    for (int i = 0; i < k; ++i) {
        size_t neighbor_idx = neighbors[(idx - n_initial) * k + i];
        if (neighbor_idx < start_idx) {
            vec[i] = b_values[neighbor_idx]; // coarse points use values
            k_coarse++; // coarse should all be before fine
        } else {
            vec[i] = b_xi[neighbor_idx]; // fine points use xi
        }
        for (int j = 0; j < N_DIM; ++j) {
            pts[i * N_DIM + j] = points[neighbor_idx * N_DIM + j];
        }
    }

    // load current point
    for (int j = 0; j < N_DIM; ++j) {
        pts[k * N_DIM + j] = points[idx * N_DIM + j];
    }
    vec[k] = b_xi[idx];

    // refinement operation
    cov_lookup_matrix<N_DIM>(pts, cov_bins, b_cov_vals, mat, k + 1, n_cov); // joint covariance
    cholesky(mat, k + 1); // factorize
    solve_cholesky_forward(mat, vec, k_coarse, 1); // x = Lcc^-1 @ v
    b_values[idx] = dot(mat + tri(k, 0), vec, k + 1); // v = L @ x
}



template <int MAX_K, int N_DIM>
__host__ void refine(
    cudaStream_t stream,
    const float* points,
    const uint32_t* neighbors,
    const uint32_t* offsets,
    const float* cov_bins,
    const float* cov_vals,
    const float* initial_values,
    const float* xi,
    float* values,
    int k,
    size_t n_points,
    size_t n_initial,
    size_t n_levels,
    size_t n_cov,
    size_t n_batches // batch dim only affects cov_vals, initial_values, xi, and output values
) {
    size_t n_threads;
    size_t threads_per_block = 256;
    size_t n_blocks;

    // copy offsets to host
    uint32_t *offsets_host;
    offsets_host = (uint32_t*)malloc(n_levels * sizeof(uint32_t));
    if (offsets_host == nullptr) throw std::runtime_error("Failed to allocate memory for offsets on host");
    cudaMemcpy(offsets_host, offsets, n_levels * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // copy initial values and xi to output values
    n_threads = n_batches * n_initial;
    n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
    batch_concat<<<n_blocks, threads_per_block, 0, stream>>>(
        values, initial_values, xi, n_batches, n_initial, n_points - n_initial
    );

    // iteratively refine levels
    for (size_t level = 0; level < n_levels; ++level) {
        uint32_t start_idx = offsets_host[level];
        uint32_t end_idx = level + 1 < n_levels ? offsets_host[level + 1] : n_points;
        n_threads = (end_idx - start_idx) * n_batches;
        n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
        refine_kernel<MAX_K, N_DIM><<<n_blocks, threads_per_block, 0, stream>>>(
            points, neighbors, cov_bins, cov_vals, xi, values, k, n_points, n_initial, n_cov, n_batches, start_idx, n_threads
        );
    }

    free(offsets_host);
}