// refine_transpose.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "common.h"
#include "linalg.h"
#include "covariance.h"

template <int K_COARSE, int N_DIM>  
__global__ void refine_linear_transpose_kernel(
    const float *points, // (N, d)
    const uint32_t *neighbors, // (N, k)
    const float *cov_bins, // (R,)
    const float *cov_vals, // (B, R)
    float *values_tangent, // (B, N)
    float *xi_tangent, // (B, N)
    size_t n_points,
    size_t n_cov,
    size_t n_batches, // number of batches, affects cov_vals, values_tangent, and xi_tangent
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
    float *b_values_tangent = values_tangent + b * n_points;
    float *b_xi_tangent = xi_tangent + b * n_points;

    // define working variables, these should fit on register
    float pts[(K_COARSE + 1) * N_DIM]; // fine point + K coarse points
    float vec[K_COARSE + 1];
    float mat[((K_COARSE + 1) * (K_COARSE + 2)) / 2]; // lower triangular matrix for joint covariance

    // load coarse points, coarse values, fine points
    for (int i = 0; i < K_COARSE; ++i) {
        size_t neighbor_idx = neighbors[idx * K_COARSE + i];
        for (int j = 0; j < N_DIM; ++j) {
            pts[(K_COARSE - 1 - i) * N_DIM + j] = points[neighbor_idx * N_DIM + j];
        }
    }
    for (int j = 0; j < N_DIM; ++j) {
        pts[K_COARSE * N_DIM + j] = points[idx * N_DIM + j];
    }

    // build joint covariance matrix
    cov_lookup_matrix_triangular<K_COARSE, N_DIM>(pts, cov_bins, b_cov_vals, mat, n_cov); // Kcc
    cov_lookup_matrix_full<1, K_COARSE, N_DIM>(pts + K_COARSE * N_DIM, pts, cov_bins, b_cov_vals, mat + tri(K_COARSE, 0), n_cov); // Kfc
    mat[tri(K_COARSE, K_COARSE)] = cov_lookup(0.0f, cov_bins, b_cov_vals, n_cov); // Kff

    // factorize and write xi_tangent and values_tangent
    cholesky<K_COARSE + 1>(mat); // L
    float fine_value_tangent = b_values_tangent[idx];
    matmul<K_COARSE + 1, 1, 1>(mat + tri(K_COARSE, 0), &fine_value_tangent, vec);
    solve_cholesky_backward<K_COARSE, K_COARSE>(mat, vec);
    b_xi_tangent[idx] = mat[tri(K_COARSE, K_COARSE)] * fine_value_tangent;
    for (int i = 0; i < K_COARSE; ++i) {
        atomicAdd(b_values_tangent + neighbors[idx * K_COARSE + i], vec[i]);
    }
}

template <int K_COARSE, int N_DIM>
__host__ void refine_linear_transpose(
    cudaStream_t stream,
    const float *points,
    const uint32_t *neighbors,
    const uint32_t *offsets,
    const float *cov_bins,
    const float *cov_vals,
    const float *initial_cholesky,
    const float *values_tangent,
    float *values_tangent_buffer,
    float *xi_tangent,
    size_t n_points,
    size_t n_levels,
    size_t n_cov,
    size_t n_batches // batch dim only affects cov_vals, values_tangent, and all output buffers
) {
    size_t n_threads;
    size_t threads_per_block = 256;
    size_t n_blocks;

    // copy offsets to host
    uint32_t *offsets_host;
    offsets_host = (uint32_t*)malloc(n_levels * sizeof(uint32_t));
    if (offsets_host == nullptr) throw std::runtime_error("Failed to allocate memory for offsets on host");
    cudaMemcpy(offsets_host, offsets, n_levels * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // copy value tangents to buffer 
    cudaMemcpyAsync(values_tangent_buffer, values_tangent, n_batches * n_points * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // walk backwards through levels and compute tangents
    for (size_t level = n_levels; level-- > 0;) {
        uint32_t start_idx = offsets_host[level];
        uint32_t end_idx = (level + 1 < n_levels) ? offsets_host[level + 1] : n_points;
        n_threads = (end_idx - start_idx) * n_batches;
        n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
        refine_linear_transpose_kernel<K_COARSE, N_DIM><<<n_blocks, threads_per_block, 0, stream>>>(
            points, neighbors, cov_bins, cov_vals, values_tangent_buffer, xi_tangent, n_points, n_cov, n_batches, start_idx, n_threads
        );
    }

    // multiply initial_cholesky transpose by values_tangent_buffer to get initial values_tangent
    n_threads = n_batches * offsets_host[0];
    n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
    batched_transpose_matvec_kernel<<<n_blocks, threads_per_block, 0, stream>>>(
        initial_cholesky, values_tangent_buffer, xi_tangent, n_batches, offsets_host[0], n_points
    );

    free(offsets_host);
}