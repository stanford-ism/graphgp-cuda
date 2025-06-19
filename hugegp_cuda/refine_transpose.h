// refine_transpose.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "common.h"
#include "linalg.h"
#include "covariance.h"

template <int K_COARSE, int N_DIM>
__global__ void refine_transpose_kernel(
    const float *points, // (N, d)
    const uint32_t *neighbors, // (N, k)
    const float *cov_r, // (R,)
    const float *cov_eval, // (R,)
    float *values_tangent, // (N,)
    float *xi_tangent, // (N,)
    size_t n_cov,
    size_t start_idx,
    size_t n_threads
) {
    // compute global index of the point to refine
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_threads) return;
    size_t idx = start_idx + tid;

    // define working variables, these should fit on register
    float fine_point[N_DIM];
    float neighbor_points[K_COARSE * N_DIM];
    float vec1[K_COARSE];
    float vec2[K_COARSE];
    float mat1[K_COARSE * K_COARSE];
    float mat2[K_COARSE * K_COARSE];

    // load fine point and coarse points
    #pragma unroll
    for (int i = 0; i < N_DIM; ++i) {
        fine_point[i] = points[idx * N_DIM + i];
    }
    #pragma unroll
    for (int i = 0; i < K_COARSE; ++i) {
        size_t neighbor_idx = neighbors[idx * K_COARSE + i];
        #pragma unroll
        for (int j = 0; j < N_DIM; ++j) {
            neighbor_points[i * N_DIM + j] = points[neighbor_idx * N_DIM + j];
        }
    }

    // compute conditional variance
    float variance = test_cov(0.0f); // Kff
    compute_test_cov_matrix<K_COARSE, 1, N_DIM>(neighbor_points, fine_point, vec1); // Kcf
    compute_test_cov_matrix<K_COARSE, K_COARSE, N_DIM>(neighbor_points, neighbor_points, mat1); // Kcc
    cholesky<K_COARSE>(mat1, mat2); // L
    solve_cholesky<K_COARSE>(mat2, vec1, vec2); // Kcc^-1 @ Kfc, must recompute is this order for stability
    variance -= dot<K_COARSE>(vec1, vec2); // Kff - Kcf @ (Kcc^-1 @ Kfc)
    variance = fmaxf(variance, 0.0f); // ensure non-negative variance

    // write to xi_tangent
    xi_tangent[idx] = sqrtf(variance) * values_tangent[idx];

    // compute addition to values_tangent
    float *fine_value_tangent = values_tangent + idx;
    matmul<K_COARSE, 1, 1>(vec1, fine_value_tangent, vec2); // Kcf @ vt
    solve_cholesky<K_COARSE>(mat2, vec2, vec1); // Kcc^-1 @ (Kcf @ vt)

    // add to values_tangent (multiple threads may write to the same index)
    for (int i = 0; i < K_COARSE; ++i) {
        atomicAdd(values_tangent + neighbors[idx * K_COARSE + i], vec1[i]);
    }


}

template <int K_COARSE, int N_DIM>
__host__ void refine_transpose(
    cudaStream_t stream,
    const float *points,
    const uint32_t *neighbors,
    const uint32_t *level_offsets,
    const float *cov_r,
    const float *cov,
    const float *values_tangent,
    float *xi_tangent,
    float *initial_values_tangent,
    size_t n_points,
    size_t n_levels,
    size_t n_cov
) {
    size_t n_threads;
    size_t threads_per_block = 256;
    size_t n_blocks;

    // copy level offsets to host
    uint32_t *level_offsets_host = (uint32_t*) malloc(n_levels * sizeof(uint32_t));
    if (level_offsets_host == nullptr) throw std::runtime_error("Host memory allocation failed");
    cudaMemcpy(level_offsets_host, level_offsets, n_levels * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // fill xi_tangent with zeros for initial level
    cudaMemsetAsync(xi_tangent, 0, level_offsets_host[0] * sizeof(float), stream);

    // allocate temporary buffer for propagating values_tangent
    float *final_values_tangent;
    cudaMalloc(&final_values_tangent, n_points * sizeof(float));
    // cudaMemsetAsync(final_values_tangent, 0, n_points * sizeof(float), stream);
    cudaMemcpyAsync(final_values_tangent, values_tangent, n_points * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // walk backwards through levels and compute tangents
    for (size_t i = 0; i < n_levels; ++i) {
        size_t level = n_levels - 1 - i;
        size_t start_idx = level_offsets_host[level];
        size_t end_idx = (level + 1 < n_levels) ? level_offsets_host[level + 1] : n_points;
        n_threads = end_idx - start_idx;
        n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
        refine_transpose_kernel<K_COARSE, N_DIM><<<n_blocks, threads_per_block, 0, stream>>>(
            points, neighbors, cov_r, cov, final_values_tangent, xi_tangent, n_cov, start_idx, n_threads
        );
    }

    // copy final values_tangent back to initial_values_tangent
    cudaMemcpyAsync(initial_values_tangent, final_values_tangent, level_offsets_host[0] * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // free
    free(level_offsets_host);
    cudaFree(final_values_tangent);
}