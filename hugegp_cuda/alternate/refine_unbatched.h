// refine.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "common.h"
#include "linalg.h"
#include "covariance.h"

template <int K_COARSE, int N_DIM>
__global__ void refine_kernel(
    const float* points, // (N, d)
    const uint32_t* neighbors, // (N, k)
    const float* cov_bins, // (R,)
    const float* cov_vals, // (B, R)
    const float* xi, // (B, R)
    float* values, // (B, R)
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
    float mat1[(K_COARSE * (K_COARSE + 1)) / 2]; // lower triangular matrix for covariance

    // load fine point, coarse points, and coarse values
    for (int i = 0; i < N_DIM; ++i) {
        fine_point[i] = points[idx * N_DIM + i];
    }
    for (int i = 0; i < K_COARSE; ++i) {
        size_t neighbor_idx = neighbors[idx * K_COARSE + i];
        vec1[i] = values[neighbor_idx]; // assumed to be generated already
        for (int j = 0; j < N_DIM; ++j) {
            neighbor_points[i * N_DIM + j] = points[neighbor_idx * N_DIM + j];
        }
    }

    // load covariance matrices
    cov_lookup_matrix_triangular<K_COARSE, N_DIM>(neighbor_points, cov_bins, cov_vals, mat1, n_cov); // Kcc
    cov_lookup_matrix_full<1, K_COARSE, N_DIM>(fine_point, neighbor_points, cov_bins, cov_vals, vec2, n_cov); // Kfc
    float variance = cov_lookup(0.0f, cov_bins, cov_vals, n_cov); // Kff

    // compute conditional mean
    cholesky<K_COARSE>(mat1); // L
    solve_cholesky<K_COARSE, 1>(mat1, vec1); // Kcc^-1 @ v
    float mean = dot<K_COARSE>(vec2, vec1); // Kfc @ (Kcc^-1 @ v)

    // compute conditional variance
    vec_copy<K_COARSE>(vec2, vec1); // Kcf = Kfc
    solve_cholesky<K_COARSE, 1>(mat1, vec1); // Kcc^-1 @ Kcf
    variance -= dot<K_COARSE>(vec2, vec1); // Kff - Kfc @ (Kcc^-1 @ Kcf)
    variance = fmaxf(variance, 0.0f); // better not to rely on this and add jitter to cov_vals[0]

    // write conditional value
    values[idx] = mean + sqrtf(variance) * xi[idx];
}

template <int K_COARSE, int N_DIM>
__host__ void refine(
    cudaStream_t stream,
    const float* points,
    const uint32_t* offsets,
    const uint32_t* neighbors,
    const float* cov_bins,
    const float* cov_vals,
    const float* initial_values,
    const float* xi,
    float* values,
    size_t n_points,
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

    // copy initial values to output buffer
    cudaMemcpyAsync(values, initial_values, offsets_host[0] * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // iteratively refine levels
    for (size_t level = 0; level < n_levels; ++level) {
        uint32_t start_idx = offsets_host[level];
        uint32_t end_idx = level + 1 < n_levels ? offsets_host[level + 1] : n_points;
        n_threads = end_idx - start_idx;
        n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
        refine_kernel<K_COARSE, N_DIM><<<n_blocks, threads_per_block, 0, stream>>>(
            points, neighbors, cov_bins, cov_vals, xi, values, n_cov, start_idx, n_threads
        );
    }

    free(offsets_host);
}