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
    const float* xi, // (N,)
    const uint32_t* indices, // (N,)
    const uint32_t* neighbors, // (N, k)
    const float* cov_distances, // (R,)
    const float* cov_values, // (R,)
    float* values, // (N,)
    size_t n_cov,
    size_t start_idx,
    size_t n_threads
) {
    // compute global index of the point to refine
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_threads) return;
    size_t idx = start_idx + tid;
    size_t current_index = indices[idx];

    // define working variables, these should fit on register
    float fine_point[N_DIM];
    float neighbor_points[K_COARSE * N_DIM];
    float vec1[K_COARSE];
    float vec2[K_COARSE];
    float mat1[K_COARSE * K_COARSE];
    float mat2[K_COARSE * K_COARSE];

    // load fine point, coarse points, and coarse values
    #pragma unroll
    for (int i = 0; i < N_DIM; ++i) {
        fine_point[i] = points[current_index * N_DIM + i];
    }
    #pragma unroll
    for (int i = 0; i < K_COARSE; ++i) {
        size_t neighbor_idx = neighbors[idx * K_COARSE + i];
        vec1[i] = values[neighbor_idx]; // assumed to be generated already
        size_t original_neighbor_idx = indices[neighbor_idx];
        #pragma unroll
        for (int j = 0; j < N_DIM; ++j) {
            neighbor_points[i * N_DIM + j] = points[original_neighbor_idx * N_DIM + j];
        }
    }

    // compute conditional mean
    compute_test_cov_matrix<K_COARSE, K_COARSE, N_DIM>(neighbor_points, neighbor_points, mat1); // Kcc
    cholesky<K_COARSE>(mat1, mat2); // L
    solve_cholesky<K_COARSE>(mat2, vec1, vec2); // Kcc^-1 @ v
    compute_test_cov_matrix<1, K_COARSE, N_DIM>(fine_point, neighbor_points, vec1); // Kfc
    float mean = dot<K_COARSE>(vec1, vec2); // Kfc @ (Kcc^-1 @ v)

    // compute conditional variance
    float variance = test_cov(0.0f);
    solve_cholesky<K_COARSE>(mat2, vec1, vec2); // Kcc^-1 @ Kfc
    variance -= dot<K_COARSE>(vec1, vec2); // Kff - Kcf @ (Kcc^-1 @ Kfc)
    variance = fmaxf(variance, 0.0f); // ensure non-negative variance

    // write conditional value
    values[idx] = mean + sqrtf(variance) * xi[current_index];
}

template <int K_COARSE, int N_DIM>
__host__ void refine(
    cudaStream_t stream,
    const float* points,
    const float* xi,
    const uint32_t* indices,
    const uint32_t* neighbors,
    const uint32_t* level_offsets,
    const float* cov_distances,
    const float* cov_values,
    const float* initial_values,
    float* values,
    size_t n_points,
    size_t n_levels,
    size_t n_cov
) {
    size_t n_threads;
    size_t threads_per_block = 256;
    size_t n_blocks;

    // copy level offsets to host
    uint32_t *level_offsets_host = (uint32_t*) malloc(n_levels * sizeof(uint32_t));
    cudaMemcpy(level_offsets_host, level_offsets, n_levels * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // allocate values buffer in topological order and copy initial values
    float *values_ordered;
    cudaMalloc(&values_ordered, n_points * sizeof(float));
    cudaMemcpy(values_ordered, initial_values, (level_offsets_host[0] - 1) * sizeof(float), cudaMemcpyDeviceToDevice);

    // iteratively refine levels
    for (size_t level = 0; level < n_levels; ++level) {
        size_t start_idx = level_offsets_host[level];
        size_t end_idx = (level + 1 < n_levels) ? level_offsets_host[level + 1] : n_points;
        n_threads = end_idx - start_idx;
        n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
        refine_kernel<K_COARSE, N_DIM><<<n_blocks, threads_per_block, 0, stream>>>(
            points, xi, indices, neighbors, cov_distances, cov_values, values_ordered, n_cov, start_idx, n_threads
        );
    }

    // copy to output array in correct order
    n_threads = n_points;
    n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
    restore_order<float><<<n_blocks, threads_per_block, 0, stream>>>(
        values_ordered, indices, values, n_threads
    );

    // free memory
    free(level_offsets_host);
    cudaFree(values_ordered);
}