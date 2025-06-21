// refine_transpose.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "common.h"
#include "linalg.h"
#include "covariance.h"

#include <cub/cub.cuh>

template <int K_COARSE, int N_DIM>
__global__ void refine_transpose_kernel(
    const float *points, // (N, d)
    const uint32_t *neighbors, // (N, k)
    const float *cov_bins, // (R,)
    const float *cov_vals, // (R,)
    const float *values_tangent, // (N,)
    float *temp_val, // (N, k)
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
    const float *fine_value_tangent = values_tangent + idx;
    matmul<K_COARSE, 1, 1>(vec1, fine_value_tangent, vec2); // Kcf @ vt
    solve_cholesky<K_COARSE>(mat2, vec2, vec1); // Kcc^-1 @ (Kcf @ vt)

    // add to values_tangent (multiple threads may write to the same index)
    for (int i = 0; i < K_COARSE; ++i) {
        // atomicAdd(values_tangent + neighbors[idx * K_COARSE + i], vec1[i]);
        temp_val[idx * K_COARSE + i] = vec1[i];
    }
}

template <int K_COARSE, int N_DIM>
__host__ void refine_transpose(
    cudaStream_t stream,
    const float *points,
    const uint32_t *offsets,
    const uint32_t *neighbors,
    const float *cov_bins,
    const float *cov_vals,
    const float *values_tangent,
    float *initial_values_tangent,
    float *xi_tangent,
    size_t n_points,
    size_t n_levels,
    size_t n_cov
) {
    size_t n_threads;
    size_t threads_per_block = 256;
    size_t n_blocks;

    // fill xi_tangent with zeros for initial level
    size_t n_initial;
    cudaMemcpy(&n_initial, offsets, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemsetAsync(xi_tangent, 0, n_initial * sizeof(float), stream);

    // allocate temporary buffers for propagating values_tangent
    float *values_tangent_buffer;
    float *temp_val;
    cudaMalloc(&values_tangent_buffer, n_points * sizeof(float));
    cudaMalloc(&temp_val, n_points * K_COARSE * sizeof(float));
    cudaMemcpyAsync(values_tangent_buffer, values_tangent, n_points * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // walk backwards through levels and compute tangents
    for (size_t i = 0; i < n_levels; ++i) {
        size_t level = n_levels - 1 - i;
        uint32_t start_idx;
        uint32_t end_idx = n_points;
        cudaMemcpy(&start_idx, offsets + level, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (level + 1 < n_levels) {
            cudaMemcpy(&end_idx, offsets + (level + 1), sizeof(uint32_t), cudaMemcpyDeviceToHost);
        }
        n_threads = end_idx - start_idx;
        n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
        refine_transpose_kernel<K_COARSE, N_DIM><<<n_blocks, threads_per_block, 0, stream>>>(
            points, neighbors, cov_r, cov, values_tangent_buffer, temp_val, xi_tangent, n_cov, start_idx, n_threads
        );

        // make sort destination arrays
        float *sorted_val;
        uint32_t *sorted_idx;
        cudaMalloc(&sorted_val, n_threads * K_COARSE * sizeof(float));
        cudaMalloc(&sorted_idx, n_threads * K_COARSE * sizeof(uint32_t));

        // sort by index
        void *temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        size_t start_shift = start_idx * K_COARSE;
        cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_bytes, neighbors + start_shift, sorted_idx, temp_val + start_shift, sorted_val, n_threads * K_COARSE, 0, sizeof(uint32_t) * 8, stream);
        cudaMalloc(&temp_storage, temp_storage_bytes);
        cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_bytes, neighbors + start_shift, sorted_idx, temp_val + start_shift, sorted_val, n_threads * K_COARSE, 0, sizeof(uint32_t) * 8, stream);

        // make reduction destination arrays
        float *idx_val;
        uint32_t *unique_idx;
        size_t *n_unique;
        cudaMalloc(&idx_val, n_points * sizeof(float));
        cudaMalloc(&unique_idx, n_points * sizeof(uint32_t));
        cudaMalloc(&n_unique, sizeof(size_t));

        // reduce by key
        void *temp_storage_reduction = nullptr;
        size_t temp_storage_reduction_bytes = 0;
        cub::DeviceReduce::ReduceByKey(temp_storage_reduction, temp_storage_reduction_bytes, sorted_idx, unique_idx, sorted_val, idx_val, n_unique, cub::Sum(), n_threads * K_COARSE, stream);
        cudaMalloc(&temp_storage_reduction, temp_storage_reduction_bytes);
        cub::DeviceReduce::ReduceByKey(temp_storage_reduction, temp_storage_reduction_bytes, sorted_idx, unique_idx, sorted_val, idx_val, n_unique, cub::Sum(), n_threads * K_COARSE, stream);

        // copy reduced values back to final_values_tangent
        cudaStreamSynchronize(stream);
        cudaMemcpy(&n_threads, n_unique, sizeof(size_t), cudaMemcpyDeviceToHost);
        n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
        add_to_indices<<<n_blocks, threads_per_block, 0, stream>>>(
            idx_val, unique_idx, values_tangent_buffer, n_threads
        );

        cudaStreamSynchronize(stream);
        cudaFree(idx_val);
        cudaFree(unique_idx);
        cudaFree(n_unique);
        cudaFree(sorted_val);
        cudaFree(sorted_idx);
        cudaFree(temp_storage_reduction);
        cudaFree(temp_storage);
    }

    // copy final values_tangent back to initial_values_tangent
    cudaMemcpyAsync(initial_values_tangent, values_tangent_buffer, offsets_host[0] * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // free
    cudaStreamSynchronize(stream);
    cudaFree(values_tangent_buffer);
    cudaFree(temp_val);
}