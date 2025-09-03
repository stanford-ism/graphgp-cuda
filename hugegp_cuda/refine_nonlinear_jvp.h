// refine_nonlinear_jvp.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "common.h"
#include "linalg.h"
#include "covariance.h"

template <int MAX_K, int N_DIM>
__global__ void refine_nonlinear_jvp_kernel(
    const float* points, // (N, d)
    const int* neighbors, // (N - n0, k)
    const float* cov_bins, // (R,)
    const float* cov_vals, // (B, R)
    const float* xi, // (B, R)
    const float* cov_vals_tangent, // (B, R)
    const float* xi_tangent, // (B, R)
    float* values, // (B, R)
    float* values_tangent, // (B, R)
    int n0,
    int k,
    int n_points,
    int n_cov,
    int n_batches, // number of batches, affects cov_vals, xi, and values
    int start_idx, // = offsets[level]
    int n_threads // = (end_idx - start_idx) * n_batches
) {
    // compute global index of the point to refine
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_threads) return;
    int n_points_per_batch = n_threads / n_batches;
    int b = tid / n_points_per_batch; // batch index
    size_t idx = start_idx + (tid % n_points_per_batch); // point index within batch

    // batched memory access
    const float *b_cov_vals = cov_vals + b * n_cov;
    const float *b_xi = xi + b * (n_points - n0);
    const float *b_cov_vals_tangent = cov_vals_tangent + b * n_cov;
    const float *b_xi_tangent = xi_tangent + b * (n_points - n0);
    float *b_values = values + b * n_points;
    float *b_values_tangent = values_tangent + b * n_points;

    // define working variables, these should fit on register
    float pts[(MAX_K + 1) * N_DIM]; // fine point + K coarse points
    float vec[MAX_K + 1];
    float vec_tangent[MAX_K + 1];
    float mat[((MAX_K + 1) * (MAX_K + 2)) / 2]; // lower triangular matrix for joint covariance
    float mat_tangent[((MAX_K + 1) * (MAX_K + 2)) / 2];

    // load neighbor points and values
    for (int i = 0; i < k; ++i) {
        size_t neighbor_idx = neighbors[(idx - n0) * k + i];
        vec[i] = b_values[neighbor_idx]; // coarse points use values
        vec_tangent[i] = b_values_tangent[neighbor_idx];
        for (int j = 0; j < N_DIM; ++j) {
            pts[i * N_DIM + j] = points[neighbor_idx * N_DIM + j];
        }
    }

    // load current point
    for (int j = 0; j < N_DIM; ++j) {
        pts[k * N_DIM + j] = points[idx * N_DIM + j];
    }
    vec[k] = b_xi[idx - n0];
    vec_tangent[k] = b_xi_tangent[idx - n0];

    
    // load and factorize joint covariance, with jvp
    cov_lookup_matrix<N_DIM>(pts, cov_bins, b_cov_vals, mat, k + 1, n_cov); // K
    cov_lookup_matrix<N_DIM>(pts, cov_bins, b_cov_vals_tangent, mat_tangent, k + 1, n_cov); // dK
    cholesky(mat, k + 1); // L
    cholesky_jvp(mat, mat_tangent, k + 1); // dL

    // compute values and linear jvp
    solve_cholesky_forward(mat, vec, k, 1);
    solve_cholesky_forward(mat, vec_tangent, k, 1);
    b_values[idx] = dot(mat + tri(k, 0), vec, k + 1);
    b_values_tangent[idx] = dot(mat + tri(k, 0), vec_tangent, k + 1);

    // nonlinear from dot product
    b_values_tangent[idx] += dot(mat_tangent + tri(k, 0), vec, k + 1);

    // nonlinear from inverse
    apply_cholesky(mat_tangent, vec, k, 1);
    solve_cholesky_forward(mat, vec, k, 1);
    b_values_tangent[idx] -= dot(mat + tri(k, 0), vec, k);
}



template <int MAX_K, int N_DIM>
__host__ void refine_nonlinear_jvp(
    cudaStream_t stream,
    const float* points,
    const int* neighbors,
    const int* offsets,
    const float* cov_bins,
    const float* cov_vals,
    const float* initial_values,
    const float* xi,
    const float* cov_vals_tangent,
    const float* initial_values_tangent,
    const float* xi_tangent,
    float* values,
    float* values_tangent,
    int n0,
    int k,
    int n_points,
    int n_levels,
    int n_cov,
    int n_batches // batch dim only affects cov_vals, initial_values, xi, and output values
) {
    int n_threads;
    int threads_per_block = 256;
    int n_blocks;

    // copy offsets to host
    int *offsets_host;
    offsets_host = (int*)malloc(n_levels * sizeof(int));
    if (offsets_host == nullptr) throw std::runtime_error("Failed to allocate memory for offsets on host");
    cudaMemcpy(offsets_host, offsets, n_levels * sizeof(int), cudaMemcpyDeviceToHost);

    // copy initial values to output values
    n_threads = n_batches * n0;
    n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
    batch_copy<<<n_blocks, threads_per_block, 0, stream>>>(values, initial_values, n_batches, n0, n_points);
    batch_copy<<<n_blocks, threads_per_block, 0, stream>>>(values_tangent, initial_values_tangent, n_batches, n0, n_points);

    // iteratively refine levels
    for (int level = 1; level < n_levels; ++level) {
        int start_idx = offsets_host[level - 1];
        int end_idx = offsets_host[level];
        n_threads = (end_idx - start_idx) * n_batches;
        n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
        refine_nonlinear_jvp_kernel<MAX_K, N_DIM><<<n_blocks, threads_per_block, 0, stream>>>(
            points, neighbors, cov_bins, cov_vals, xi, cov_vals_tangent, xi_tangent, values, values_tangent, n0, k, n_points, n_cov, n_batches, start_idx, n_threads
        );
    }

    free(offsets_host);
}