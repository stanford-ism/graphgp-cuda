// refine_nonlinear_vjp.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "common.h"
#include "linalg.h"
#include "covariance.h"

template <int MAX_K, int N_DIM>  
__global__ void refine_nonlinear_vjp_kernel(
    const float* points, // (N, d)
    const int* neighbors, // (N - n0, k)
    const float* cov_bins, // (R,)
    const float* cov_vals, // (B, R)
    const float* xi, // (B, N - n0)
    const float* values, // (B, N)
    float* cov_vals_tangent, // (B, R)
    float* values_tangent, // (B, N)
    float* xi_tangent, // (B, N - n0)
    int n0,
    int k, 
    int n_points,
    int n_cov,
    int n_batches, // number of batches, affects cov_vals, values_tangent, and xi_tangent
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
    const float* b_cov_vals = cov_vals + b * n_cov;
    const float* b_xi = xi + b * (n_points - n0);
    const float* b_values = values + b * n_points;
    float* b_cov_vals_tangent = cov_vals_tangent + b * n_cov;
    float* b_values_tangent = values_tangent + b * n_points;
    float* b_xi_tangent = xi_tangent + b * (n_points - n0);

    // define working variables, these should fit on register
    float pts[(MAX_K + 1) * N_DIM]; // fine point + K coarse points
    float vec[MAX_K + 1];
    float vec_tangent[MAX_K + 1];
    float mat[((MAX_K + 1) * (MAX_K + 2)) / 2]; // lower triangular matrix for joint covariance
    float mat_tangent[((MAX_K + 1) * (MAX_K + 2)) / 2];

    // load neighbor points
    for (int i = 0; i < k; ++i) {
        size_t neighbor_idx = neighbors[(idx - n0) * k + i];
        vec[i] = b_values[neighbor_idx];
        for (int j = 0; j < N_DIM; ++j) {
            pts[i * N_DIM + j] = points[neighbor_idx * N_DIM + j];
        }
    }

    // load current point
    for (int j = 0; j < N_DIM; ++j) {
        pts[k * N_DIM + j] = points[idx * N_DIM + j];
    }
    vec[k] = b_xi[idx - n0];
    float fine_value_tangent = b_values_tangent[idx];

    // factorize matrix and prepare intermediate vectors
    cov_lookup_matrix(pts, cov_bins, b_cov_vals, mat, k + 1, N_DIM, n_cov); // joint covariance
    cholesky(mat, k + 1); // L
    solve_cholesky_forward(mat, vec, k, 1); // "xi_c"
    matmul(mat + tri(k, 0), &fine_value_tangent, vec_tangent, k + 1, 1, 1); // not actually the tangent of vec
    solve_cholesky_backward(mat, vec_tangent, k, 1); // Lcc^-T

    // linear transpose
    atomicAdd(b_xi_tangent + (idx - n0), vec_tangent[k]);
    for (int i = 0; i < k; ++i) {
        atomicAdd(b_values_tangent + neighbors[(idx - n0) * k + i], vec_tangent[i]);
    }

    // covariance vjp
    mat_tangent[tri(k, k)] = fine_value_tangent * vec[k]; // Lbar_ff
    for (int i = 0; i < k; ++i) {
        mat_tangent[tri(k, i)] = fine_value_tangent * vec[i]; // Lbar_fc
        for (int j = 0; j <= i; ++j) {
            mat_tangent[tri(i, j)] = -vec_tangent[i] * vec[j]; // Lbar_cc
        }
    }
    cholesky_vjp(mat, mat_tangent, k + 1);
    cov_lookup_matrix_vjp(pts, mat_tangent, cov_bins, b_cov_vals_tangent, k + 1, N_DIM, n_cov);
}

template <int MAX_K, int N_DIM>
__host__ void refine_nonlinear_vjp(
    cudaStream_t stream,
    const float* points,
    const int* neighbors,
    const int* offsets,
    const float* cov_bins,
    const float* cov_vals,
    const float* initial_values,
    const float* xi,
    const float* values,
    const float* values_tangent,
    float* values_tangent_buffer,
    float* cov_vals_tangent,
    float* initial_values_tangent,
    float* xi_tangent,
    int n0,
    int k,
    int n_points,
    int n_levels,
    int n_cov,
    int n_batches // batch dim only affects cov_vals, values_tangent, and all output buffers
) {
    // copy offsets to host
    int *offsets_host;
    offsets_host = (int*)malloc(n_levels * sizeof(int));
    if (offsets_host == nullptr) throw std::runtime_error("Failed to allocate memory for offsets on host");
    cudaMemcpy(offsets_host, offsets, n_levels * sizeof(int), cudaMemcpyDeviceToHost);

    // initialize output arrays
    cudaMemcpyAsync(values_tangent_buffer, values_tangent, n_batches * n_points * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemsetAsync(xi_tangent, 0, n_batches * (n_points - n0) * sizeof(float), stream);
    cudaMemsetAsync(cov_vals_tangent, 0, n_batches * n_cov * sizeof(float), stream);

    // walk backwards through levels and compute tangents
    for (int level = n_levels; level-- > 1;) {
        int start_idx = offsets_host[level - 1];
        int end_idx = offsets_host[level];
        int n_threads = (end_idx - start_idx) * n_batches;
        CUDA_LAUNCH(
            (refine_nonlinear_vjp_kernel<MAX_K, N_DIM>),
            n_threads, 
            stream,
            points,
            neighbors,
            cov_bins,
            cov_vals,
            xi,
            values,
            cov_vals_tangent,
            values_tangent_buffer,
            xi_tangent,
            n0,
            k,
            n_points,
            n_cov,
            n_batches,
            start_idx);
    }

    // copy initial values_tangent to output
    CUDA_LAUNCH(batch_copy, n_batches * n0, stream, initial_values_tangent, values_tangent_buffer, n_batches, n0, n_points);

    free(offsets_host);
}