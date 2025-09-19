// refine.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "common.h"
#include "linalg.h"
#include "covariance.h"

template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__global__ void refine_kernel(
    const f_t* points, // (N, d)
    const i_t* neighbors, // (N - n0, k)
    const f_t* cov_bins, // (R,)
    const f_t* cov_vals, // (B, R)
    const f_t* xi, // (B, N - n0)
    f_t* values, // (B, N)
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
    const f_t *b_xi = xi + b * (n_points - n0);
    f_t *b_values = values + b * n_points;

    // define working variables, these should fit on register
    f_t pts[(MAX_K + 1) * N_DIM]; // fine point + K coarse points
    f_t vec[MAX_K + 1];
    f_t mat[((MAX_K + 1) * (MAX_K + 2)) / 2]; // lower triangular matrix for joint covariance

    // load neighbor points and values
    for (size_t i = 0; i < k; ++i) {
        size_t neighbor_idx = neighbors[(idx - n0) * k + i];
        vec[i] = b_values[neighbor_idx]; // coarse points use values
        for (size_t j = 0; j < N_DIM; ++j) {
            pts[i * N_DIM + j] = points[neighbor_idx * N_DIM + j];
        }
    }

    // load current point
    for (size_t j = 0; j < N_DIM; ++j) {
        pts[k * N_DIM + j] = points[idx * N_DIM + j];
    }
    vec[k] = b_xi[idx - n0];

    // refinement operation
    cov_lookup_matrix(pts, cov_bins, b_cov_vals, mat, k + 1, N_DIM, n_cov); // joint covariance
    cholesky(mat, k + 1); // factorize
    solve_cholesky_forward(mat, vec, k, 1); // "xi_c" = L_cc^-1 @ v_c
    b_values[idx] = dot(mat + tri(k, 0), vec, k + 1); // v = L @ xi
}



template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__host__ void refine(
    cudaStream_t stream,
    const f_t* points,
    const i_t* neighbors,
    const i_t* offsets,
    const f_t* cov_bins,
    const f_t* cov_vals,
    const f_t* initial_values,
    const f_t* xi,
    f_t* values,
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

    // copy initial values to output values
    batch_copy<<<cld(n_batches * n0, 256), 256, 0, stream>>>(initial_values, values, n_batches, n0, n_points, n0);

    // iteratively refine levels
    for (int level = 1; level < n_levels; ++level) {
        size_t start_idx = offsets_host[level - 1];
        size_t end_idx = offsets_host[level];
        size_t n_threads = (end_idx - start_idx) * n_batches;
        refine_kernel<MAX_K, N_DIM, i_t, f_t><<<cld(n_threads, 256), 256, 0, stream>>>(
            points,
            neighbors,
            cov_bins,
            cov_vals,
            xi,
            values,
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

template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>  
__global__ void refine_transpose_kernel(
    const f_t* points, // (N, d)
    const i_t* neighbors, // (N - n0, k)
    const f_t* cov_bins, // (R,)
    const f_t* cov_vals, // (B, R)
    f_t* values, // (B, N)
    f_t* xi, // (B, N - n0)
    size_t n0,
    size_t k, 
    size_t n_points,
    size_t n_cov,
    size_t n_batches, // number of batches, affects cov_vals, values, and xi
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
    f_t *b_values = values + b * n_points;
    f_t *b_xi = xi + b * (n_points - n0);

    // define working variables, these should fit on register
    f_t pts[(MAX_K + 1) * N_DIM]; // fine point + K coarse points
    f_t vec[MAX_K + 1];
    f_t mat[((MAX_K + 1) * (MAX_K + 2)) / 2]; // lower triangular matrix for joint covariance

    // load neighbor points
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
    f_t fine_value = b_values[idx];

    // refinement operation transpose
    cov_lookup_matrix(pts, cov_bins, b_cov_vals, mat, k + 1, N_DIM, n_cov); // joint covariance
    cholesky(mat, k + 1); // L
    matmul(mat + tri(k, 0), &fine_value, vec, k + 1, 1, 1);
    atomicAdd(b_xi + (idx - n0), vec[k]);
    solve_cholesky_backward(mat, vec, k, 1); // (Lcc^T)^-1
    for (size_t i = 0; i < k; ++i) {
        atomicAdd(b_values + neighbors[(idx - n0) * k + i], vec[i]);
    }
}

// refine is linear in initial_values and xi, so we transpose with respect to those
template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__host__ void refine_transpose(
    cudaStream_t stream,
    const f_t* points,
    const i_t* neighbors,
    const i_t* offsets,
    const f_t* cov_bins,
    const f_t* cov_vals,
    const f_t* values,
    f_t* values_buffer,
    f_t* initial_values,
    f_t* xi,
    size_t n0,
    size_t k,
    size_t n_points,
    size_t n_levels,
    size_t n_cov,
    size_t n_batches // batch dim only affects cov_vals, values, and all output buffers
) {
    // copy offsets to host
    i_t *offsets_host;
    offsets_host = (i_t*)malloc(n_levels * sizeof(i_t));
    if (offsets_host == nullptr) throw std::runtime_error("Failed to allocate memory for offsets on host");
    cudaMemcpy(offsets_host, offsets, n_levels * sizeof(i_t), cudaMemcpyDeviceToHost);

    // initialize output arrays
    cudaMemcpyAsync(values_buffer, values, n_batches * n_points * sizeof(f_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemsetAsync(xi, 0, n_batches * (n_points - n0) * sizeof(f_t), stream);

    // walk backwards through levels and compute tangents
    for (size_t level = n_levels; level-- > 1;) {
        size_t start_idx = offsets_host[level - 1];
        size_t end_idx = offsets_host[level];
        size_t n_threads = (end_idx - start_idx) * n_batches;
        refine_transpose_kernel<MAX_K, N_DIM, i_t, f_t><<<cld(n_threads, 256), 256, 0, stream>>>(
            points,
            neighbors,
            cov_bins,
            cov_vals,
            values_buffer,
            xi,
            n0,
            k,
            n_points,
            n_cov,
            n_batches,
            start_idx,
            n_threads);
    }

    // copy initial values to output
    batch_copy<<<cld(n_batches * n0, 256), 256, 0, stream>>>(values_buffer, initial_values, n_batches, n_points, n0, n0);

    free(offsets_host);
}

template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__global__ void refine_jvp_kernel(
    const f_t* points, // (N, d)
    const i_t* neighbors, // (N - n0, k)
    const f_t* cov_bins, // (R,)
    const f_t* cov_vals, // (B, R)
    const f_t* xi, // (B, R)
    const f_t* cov_vals_tangent, // (B, R)
    const f_t* xi_tangent, // (B, R)
    f_t* values, // (B, R)
    f_t* values_tangent, // (B, R)
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
    const f_t *b_xi = xi + b * (n_points - n0);
    const f_t *b_cov_vals_tangent = cov_vals_tangent + b * n_cov;
    const f_t *b_xi_tangent = xi_tangent + b * (n_points - n0);
    f_t *b_values = values + b * n_points;
    f_t *b_values_tangent = values_tangent + b * n_points;

    // define working variables, these should fit on register
    f_t pts[(MAX_K + 1) * N_DIM]; // fine point + K coarse points
    f_t vec[MAX_K + 1];
    f_t vec_tangent[MAX_K + 1];
    f_t mat[((MAX_K + 1) * (MAX_K + 2)) / 2]; // lower triangular matrix for joint covariance
    f_t mat_tangent[((MAX_K + 1) * (MAX_K + 2)) / 2];

    // load neighbor points and values
    for (size_t i = 0; i < k; ++i) {
        size_t neighbor_idx = neighbors[(idx - n0) * k + i];
        vec[i] = b_values[neighbor_idx]; // coarse points use values
        vec_tangent[i] = b_values_tangent[neighbor_idx];
        for (size_t j = 0; j < N_DIM; ++j) {
            pts[i * N_DIM + j] = points[neighbor_idx * N_DIM + j];
        }
    }

    // load current point
    for (size_t j = 0; j < N_DIM; ++j) {
        pts[k * N_DIM + j] = points[idx * N_DIM + j];
    }
    vec[k] = b_xi[idx - n0];
    vec_tangent[k] = b_xi_tangent[idx - n0];
    
    // load and factorize joint covariance, with jvp
    cov_lookup_matrix(pts, cov_bins, b_cov_vals, mat, k + 1, N_DIM, n_cov); // K
    cov_lookup_matrix(pts, cov_bins, b_cov_vals_tangent, mat_tangent, k + 1, N_DIM, n_cov); // dK
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



template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__host__ void refine_jvp(
    cudaStream_t stream,
    const f_t* points,
    const i_t* neighbors,
    const i_t* offsets,
    const f_t* cov_bins,
    const f_t* cov_vals,
    const f_t* initial_values,
    const f_t* xi,
    const f_t* cov_vals_tangent,
    const f_t* initial_values_tangent,
    const f_t* xi_tangent,
    f_t* values,
    f_t* values_tangent,
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

    // copy initial values to output values
    batch_copy<<<cld(n_batches * n0, 256), 256, 0, stream>>>(initial_values, values, n_batches, n0, n_points, n0);
    batch_copy<<<cld(n_batches * n0, 256), 256, 0, stream>>>(initial_values_tangent, values_tangent, n_batches, n0, n_points, n0);

    // iteratively refine levels
    for (size_t level = 1; level < n_levels; ++level) {
        size_t start_idx = offsets_host[level - 1];
        size_t end_idx = offsets_host[level];
        size_t n_threads = (end_idx - start_idx) * n_batches;
        refine_jvp_kernel<MAX_K, N_DIM, i_t, f_t><<<cld(n_threads, 256), 256, 0, stream>>>(
            points,
            neighbors,
            cov_bins,
            cov_vals,
            xi,
            cov_vals_tangent,
            xi_tangent,
            values,
            values_tangent,
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

template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>  
__global__ void refine_vjp_kernel(
    const f_t* points, // (N, d)
    const i_t* neighbors, // (N - n0, k)
    const f_t* cov_bins, // (R,)
    const f_t* cov_vals, // (B, R)
    const f_t* xi, // (B, N - n0)
    const f_t* values, // (B, N)
    f_t* cov_vals_tangent, // (B, R)
    f_t* values_tangent, // (B, N)
    f_t* xi_tangent, // (B, N - n0)
    size_t n0,
    size_t k, 
    size_t n_points,
    size_t n_cov,
    size_t n_batches, // number of batches, affects cov_vals, values_tangent, and xi_tangent
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
    const f_t* b_cov_vals = cov_vals + b * n_cov;
    const f_t* b_xi = xi + b * (n_points - n0);
    const f_t* b_values = values + b * n_points;
    f_t* b_cov_vals_tangent = cov_vals_tangent + b * n_cov;
    f_t* b_values_tangent = values_tangent + b * n_points;
    f_t* b_xi_tangent = xi_tangent + b * (n_points - n0);

    // define working variables, these should fit on register
    f_t pts[(MAX_K + 1) * N_DIM]; // fine point + K coarse points
    f_t vec[MAX_K + 1];
    f_t vec_tangent[MAX_K + 1];
    f_t mat[((MAX_K + 1) * (MAX_K + 2)) / 2]; // lower triangular matrix for joint covariance
    f_t mat_tangent[((MAX_K + 1) * (MAX_K + 2)) / 2];

    // load neighbor points
    for (size_t i = 0; i < k; ++i) {
        size_t neighbor_idx = neighbors[(idx - n0) * k + i];
        vec[i] = b_values[neighbor_idx];
        for (size_t j = 0; j < N_DIM; ++j) {
            pts[i * N_DIM + j] = points[neighbor_idx * N_DIM + j];
        }
    }

    // load current point
    for (size_t j = 0; j < N_DIM; ++j) {
        pts[k * N_DIM + j] = points[idx * N_DIM + j];
    }
    vec[k] = b_xi[idx - n0];
    f_t fine_value_tangent = b_values_tangent[idx];

    // factorize matrix and prepare intermediate vectors
    cov_lookup_matrix(pts, cov_bins, b_cov_vals, mat, k + 1, N_DIM, n_cov); // joint covariance
    cholesky(mat, k + 1); // L
    solve_cholesky_forward(mat, vec, k, 1); // "xi_c"
    matmul(mat + tri(k, 0), &fine_value_tangent, vec_tangent, k + 1, 1, 1); // not actually the tangent of vec
    solve_cholesky_backward(mat, vec_tangent, k, 1); // Lcc^-T

    // linear transpose
    atomicAdd(b_xi_tangent + (idx - n0), vec_tangent[k]);
    for (size_t i = 0; i < k; ++i) {
        atomicAdd(b_values_tangent + neighbors[(idx - n0) * k + i], vec_tangent[i]);
    }

    // covariance vjp
    mat_tangent[tri(k, k)] = fine_value_tangent * vec[k]; // Lbar_ff
    for (size_t i = 0; i < k; ++i) {
        mat_tangent[tri(k, i)] = fine_value_tangent * vec[i]; // Lbar_fc
        for (size_t j = 0; j <= i; ++j) {
            mat_tangent[tri(i, j)] = -vec_tangent[i] * vec[j]; // Lbar_cc
        }
    }
    cholesky_vjp(mat, mat_tangent, k + 1);
    cov_lookup_matrix_vjp(pts, mat_tangent, cov_bins, b_cov_vals_tangent, k + 1, N_DIM, n_cov);
}

template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__host__ void refine_vjp(
    cudaStream_t stream,
    const f_t* points,
    const i_t* neighbors,
    const i_t* offsets,
    const f_t* cov_bins,
    const f_t* cov_vals,
    const f_t* initial_values,
    const f_t* xi,
    const f_t* values,
    const f_t* values_tangent,
    f_t* values_tangent_buffer,
    f_t* cov_vals_tangent,
    f_t* initial_values_tangent,
    f_t* xi_tangent,
    size_t n0,
    size_t k,
    size_t n_points,
    size_t n_levels,
    size_t n_cov,
    size_t n_batches // batch dim only affects cov_vals, values_tangent, and all output buffers
) {
    // copy offsets to host
    i_t *offsets_host;
    offsets_host = (i_t*)malloc(n_levels * sizeof(i_t));
    if (offsets_host == nullptr) throw std::runtime_error("Failed to allocate memory for offsets on host");
    cudaMemcpy(offsets_host, offsets, n_levels * sizeof(i_t), cudaMemcpyDeviceToHost);

    // initialize output arrays
    cudaMemcpyAsync(values_tangent_buffer, values_tangent, n_batches * n_points * sizeof(f_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemsetAsync(xi_tangent, 0, n_batches * (n_points - n0) * sizeof(f_t), stream);
    cudaMemsetAsync(cov_vals_tangent, 0, n_batches * n_cov * sizeof(f_t), stream);

    // walk backwards through levels and compute tangents
    for (size_t level = n_levels; level-- > 1;) {
        size_t start_idx = offsets_host[level - 1];
        size_t end_idx = offsets_host[level];
        size_t n_threads = (end_idx - start_idx) * n_batches;
        refine_vjp_kernel<MAX_K, N_DIM, i_t, f_t><<<cld(n_threads, 256), 256, 0, stream>>>(
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
            start_idx,
            n_threads);
    }

    // copy initial values_tangent to output
    batch_copy<<<cld(n_batches * n0, 256), 256, 0, stream>>>(values_tangent_buffer, initial_values_tangent, n_batches, n_points, n0, n0);

    free(offsets_host);
}