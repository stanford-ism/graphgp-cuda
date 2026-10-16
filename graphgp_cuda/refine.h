// refine.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "common.h"
#include "linalg.h"
#include "covariance.h"

// index of the i-th neighbor, in pivoted order if clamping
template <bool CLAMP, typename i_t>
__forceinline__ __device__ size_t neighbor_index(const size_t* nbr, const i_t* nbr_row, size_t i) {
    if constexpr (CLAMP) {
        return nbr[i];
    } else {
        return nbr_row[i];
    }
}

// Reorder the k neighbors of a point for diagonally pivoted Cholesky (largest conditional variance first, the fine point
// stays last). Requires the neighbor points and the fine point in pts, uses mat as scratch. Writes the reordered
// neighbor indices to nbr and reloads the neighbor points into pts in that order.
template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__forceinline__ __device__ void pivot_neighbors(
    const f_t* points, // (N, d)
    const i_t* neighbors_row, // (k,)
    const f_t* cov_rates, // (M,)
    const f_t* cov_amps, // (M,)
    f_t* pts, // (k + 1, d)
    f_t* mat, // (k + 1, k + 1) lower triangular
    size_t* nbr, // (k,)
    size_t k,
    size_t n_scales
) {
    int perm[MAX_K + 1];
    cov_matrix(pts, cov_rates, cov_amps, mat, k + 1, N_DIM, n_scales);
    pivot_order(mat, perm, (int)k + 1, (int)k);
    for (size_t i = 0; i < k; ++i) {
        nbr[i] = neighbors_row[perm[i]];
        for (size_t j = 0; j < N_DIM; ++j) {
            pts[i * N_DIM + j] = points[nbr[i] * N_DIM + j];
        }
    }
}

template <size_t MAX_K, size_t N_DIM, bool CLAMP, typename i_t, typename f_t>
__global__ void refine_kernel(
    const f_t* points, // (N, d)
    const i_t* neighbors, // (N - n0, k)
    const f_t* cov_rates, // (M,)
    const f_t* cov_amps, // (B, M)
    const f_t* xi, // (B, N - n0)
    f_t* values, // (B, N)
    size_t n0,
    size_t k,
    size_t n_points,
    size_t n_scales,
    size_t n_batches, // number of batches, affects cov_amps, xi, and values
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
    const f_t *b_cov_amps = cov_amps + b * n_scales;
    const f_t *b_xi = xi + b * (n_points - n0);
    f_t *b_values = values + b * n_points;

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
    vec[k] = b_xi[idx - n0];

    // with clamping, reorder neighbors by pivoting
    const i_t* nbr_row = neighbors + (idx - n0) * k;
    size_t nbr[CLAMP ? MAX_K : 1];
    if constexpr (CLAMP) {
        pivot_neighbors<MAX_K, N_DIM>(points, nbr_row, cov_rates, b_cov_amps, pts, mat, nbr, k, n_scales);
    }
    for (size_t i = 0; i < k; ++i) {
        vec[i] = b_values[neighbor_index<CLAMP>(nbr, nbr_row, i)];
    }

    // refinement operation
    cov_matrix(pts, cov_rates, b_cov_amps, mat, k + 1, N_DIM, n_scales); // joint covariance
    cholesky<CLAMP>(mat, k + 1); // factorize
    solve_cholesky_forward<CLAMP>(mat, vec, k, 1); // "xi_c" = L_cc^-1 @ v_c
    b_values[idx] = dot(mat + tri(k, 0), vec, k + 1); // v = L @ xi
}



template <size_t MAX_K, size_t N_DIM, bool CLAMP, typename i_t, typename f_t>
__host__ void refine(
    cudaStream_t stream,
    const f_t* points,
    const i_t* neighbors,
    const i_t* offsets,
    const f_t* cov_rates,
    const f_t* cov_amps,
    const f_t* initial_values,
    const f_t* xi,
    f_t* values,
    size_t n0,
    size_t k,
    size_t n_points,
    size_t n_levels,
    size_t n_scales,
    size_t n_batches // batch dim only affects cov_amps, initial_values, xi, and output values
) {
    // copy initial values to output values
    batch_copy<<<cld(n_batches * n0, 256), 256, 0, stream>>>(initial_values, values, n_batches, n0, n_points, n0);

    // iteratively refine levels
    i_t start_idx;
    i_t end_idx;
    for (int level = 1; level < n_levels; ++level) {
        cudaMemcpyAsync(&start_idx, offsets + level - 1, sizeof(i_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&end_idx, offsets + level, sizeof(i_t), cudaMemcpyDeviceToHost, stream);
        size_t n_threads = (end_idx - start_idx) * n_batches;
        refine_kernel<MAX_K, N_DIM, CLAMP, i_t, f_t><<<cld(n_threads, 256), 256, 0, stream>>>(
            points,
            neighbors,
            cov_rates,
            cov_amps,
            xi,
            values,
            n0,
            k,
            n_points,
            n_scales,
            n_batches,
            start_idx,
            n_threads);
    }
}

template <size_t MAX_K, size_t N_DIM, bool CLAMP, typename i_t, typename f_t>  
__global__ void refine_transpose_kernel(
    const f_t* points, // (N, d)
    const i_t* neighbors, // (N - n0, k)
    const f_t* cov_rates, // (M,)
    const f_t* cov_amps, // (B, M)
    f_t* values, // (B, N)
    f_t* xi, // (B, N - n0)
    size_t n0,
    size_t k, 
    size_t n_points,
    size_t n_scales,
    size_t n_batches, // number of batches, affects cov_amps, values, and xi
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
    const f_t *b_cov_amps = cov_amps + b * n_scales;
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

    // with clamping, reorder neighbors by pivoting
    const i_t* nbr_row = neighbors + (idx - n0) * k;
    size_t nbr[CLAMP ? MAX_K : 1];
    if constexpr (CLAMP) {
        pivot_neighbors<MAX_K, N_DIM>(points, nbr_row, cov_rates, b_cov_amps, pts, mat, nbr, k, n_scales);
    }

    // refinement operation transpose
    cov_matrix(pts, cov_rates, b_cov_amps, mat, k + 1, N_DIM, n_scales); // joint covariance
    cholesky<CLAMP>(mat, k + 1); // L
    matmul(mat + tri(k, 0), &fine_value, vec, k + 1, 1, 1);
    atomicAdd(b_xi + (idx - n0), vec[k]);
    solve_cholesky_backward<CLAMP>(mat, vec, k, 1); // (Lcc^T)^-1
    for (size_t i = 0; i < k; ++i) {
        atomicAdd(b_values + neighbor_index<CLAMP>(nbr, nbr_row, i), vec[i]);
    }
}

// refine is linear in initial_values and xi, so we transpose with respect to those
template <size_t MAX_K, size_t N_DIM, bool CLAMP, typename i_t, typename f_t>
__host__ void refine_transpose(
    cudaStream_t stream,
    const f_t* points,
    const i_t* neighbors,
    const i_t* offsets,
    const f_t* cov_rates,
    const f_t* cov_amps,
    const f_t* values,
    f_t* values_buffer,
    f_t* initial_values,
    f_t* xi,
    size_t n0,
    size_t k,
    size_t n_points,
    size_t n_levels,
    size_t n_scales,
    size_t n_batches // batch dim only affects cov_amps, values, and all output buffers
) {
    // initialize output arrays
    cudaMemcpyAsync(values_buffer, values, n_batches * n_points * sizeof(f_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemsetAsync(xi, 0, n_batches * (n_points - n0) * sizeof(f_t), stream);

    // walk backwards through levels and compute tangents
    i_t start_idx;
    i_t end_idx;
    for (size_t level = n_levels; level-- > 1;) {
        cudaMemcpyAsync(&start_idx, offsets + level - 1, sizeof(i_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&end_idx, offsets + level, sizeof(i_t), cudaMemcpyDeviceToHost, stream);
        size_t n_threads = (end_idx - start_idx) * n_batches;
        refine_transpose_kernel<MAX_K, N_DIM, CLAMP, i_t, f_t><<<cld(n_threads, 256), 256, 0, stream>>>(
            points,
            neighbors,
            cov_rates,
            cov_amps,
            values_buffer,
            xi,
            n0,
            k,
            n_points,
            n_scales,
            n_batches,
            start_idx,
            n_threads);
    }

    // copy initial values to output
    batch_copy<<<cld(n_batches * n0, 256), 256, 0, stream>>>(values_buffer, initial_values, n_batches, n_points, n0, n0);
}

template <size_t MAX_K, size_t N_DIM, bool CLAMP, typename i_t, typename f_t>
__global__ void refine_jvp_kernel(
    const f_t* points, // (N, d)
    const i_t* neighbors, // (N - n0, k)
    const f_t* cov_rates, // (M,)
    const f_t* cov_amps, // (B, M)
    const f_t* xi, // (B, N - n0)
    const f_t* values, // (B, N)
    const f_t* cov_amps_tangent, // (B, M)
    const f_t* xi_tangent, // (B, N - n0)
    f_t* values_tangent, // (B, N)
    size_t n0,
    size_t k,
    size_t n_points,
    size_t n_scales,
    size_t n_batches, // number of batches, affects cov_amps, xi, and values
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
    const f_t *b_cov_amps = cov_amps + b * n_scales;
    const f_t *b_xi = xi + b * (n_points - n0);
    const f_t *b_cov_amps_tangent = cov_amps_tangent + b * n_scales;
    const f_t *b_xi_tangent = xi_tangent + b * (n_points - n0);
    const f_t *b_values = values + b * n_points;
    f_t *b_values_tangent = values_tangent + b * n_points;

    // define working variables, these should fit on register
    f_t pts[(MAX_K + 1) * N_DIM]; // fine point + K coarse points
    f_t vec[MAX_K + 1];
    f_t vec_tangent[MAX_K + 1];
    f_t mat[((MAX_K + 1) * (MAX_K + 2)) / 2]; // lower triangular matrix for joint covariance
    f_t mat_tangent[((MAX_K + 1) * (MAX_K + 2)) / 2];

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
    vec[k] = b_xi[idx - n0];
    vec_tangent[k] = b_xi_tangent[idx - n0];
    
    // with clamping, reorder neighbors by pivoting
    const i_t* nbr_row = neighbors + (idx - n0) * k;
    size_t nbr[CLAMP ? MAX_K : 1];
    if constexpr (CLAMP) {
        pivot_neighbors<MAX_K, N_DIM>(points, nbr_row, cov_rates, b_cov_amps, pts, mat, nbr, k, n_scales);
    }
    for (size_t i = 0; i < k; ++i) {
        vec[i] = b_values[neighbor_index<CLAMP>(nbr, nbr_row, i)];
        vec_tangent[i] = b_values_tangent[neighbor_index<CLAMP>(nbr, nbr_row, i)];
    }

    // load and factorize joint covariance, with jvp
    cov_matrix(pts, cov_rates, b_cov_amps, mat, k + 1, N_DIM, n_scales); // K
    cov_matrix(pts, cov_rates, b_cov_amps_tangent, mat_tangent, k + 1, N_DIM, n_scales); // dK
    cholesky<CLAMP>(mat, k + 1); // L
    cholesky_jvp<CLAMP>(mat, mat_tangent, k + 1); // dL

    // compute values and linear jvp
    solve_cholesky_forward<CLAMP>(mat, vec, k, 1);
    solve_cholesky_forward<CLAMP>(mat, vec_tangent, k, 1);
    // b_values[idx] = dot(mat + tri(k, 0), vec, k + 1); // don't write values, we assume they are given
    b_values_tangent[idx] = dot(mat + tri(k, 0), vec_tangent, k + 1);

    // nonlinear from dot product
    b_values_tangent[idx] += dot(mat_tangent + tri(k, 0), vec, k + 1);

    // nonlinear from inverse
    apply_cholesky(mat_tangent, vec, k, 1);
    solve_cholesky_forward<CLAMP>(mat, vec, k, 1);
    b_values_tangent[idx] -= dot(mat + tri(k, 0), vec, k);
}



template <size_t MAX_K, size_t N_DIM, bool CLAMP, typename i_t, typename f_t>
__host__ void refine_jvp(
    cudaStream_t stream,
    const f_t* points,
    const i_t* neighbors,
    const i_t* offsets,
    const f_t* cov_rates,
    const f_t* cov_amps,
    const f_t* xi,
    const f_t* values,
    const f_t* cov_amps_tangent,
    const f_t* initial_values_tangent,
    const f_t* xi_tangent,
    f_t* values_tangent,
    size_t n0,
    size_t k,
    size_t n_points,
    size_t n_levels,
    size_t n_scales,
    size_t n_batches // batch dim only affects cov_amps, initial_values, xi, and output values
) {
    // copy initial values to output values
    // batch_copy<<<cld(n_batches * n0, 256), 256, 0, stream>>>(initial_values, values, n_batches, n0, n_points, n0); // don't copy any values, we assume they are given
    batch_copy<<<cld(n_batches * n0, 256), 256, 0, stream>>>(initial_values_tangent, values_tangent, n_batches, n0, n_points, n0);

    // iteratively refine levels
    i_t start_idx;
    i_t end_idx;
    for (size_t level = 1; level < n_levels; ++level) {
        cudaMemcpyAsync(&start_idx, offsets + level - 1, sizeof(i_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&end_idx, offsets + level, sizeof(i_t), cudaMemcpyDeviceToHost, stream);
        size_t n_threads = (end_idx - start_idx) * n_batches;
        refine_jvp_kernel<MAX_K, N_DIM, CLAMP, i_t, f_t><<<cld(n_threads, 256), 256, 0, stream>>>(
            points,
            neighbors,
            cov_rates,
            cov_amps,
            xi,
            values,
            cov_amps_tangent,
            xi_tangent,
            values_tangent,
            n0,
            k,
            n_points,
            n_scales,
            n_batches,
            start_idx,
            n_threads);
    }
}

template <size_t MAX_K, size_t N_DIM, bool CLAMP, typename i_t, typename f_t>  
__global__ void refine_vjp_kernel(
    const f_t* points, // (N, d)
    const i_t* neighbors, // (N - n0, k)
    const f_t* cov_rates, // (M,)
    const f_t* cov_amps, // (B, M)
    const f_t* xi, // (B, N - n0)
    const f_t* values, // (B, N)
    f_t* cov_amps_tangent, // (B, M)
    f_t* values_tangent, // (B, N)
    f_t* xi_tangent, // (B, N - n0)
    size_t n0,
    size_t k, 
    size_t n_points,
    size_t n_scales,
    size_t n_batches, // number of batches, affects cov_amps, values_tangent, and xi_tangent
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
    const f_t* b_cov_amps = cov_amps + b * n_scales;
    const f_t* b_xi = xi + b * (n_points - n0);
    const f_t* b_values = values + b * n_points;
    f_t* b_cov_amps_tangent = cov_amps_tangent + b * n_scales;
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

    // with clamping, reorder neighbors by pivoting
    const i_t* nbr_row = neighbors + (idx - n0) * k;
    size_t nbr[CLAMP ? MAX_K : 1];
    if constexpr (CLAMP) {
        pivot_neighbors<MAX_K, N_DIM>(points, nbr_row, cov_rates, b_cov_amps, pts, mat, nbr, k, n_scales);
    }
    for (size_t i = 0; i < k; ++i) {
        vec[i] = b_values[neighbor_index<CLAMP>(nbr, nbr_row, i)];
    }

    // factorize matrix and prepare intermediate vectors
    cov_matrix(pts, cov_rates, b_cov_amps, mat, k + 1, N_DIM, n_scales); // joint covariance
    cholesky<CLAMP>(mat, k + 1); // L
    solve_cholesky_forward<CLAMP>(mat, vec, k, 1); // "xi_c"
    matmul(mat + tri(k, 0), &fine_value_tangent, vec_tangent, k + 1, 1, 1); // not actually the tangent of vec
    solve_cholesky_backward<CLAMP>(mat, vec_tangent, k, 1); // Lcc^-T

    // linear transpose
    atomicAdd(b_xi_tangent + (idx - n0), vec_tangent[k]);
    for (size_t i = 0; i < k; ++i) {
        atomicAdd(b_values_tangent + neighbor_index<CLAMP>(nbr, nbr_row, i), vec_tangent[i]);
    }

    // covariance vjp
    mat_tangent[tri(k, k)] = fine_value_tangent * vec[k]; // Lbar_ff
    for (size_t i = 0; i < k; ++i) {
        mat_tangent[tri(k, i)] = fine_value_tangent * vec[i]; // Lbar_fc
        for (size_t j = 0; j <= i; ++j) {
            mat_tangent[tri(i, j)] = -vec_tangent[i] * vec[j]; // Lbar_cc
        }
    }
    cholesky_vjp<CLAMP>(mat, mat_tangent, k + 1);
    cov_matrix_vjp(pts, mat_tangent, cov_rates, b_cov_amps_tangent, mat, k + 1, N_DIM, n_scales);
}

template <size_t MAX_K, size_t N_DIM, bool CLAMP, typename i_t, typename f_t>
__host__ void refine_vjp(
    cudaStream_t stream,
    const f_t* points,
    const i_t* neighbors,
    const i_t* offsets,
    const f_t* cov_rates,
    const f_t* cov_amps,
    const f_t* xi,
    const f_t* values,
    const f_t* values_tangent,
    f_t* values_tangent_buffer,
    f_t* cov_amps_tangent,
    f_t* initial_values_tangent,
    f_t* xi_tangent,
    size_t n0,
    size_t k,
    size_t n_points,
    size_t n_levels,
    size_t n_scales,
    size_t n_batches // batch dim only affects cov_amps, values_tangent, and all output buffers
) {
    // initialize output arrays
    cudaMemcpyAsync(values_tangent_buffer, values_tangent, n_batches * n_points * sizeof(f_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemsetAsync(xi_tangent, 0, n_batches * (n_points - n0) * sizeof(f_t), stream);
    cudaMemsetAsync(cov_amps_tangent, 0, n_batches * n_scales * sizeof(f_t), stream);

    // walk backwards through levels and compute tangents
    i_t start_idx;
    i_t end_idx;
    for (size_t level = n_levels; level-- > 1;) {
        cudaMemcpyAsync(&start_idx, offsets + level - 1, sizeof(i_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&end_idx, offsets + level, sizeof(i_t), cudaMemcpyDeviceToHost, stream);
        size_t n_threads = (end_idx - start_idx) * n_batches;
        refine_vjp_kernel<MAX_K, N_DIM, CLAMP, i_t, f_t><<<cld(n_threads, 256), 256, 0, stream>>>(
            points,
            neighbors,
            cov_rates,
            cov_amps,
            xi,
            values,
            cov_amps_tangent,
            values_tangent_buffer,
            xi_tangent,
            n0,
            k,
            n_points,
            n_scales,
            n_batches,
            start_idx,
            n_threads);
    }

    // copy initial values_tangent to output
    batch_copy<<<cld(n_batches * n0, 256), 256, 0, stream>>>(values_tangent_buffer, initial_values_tangent, n_batches, n_points, n0, n0);
}