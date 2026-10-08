// refine_level.h
//
// Level-wise refinement in an arbitrary (user) point order for distributed generation.
//
// The points of one device ("local", M of them) carry a level id each; their neighbors
// index a global point set (N points, replicated on every device) whose current values
// are gathered before every level. One launch processes one level: every local point
// is visited, points of other levels copy their current value, points of the level are
// refined exactly as in refine.h. The kernels are linear in (values_global, xi,
// values_local) for fixed covariance; the jvp/vjp kernels linearize the covariance
// dependence as refine_jvp/refine_vjp do.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "common.h"
#include "linalg.h"
#include "covariance.h"

// ---------------------------------------------------------------- forward
template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__global__ void refine_level_kernel(
    const f_t* points_global, // (N, d)
    const f_t* points_local, // (M, d)
    const i_t* neighbors, // (M, k) global indices
    const i_t* levels, // (M,)
    i_t level,
    const f_t* cov_bins, // (R,)
    const f_t* cov_vals, // (B, R)
    const f_t* values_global, // (B, N)
    const f_t* xi, // (B, M)
    const f_t* values_local, // (B, M)
    f_t* out, // (B, M)
    size_t k,
    size_t n_global,
    size_t n_local,
    size_t n_cov,
    size_t n_threads // = n_local * n_batches
) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;
    size_t b = tid / n_local;
    size_t idx = tid % n_local;

    const f_t* b_cov_vals = cov_vals + b * n_cov;
    const f_t* b_values_global = values_global + b * n_global;
    const f_t* b_xi = xi + b * n_local;
    const f_t* b_values_local = values_local + b * n_local;
    f_t* b_out = out + b * n_local;

    if (levels[idx] != level) {
        b_out[idx] = b_values_local[idx];
        return;
    }

    f_t pts[(MAX_K + 1) * N_DIM];
    f_t vec[MAX_K + 1];
    f_t mat[((MAX_K + 1) * (MAX_K + 2)) / 2];

    for (size_t i = 0; i < k; ++i) {
        size_t neighbor_idx = neighbors[idx * k + i];
        vec[i] = b_values_global[neighbor_idx];
        for (size_t j = 0; j < N_DIM; ++j) {
            pts[i * N_DIM + j] = points_global[neighbor_idx * N_DIM + j];
        }
    }
    for (size_t j = 0; j < N_DIM; ++j) {
        pts[k * N_DIM + j] = points_local[idx * N_DIM + j];
    }
    vec[k] = b_xi[idx];

    cov_lookup_matrix(pts, cov_bins, b_cov_vals, mat, k + 1, N_DIM, n_cov);
    cholesky(mat, k + 1);
    solve_cholesky_forward(mat, vec, k, 1);
    b_out[idx] = dot(mat + tri(k, 0), vec, k + 1);
}

template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__host__ void refine_level(
    cudaStream_t stream,
    const f_t* points_global,
    const f_t* points_local,
    const i_t* neighbors,
    const i_t* levels,
    const i_t* level_dev, // (1,) on device
    const f_t* cov_bins,
    const f_t* cov_vals,
    const f_t* values_global,
    const f_t* xi,
    const f_t* values_local,
    f_t* out,
    size_t k,
    size_t n_global,
    size_t n_local,
    size_t n_cov,
    size_t n_batches
) {
    i_t level;
    cudaMemcpyAsync(&level, level_dev, sizeof(i_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    size_t n_threads = n_local * n_batches;
    refine_level_kernel<MAX_K, N_DIM, i_t, f_t><<<cld(n_threads, 256), 256, 0, stream>>>(
        points_global, points_local, neighbors, levels, level, cov_bins, cov_vals,
        values_global, xi, values_local, out, k, n_global, n_local, n_cov, n_threads);
}

// ---------------------------------------------------------------- linear transpose
// Transpose w.r.t. (values_global, xi, values_local) given the covariance.
template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__global__ void refine_level_transpose_kernel(
    const f_t* points_global,
    const f_t* points_local,
    const i_t* neighbors,
    const i_t* levels,
    i_t level,
    const f_t* cov_bins,
    const f_t* cov_vals,
    const f_t* out_tangent, // (B, M) cotangent of the output
    f_t* values_global_tangent, // (B, N), zero-initialized, atomics
    f_t* xi_tangent, // (B, M)
    f_t* values_local_tangent, // (B, M)
    size_t k,
    size_t n_global,
    size_t n_local,
    size_t n_cov,
    size_t n_threads
) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;
    size_t b = tid / n_local;
    size_t idx = tid % n_local;

    const f_t* b_cov_vals = cov_vals + b * n_cov;
    const f_t* b_out_tangent = out_tangent + b * n_local;
    f_t* b_values_global_tangent = values_global_tangent + b * n_global;
    f_t* b_xi_tangent = xi_tangent + b * n_local;
    f_t* b_values_local_tangent = values_local_tangent + b * n_local;

    if (levels[idx] != level) {
        b_values_local_tangent[idx] = b_out_tangent[idx];
        b_xi_tangent[idx] = f_t(0);
        return;
    }
    b_values_local_tangent[idx] = f_t(0);

    f_t pts[(MAX_K + 1) * N_DIM];
    f_t vec[MAX_K + 1];
    f_t mat[((MAX_K + 1) * (MAX_K + 2)) / 2];

    for (size_t i = 0; i < k; ++i) {
        size_t neighbor_idx = neighbors[idx * k + i];
        for (size_t j = 0; j < N_DIM; ++j) {
            pts[i * N_DIM + j] = points_global[neighbor_idx * N_DIM + j];
        }
    }
    for (size_t j = 0; j < N_DIM; ++j) {
        pts[k * N_DIM + j] = points_local[idx * N_DIM + j];
    }
    f_t fine_tangent = b_out_tangent[idx];

    cov_lookup_matrix(pts, cov_bins, b_cov_vals, mat, k + 1, N_DIM, n_cov);
    cholesky(mat, k + 1);
    matmul(mat + tri(k, 0), &fine_tangent, vec, k + 1, 1, 1); // L[k, :]^T * ct
    b_xi_tangent[idx] = vec[k];
    solve_cholesky_backward(mat, vec, k, 1); // Lcc^-T
    for (size_t i = 0; i < k; ++i) {
        atomicAdd(b_values_global_tangent + neighbors[idx * k + i], vec[i]);
    }
}

template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__host__ void refine_level_transpose(
    cudaStream_t stream,
    const f_t* points_global,
    const f_t* points_local,
    const i_t* neighbors,
    const i_t* levels,
    const i_t* level_dev,
    const f_t* cov_bins,
    const f_t* cov_vals,
    const f_t* out_tangent,
    f_t* values_global_tangent,
    f_t* xi_tangent,
    f_t* values_local_tangent,
    size_t k,
    size_t n_global,
    size_t n_local,
    size_t n_cov,
    size_t n_batches
) {
    i_t level;
    cudaMemcpyAsync(&level, level_dev, sizeof(i_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaMemsetAsync(values_global_tangent, 0, n_batches * n_global * sizeof(f_t), stream);
    size_t n_threads = n_local * n_batches;
    refine_level_transpose_kernel<MAX_K, N_DIM, i_t, f_t><<<cld(n_threads, 256), 256, 0, stream>>>(
        points_global, points_local, neighbors, levels, level, cov_bins, cov_vals, out_tangent,
        values_global_tangent, xi_tangent, values_local_tangent, k, n_global, n_local, n_cov, n_threads);
}

// ---------------------------------------------------------------- jvp (covariance and linear tangents)
template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__global__ void refine_level_jvp_kernel(
    const f_t* points_global,
    const f_t* points_local,
    const i_t* neighbors,
    const i_t* levels,
    i_t level,
    const f_t* cov_bins,
    const f_t* cov_vals, // (B, R)
    const f_t* values_global, // (B, N)
    const f_t* xi, // (B, M)
    const f_t* cov_vals_tangent, // (B, R)
    const f_t* values_global_tangent, // (B, N)
    const f_t* xi_tangent, // (B, M)
    const f_t* values_local_tangent, // (B, M)
    f_t* out_tangent, // (B, M)
    size_t k,
    size_t n_global,
    size_t n_local,
    size_t n_cov,
    size_t n_threads
) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;
    size_t b = tid / n_local;
    size_t idx = tid % n_local;

    const f_t* b_cov_vals = cov_vals + b * n_cov;
    const f_t* b_values_global = values_global + b * n_global;
    const f_t* b_xi = xi + b * n_local;
    const f_t* b_cov_vals_tangent = cov_vals_tangent + b * n_cov;
    const f_t* b_values_global_tangent = values_global_tangent + b * n_global;
    const f_t* b_xi_tangent = xi_tangent + b * n_local;
    const f_t* b_values_local_tangent = values_local_tangent + b * n_local;
    f_t* b_out_tangent = out_tangent + b * n_local;

    if (levels[idx] != level) {
        b_out_tangent[idx] = b_values_local_tangent[idx];
        return;
    }

    f_t pts[(MAX_K + 1) * N_DIM];
    f_t vec[MAX_K + 1];
    f_t vec_tangent[MAX_K + 1];
    f_t mat[((MAX_K + 1) * (MAX_K + 2)) / 2];
    f_t mat_tangent[((MAX_K + 1) * (MAX_K + 2)) / 2];

    for (size_t i = 0; i < k; ++i) {
        size_t neighbor_idx = neighbors[idx * k + i];
        vec[i] = b_values_global[neighbor_idx];
        vec_tangent[i] = b_values_global_tangent[neighbor_idx];
        for (size_t j = 0; j < N_DIM; ++j) {
            pts[i * N_DIM + j] = points_global[neighbor_idx * N_DIM + j];
        }
    }
    for (size_t j = 0; j < N_DIM; ++j) {
        pts[k * N_DIM + j] = points_local[idx * N_DIM + j];
    }
    vec[k] = b_xi[idx];
    vec_tangent[k] = b_xi_tangent[idx];

    cov_lookup_matrix(pts, cov_bins, b_cov_vals, mat, k + 1, N_DIM, n_cov); // K
    cov_lookup_matrix(pts, cov_bins, b_cov_vals_tangent, mat_tangent, k + 1, N_DIM, n_cov); // dK
    cholesky(mat, k + 1); // L
    cholesky_jvp(mat, mat_tangent, k + 1); // dL

    solve_cholesky_forward(mat, vec, k, 1);
    solve_cholesky_forward(mat, vec_tangent, k, 1);
    b_out_tangent[idx] = dot(mat + tri(k, 0), vec_tangent, k + 1); // linear part
    b_out_tangent[idx] += dot(mat_tangent + tri(k, 0), vec, k + 1); // dL xi
    apply_cholesky(mat_tangent, vec, k, 1); // dLcc xi_c
    solve_cholesky_forward(mat, vec, k, 1); // Lcc^-1 dLcc xi_c
    b_out_tangent[idx] -= dot(mat + tri(k, 0), vec, k);
}

template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__host__ void refine_level_jvp(
    cudaStream_t stream,
    const f_t* points_global,
    const f_t* points_local,
    const i_t* neighbors,
    const i_t* levels,
    const i_t* level_dev,
    const f_t* cov_bins,
    const f_t* cov_vals,
    const f_t* values_global,
    const f_t* xi,
    const f_t* cov_vals_tangent,
    const f_t* values_global_tangent,
    const f_t* xi_tangent,
    const f_t* values_local_tangent,
    f_t* out_tangent,
    size_t k,
    size_t n_global,
    size_t n_local,
    size_t n_cov,
    size_t n_batches
) {
    i_t level;
    cudaMemcpyAsync(&level, level_dev, sizeof(i_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    size_t n_threads = n_local * n_batches;
    refine_level_jvp_kernel<MAX_K, N_DIM, i_t, f_t><<<cld(n_threads, 256), 256, 0, stream>>>(
        points_global, points_local, neighbors, levels, level, cov_bins, cov_vals, values_global, xi,
        cov_vals_tangent, values_global_tangent, xi_tangent, values_local_tangent, out_tangent,
        k, n_global, n_local, n_cov, n_threads);
}

// ---------------------------------------------------------------- vjp (transpose of the jvp)
template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__global__ void refine_level_vjp_kernel(
    const f_t* points_global,
    const f_t* points_local,
    const i_t* neighbors,
    const i_t* levels,
    i_t level,
    const f_t* cov_bins,
    const f_t* cov_vals, // (B, R)
    const f_t* values_global, // (B, N)
    const f_t* xi, // (B, M)
    const f_t* out_tangent, // (B, M) cotangent of the output tangent
    f_t* cov_vals_tangent, // (B, R), zero-initialized, atomics
    f_t* values_global_tangent, // (B, N), zero-initialized, atomics
    f_t* xi_tangent, // (B, M)
    f_t* values_local_tangent, // (B, M)
    size_t k,
    size_t n_global,
    size_t n_local,
    size_t n_cov,
    size_t n_threads
) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;
    size_t b = tid / n_local;
    size_t idx = tid % n_local;

    const f_t* b_cov_vals = cov_vals + b * n_cov;
    const f_t* b_values_global = values_global + b * n_global;
    const f_t* b_xi = xi + b * n_local;
    const f_t* b_out_tangent = out_tangent + b * n_local;
    f_t* b_cov_vals_tangent = cov_vals_tangent + b * n_cov;
    f_t* b_values_global_tangent = values_global_tangent + b * n_global;
    f_t* b_xi_tangent = xi_tangent + b * n_local;
    f_t* b_values_local_tangent = values_local_tangent + b * n_local;

    if (levels[idx] != level) {
        b_values_local_tangent[idx] = b_out_tangent[idx];
        b_xi_tangent[idx] = f_t(0);
        return;
    }
    b_values_local_tangent[idx] = f_t(0);

    f_t pts[(MAX_K + 1) * N_DIM];
    f_t vec[MAX_K + 1];
    f_t vec_tangent[MAX_K + 1];
    f_t mat[((MAX_K + 1) * (MAX_K + 2)) / 2];
    f_t mat_tangent[((MAX_K + 1) * (MAX_K + 2)) / 2];

    for (size_t i = 0; i < k; ++i) {
        size_t neighbor_idx = neighbors[idx * k + i];
        vec[i] = b_values_global[neighbor_idx];
        for (size_t j = 0; j < N_DIM; ++j) {
            pts[i * N_DIM + j] = points_global[neighbor_idx * N_DIM + j];
        }
    }
    for (size_t j = 0; j < N_DIM; ++j) {
        pts[k * N_DIM + j] = points_local[idx * N_DIM + j];
    }
    vec[k] = b_xi[idx];
    f_t fine_tangent = b_out_tangent[idx];

    cov_lookup_matrix(pts, cov_bins, b_cov_vals, mat, k + 1, N_DIM, n_cov);
    cholesky(mat, k + 1); // L
    solve_cholesky_forward(mat, vec, k, 1); // "xi_c"
    matmul(mat + tri(k, 0), &fine_tangent, vec_tangent, k + 1, 1, 1);
    solve_cholesky_backward(mat, vec_tangent, k, 1); // Lcc^-T

    // linear transpose
    b_xi_tangent[idx] = vec_tangent[k];
    for (size_t i = 0; i < k; ++i) {
        atomicAdd(b_values_global_tangent + neighbors[idx * k + i], vec_tangent[i]);
    }

    // covariance vjp
    mat_tangent[tri(k, k)] = fine_tangent * vec[k];
    for (size_t i = 0; i < k; ++i) {
        mat_tangent[tri(k, i)] = fine_tangent * vec[i];
        for (size_t j = 0; j <= i; ++j) {
            mat_tangent[tri(i, j)] = -vec_tangent[i] * vec[j];
        }
    }
    cholesky_vjp(mat, mat_tangent, k + 1);
    cov_lookup_matrix_vjp(pts, mat_tangent, cov_bins, b_cov_vals_tangent, k + 1, N_DIM, n_cov);
}

template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__host__ void refine_level_vjp(
    cudaStream_t stream,
    const f_t* points_global,
    const f_t* points_local,
    const i_t* neighbors,
    const i_t* levels,
    const i_t* level_dev,
    const f_t* cov_bins,
    const f_t* cov_vals,
    const f_t* values_global,
    const f_t* xi,
    const f_t* out_tangent,
    f_t* cov_vals_tangent,
    f_t* values_global_tangent,
    f_t* xi_tangent,
    f_t* values_local_tangent,
    size_t k,
    size_t n_global,
    size_t n_local,
    size_t n_cov,
    size_t n_batches
) {
    i_t level;
    cudaMemcpyAsync(&level, level_dev, sizeof(i_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaMemsetAsync(cov_vals_tangent, 0, n_batches * n_cov * sizeof(f_t), stream);
    cudaMemsetAsync(values_global_tangent, 0, n_batches * n_global * sizeof(f_t), stream);
    size_t n_threads = n_local * n_batches;
    refine_level_vjp_kernel<MAX_K, N_DIM, i_t, f_t><<<cld(n_threads, 256), 256, 0, stream>>>(
        points_global, points_local, neighbors, levels, level, cov_bins, cov_vals, values_global, xi,
        out_tangent, cov_vals_tangent, values_global_tangent, xi_tangent, values_local_tangent,
        k, n_global, n_local, n_cov, n_threads);
}
