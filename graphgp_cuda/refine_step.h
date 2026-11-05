// refine_step.h
//
// One conditional step of the refinement: target points are drawn given the values of
// their neighbors among a set of source points,
//
//     target_value = K_tn K_nn^-1 source_values[neighbors] + std * xi,
//
// with one Cholesky factorization of the (k+1) x (k+1) joint covariance per target, as
// in refine.h. Nothing is assumed about the order or the layout of the points, so a
// caller can run the levels of a graph in its own layout (for example one block of
// points per device with halo values exchanged between the steps). Targets with
// valid == 0 are padding: they return 0 and take no part in the derivatives. The
// kernels are linear in (source_values, xi) for fixed covariance; the jvp/vjp kernels
// linearize the covariance dependence as refine_jvp/refine_vjp do.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "common.h"
#include "linalg.h"
#include "covariance.h"

// ---------------------------------------------------------------- forward
template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__global__ void refine_step_kernel(
    const f_t* source_points, // (N, d)
    const f_t* target_points, // (M, d)
    const i_t* neighbors, // (M, k) indices into the source points
    const i_t* valid, // (M,) nonzero: refine, zero: padding
    const f_t* cov_bins, // (R,)
    const f_t* cov_vals, // (B, R)
    const f_t* source_values, // (B, N)
    const f_t* xi, // (B, M)
    f_t* out, // (B, M)
    size_t k,
    size_t n_source,
    size_t n_target,
    size_t n_cov,
    size_t n_threads // = n_target * n_batches
) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;
    size_t b = tid / n_target;
    size_t idx = tid % n_target;

    const f_t* b_cov_vals = cov_vals + b * n_cov;
    const f_t* b_source_values = source_values + b * n_source;
    const f_t* b_xi = xi + b * n_target;
    f_t* b_out = out + b * n_target;

    if (valid[idx] == 0) {
        b_out[idx] = f_t(0);
        return;
    }

    f_t pts[(MAX_K + 1) * N_DIM];
    f_t vec[MAX_K + 1];
    f_t mat[((MAX_K + 1) * (MAX_K + 2)) / 2];

    for (size_t i = 0; i < k; ++i) {
        size_t neighbor_idx = neighbors[idx * k + i];
        vec[i] = b_source_values[neighbor_idx];
        for (size_t j = 0; j < N_DIM; ++j) {
            pts[i * N_DIM + j] = source_points[neighbor_idx * N_DIM + j];
        }
    }
    for (size_t j = 0; j < N_DIM; ++j) {
        pts[k * N_DIM + j] = target_points[idx * N_DIM + j];
    }
    vec[k] = b_xi[idx];

    cov_lookup_matrix(pts, cov_bins, b_cov_vals, mat, k + 1, N_DIM, n_cov);
    cholesky(mat, k + 1);
    solve_cholesky_forward(mat, vec, k, 1);
    b_out[idx] = dot(mat + tri(k, 0), vec, k + 1);
}

template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__host__ void refine_step(
    cudaStream_t stream,
    const f_t* source_points,
    const f_t* target_points,
    const i_t* neighbors,
    const i_t* valid,
    const f_t* cov_bins,
    const f_t* cov_vals,
    const f_t* source_values,
    const f_t* xi,
    f_t* out,
    size_t k,
    size_t n_source,
    size_t n_target,
    size_t n_cov,
    size_t n_batches
) {
    size_t n_threads = n_target * n_batches;
    refine_step_kernel<MAX_K, N_DIM, i_t, f_t><<<cld(n_threads, 256), 256, 0, stream>>>(
        source_points, target_points, neighbors, valid, cov_bins, cov_vals,
        source_values, xi, out, k, n_source, n_target, n_cov, n_threads);
}

// ---------------------------------------------------------------- linear transpose
// Transpose w.r.t. (source_values, xi) given the covariance.
template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__global__ void refine_step_transpose_kernel(
    const f_t* source_points,
    const f_t* target_points,
    const i_t* neighbors,
    const i_t* valid,
    const f_t* cov_bins,
    const f_t* cov_vals,
    const f_t* out_tangent, // (B, M) cotangent of the output
    f_t* source_values_tangent, // (B, N), zero-initialized, atomics
    f_t* xi_tangent, // (B, M)
    size_t k,
    size_t n_source,
    size_t n_target,
    size_t n_cov,
    size_t n_threads
) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;
    size_t b = tid / n_target;
    size_t idx = tid % n_target;

    const f_t* b_cov_vals = cov_vals + b * n_cov;
    const f_t* b_out_tangent = out_tangent + b * n_target;
    f_t* b_source_values_tangent = source_values_tangent + b * n_source;
    f_t* b_xi_tangent = xi_tangent + b * n_target;

    if (valid[idx] == 0) {
        b_xi_tangent[idx] = f_t(0);
        return;
    }

    f_t pts[(MAX_K + 1) * N_DIM];
    f_t vec[MAX_K + 1];
    f_t mat[((MAX_K + 1) * (MAX_K + 2)) / 2];

    for (size_t i = 0; i < k; ++i) {
        size_t neighbor_idx = neighbors[idx * k + i];
        for (size_t j = 0; j < N_DIM; ++j) {
            pts[i * N_DIM + j] = source_points[neighbor_idx * N_DIM + j];
        }
    }
    for (size_t j = 0; j < N_DIM; ++j) {
        pts[k * N_DIM + j] = target_points[idx * N_DIM + j];
    }
    f_t fine_tangent = b_out_tangent[idx];

    cov_lookup_matrix(pts, cov_bins, b_cov_vals, mat, k + 1, N_DIM, n_cov);
    cholesky(mat, k + 1);
    matmul(mat + tri(k, 0), &fine_tangent, vec, k + 1, 1, 1); // L[k, :]^T * ct
    b_xi_tangent[idx] = vec[k];
    solve_cholesky_backward(mat, vec, k, 1); // Lcc^-T
    for (size_t i = 0; i < k; ++i) {
        atomicAdd(b_source_values_tangent + neighbors[idx * k + i], vec[i]);
    }
}

template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__host__ void refine_step_transpose(
    cudaStream_t stream,
    const f_t* source_points,
    const f_t* target_points,
    const i_t* neighbors,
    const i_t* valid,
    const f_t* cov_bins,
    const f_t* cov_vals,
    const f_t* out_tangent,
    f_t* source_values_tangent,
    f_t* xi_tangent,
    size_t k,
    size_t n_source,
    size_t n_target,
    size_t n_cov,
    size_t n_batches
) {
    cudaMemsetAsync(source_values_tangent, 0, n_batches * n_source * sizeof(f_t), stream);
    size_t n_threads = n_target * n_batches;
    refine_step_transpose_kernel<MAX_K, N_DIM, i_t, f_t><<<cld(n_threads, 256), 256, 0, stream>>>(
        source_points, target_points, neighbors, valid, cov_bins, cov_vals, out_tangent,
        source_values_tangent, xi_tangent, k, n_source, n_target, n_cov, n_threads);
}

// ---------------------------------------------------------------- jvp (covariance and linear tangents)
template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__global__ void refine_step_jvp_kernel(
    const f_t* source_points,
    const f_t* target_points,
    const i_t* neighbors,
    const i_t* valid,
    const f_t* cov_bins,
    const f_t* cov_vals, // (B, R)
    const f_t* source_values, // (B, N)
    const f_t* xi, // (B, M)
    const f_t* cov_vals_tangent, // (B, R)
    const f_t* source_values_tangent, // (B, N)
    const f_t* xi_tangent, // (B, M)
    f_t* out_tangent, // (B, M)
    size_t k,
    size_t n_source,
    size_t n_target,
    size_t n_cov,
    size_t n_threads
) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;
    size_t b = tid / n_target;
    size_t idx = tid % n_target;

    const f_t* b_cov_vals = cov_vals + b * n_cov;
    const f_t* b_source_values = source_values + b * n_source;
    const f_t* b_xi = xi + b * n_target;
    const f_t* b_cov_vals_tangent = cov_vals_tangent + b * n_cov;
    const f_t* b_source_values_tangent = source_values_tangent + b * n_source;
    const f_t* b_xi_tangent = xi_tangent + b * n_target;
    f_t* b_out_tangent = out_tangent + b * n_target;

    if (valid[idx] == 0) {
        b_out_tangent[idx] = f_t(0);
        return;
    }

    f_t pts[(MAX_K + 1) * N_DIM];
    f_t vec[MAX_K + 1];
    f_t vec_tangent[MAX_K + 1];
    f_t mat[((MAX_K + 1) * (MAX_K + 2)) / 2];
    f_t mat_tangent[((MAX_K + 1) * (MAX_K + 2)) / 2];

    for (size_t i = 0; i < k; ++i) {
        size_t neighbor_idx = neighbors[idx * k + i];
        vec[i] = b_source_values[neighbor_idx];
        vec_tangent[i] = b_source_values_tangent[neighbor_idx];
        for (size_t j = 0; j < N_DIM; ++j) {
            pts[i * N_DIM + j] = source_points[neighbor_idx * N_DIM + j];
        }
    }
    for (size_t j = 0; j < N_DIM; ++j) {
        pts[k * N_DIM + j] = target_points[idx * N_DIM + j];
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
__host__ void refine_step_jvp(
    cudaStream_t stream,
    const f_t* source_points,
    const f_t* target_points,
    const i_t* neighbors,
    const i_t* valid,
    const f_t* cov_bins,
    const f_t* cov_vals,
    const f_t* source_values,
    const f_t* xi,
    const f_t* cov_vals_tangent,
    const f_t* source_values_tangent,
    const f_t* xi_tangent,
    f_t* out_tangent,
    size_t k,
    size_t n_source,
    size_t n_target,
    size_t n_cov,
    size_t n_batches
) {
    size_t n_threads = n_target * n_batches;
    refine_step_jvp_kernel<MAX_K, N_DIM, i_t, f_t><<<cld(n_threads, 256), 256, 0, stream>>>(
        source_points, target_points, neighbors, valid, cov_bins, cov_vals, source_values, xi,
        cov_vals_tangent, source_values_tangent, xi_tangent, out_tangent,
        k, n_source, n_target, n_cov, n_threads);
}

// ---------------------------------------------------------------- vjp (transpose of the jvp)
template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__global__ void refine_step_vjp_kernel(
    const f_t* source_points,
    const f_t* target_points,
    const i_t* neighbors,
    const i_t* valid,
    const f_t* cov_bins,
    const f_t* cov_vals, // (B, R)
    const f_t* source_values, // (B, N)
    const f_t* xi, // (B, M)
    const f_t* out_tangent, // (B, M) cotangent of the output tangent
    f_t* cov_vals_tangent, // (B, R), zero-initialized, atomics
    f_t* source_values_tangent, // (B, N), zero-initialized, atomics
    f_t* xi_tangent, // (B, M)
    size_t k,
    size_t n_source,
    size_t n_target,
    size_t n_cov,
    bool use_smem // accumulate the covariance cotangent per block in shared memory
) {
    extern __shared__ unsigned char smem_raw[];
    f_t* s_cov = reinterpret_cast<f_t*>(smem_raw);
    size_t b = blockIdx.y;
    f_t* b_cov_vals_tangent = cov_vals_tangent + b * n_cov;
    if (use_smem) {
        for (size_t i = threadIdx.x; i < n_cov; i += blockDim.x) s_cov[i] = f_t(0);
        __syncthreads();
    }
    f_t* cov_target = use_smem ? s_cov : b_cov_vals_tangent;

    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_target) {
        const f_t* b_cov_vals = cov_vals + b * n_cov;
        const f_t* b_source_values = source_values + b * n_source;
        const f_t* b_xi = xi + b * n_target;
        const f_t* b_out_tangent = out_tangent + b * n_target;
        f_t* b_source_values_tangent = source_values_tangent + b * n_source;
        f_t* b_xi_tangent = xi_tangent + b * n_target;

        if (valid[idx] == 0) {
            b_xi_tangent[idx] = f_t(0);
        } else {

            f_t pts[(MAX_K + 1) * N_DIM];
            f_t vec[MAX_K + 1];
            f_t vec_tangent[MAX_K + 1];
            f_t mat[((MAX_K + 1) * (MAX_K + 2)) / 2];
            f_t mat_tangent[((MAX_K + 1) * (MAX_K + 2)) / 2];

            for (size_t i = 0; i < k; ++i) {
                size_t neighbor_idx = neighbors[idx * k + i];
                vec[i] = b_source_values[neighbor_idx];
                for (size_t j = 0; j < N_DIM; ++j) {
                    pts[i * N_DIM + j] = source_points[neighbor_idx * N_DIM + j];
                }
            }
            for (size_t j = 0; j < N_DIM; ++j) {
                pts[k * N_DIM + j] = target_points[idx * N_DIM + j];
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
                atomicAdd(b_source_values_tangent + neighbors[idx * k + i], vec_tangent[i]);
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
            cov_lookup_matrix_vjp(pts, mat_tangent, cov_bins, cov_target, k + 1, N_DIM, n_cov);
        }
    }

    if (use_smem) {
        __syncthreads();
        for (size_t i = threadIdx.x; i < n_cov; i += blockDim.x) {
            if (s_cov[i] != f_t(0)) atomicAdd(b_cov_vals_tangent + i, s_cov[i]);
        }
    }
}

template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__host__ void refine_step_vjp(
    cudaStream_t stream,
    const f_t* source_points,
    const f_t* target_points,
    const i_t* neighbors,
    const i_t* valid,
    const f_t* cov_bins,
    const f_t* cov_vals,
    const f_t* source_values,
    const f_t* xi,
    const f_t* out_tangent,
    f_t* cov_vals_tangent,
    f_t* source_values_tangent,
    f_t* xi_tangent,
    size_t k,
    size_t n_source,
    size_t n_target,
    size_t n_cov,
    size_t n_batches
) {
    cudaMemsetAsync(cov_vals_tangent, 0, n_batches * n_cov * sizeof(f_t), stream);
    cudaMemsetAsync(source_values_tangent, 0, n_batches * n_source * sizeof(f_t), stream);
    size_t smem = n_cov * sizeof(f_t);
    bool use_smem = smem <= 48 * 1024;
    dim3 grid(cld(n_target, 256), n_batches);
    refine_step_vjp_kernel<MAX_K, N_DIM, i_t, f_t><<<grid, 256, use_smem ? smem : 0, stream>>>(
        source_points, target_points, neighbors, valid, cov_bins, cov_vals, source_values, xi,
        out_tangent, cov_vals_tangent, source_values_tangent, xi_tangent,
        k, n_source, n_target, n_cov, use_smem);
}
