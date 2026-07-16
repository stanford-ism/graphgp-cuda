// refine_vector.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

#include "common.h"
#include "linalg.h"

// CUDA vector-valued GraphGP refinement kernels.
//
// These kernels intentionally mirror the scalar refine/refine_inv/refine_logdet
// kernels, but replace each local scalar Cholesky problem of size (k + 1) by a
// block Cholesky problem of size (k + 1) * N_COMP.  The current device-side
// covariance is the 3-component divergence-free RBF tensor used by the Python
// vector implementation in graphgp.extras.make_div_free_rbf_covariance.
//
// params = [ell, sigma2, jitter, periodic_flag]
// point ordering inside the local matrix is point-major component-major:
//   (coarse_0,c0), (coarse_0,c1), ..., (coarse_k-1,c2), (fine,c0), ...

static constexpr size_t GRAPHGP_VECTOR_NCOMP = 3;

template <typename f_t>
__forceinline__ __device__ f_t graphgp_vector_round(f_t x) {
    return round(x);
}

template <size_t N_DIM, typename f_t>
__forceinline__ __device__ f_t div_free_rbf_cov_3(
    const f_t* x,
    const f_t* y,
    size_t ci,
    size_t cj,
    const f_t* params
) {
    const f_t ell = params[0];
    const f_t sigma2 = params[1];
    const bool periodic = params[3] != f_t(0);
    const f_t ell2 = ell * ell;

    f_t r[GRAPHGP_VECTOR_NCOMP];
    f_t rr2 = f_t(0);
    #pragma unroll
    for (size_t d = 0; d < GRAPHGP_VECTOR_NCOMP; ++d) {
        f_t rd = (d < N_DIM) ? (y[d] - x[d]) : f_t(0);
        if (periodic) {
            rd -= graphgp_vector_round(rd);
        }
        r[d] = rd;
        rr2 += rd * rd;
    }

    const f_t base = exp(f_t(-0.5) * rr2 / ell2);
    const f_t a = (f_t(1.0) - f_t(0.5) * rr2 / ell2) * base;
    const f_t b = (f_t(0.5) / ell2) * base;
    const f_t delta = (ci == cj) ? f_t(1.0) : f_t(0.0);
    return sigma2 * (a * delta + b * r[ci] * r[cj]);
}

template <size_t MAX_K, size_t N_DIM, typename f_t>
__forceinline__ __device__ void div_free_rbf_cov_3_block_matrix(
    const f_t* pts,     // ((k + 1), N_DIM), coarse points followed by fine point
    const f_t* params,  // [ell, sigma2, jitter, periodic]
    f_t* out,           // packed lower triangular ((k + 1) * 3, (k + 1) * 3)
    size_t k
) {
    const size_t n_joint_points = k + 1;
    const size_t n = n_joint_points * GRAPHGP_VECTOR_NCOMP;
    const f_t jitter = params[2];

    for (size_t row = 0; row < n; ++row) {
        const size_t pi = row / GRAPHGP_VECTOR_NCOMP;
        const size_t ci = row % GRAPHGP_VECTOR_NCOMP;
        for (size_t col = 0; col <= row; ++col) {
            const size_t pj = col / GRAPHGP_VECTOR_NCOMP;
            const size_t cj = col % GRAPHGP_VECTOR_NCOMP;
            f_t v = div_free_rbf_cov_3<N_DIM>(
                pts + pi * N_DIM,
                pts + pj * N_DIM,
                ci,
                cj,
                params
            );
            // Match the Python vector implementation: jitter is added to the
            // conditional fine Schur complement, equivalently to the fine-fine
            // diagonal block before the joint Cholesky.
            if (jitter != f_t(0) && pi == k && pj == k && ci == cj) {
                v += jitter;
            }
            out[tri(row, col)] = v;
        }
    }
}

template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__global__ void refine_vector_kernel(
    const f_t* points,          // (N, d)
    const i_t* neighbors,       // (N - n0, k)
    const f_t* xi,              // (B, N - n0, 3)
    const f_t* params,          // (4,)
    f_t* values,                // (B, N, 3)
    size_t n0,
    size_t k,
    size_t n_points,
    size_t n_batches,
    size_t start_idx,
    size_t n_threads
) {
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;

    const size_t n_points_per_batch = n_threads / n_batches;
    const size_t b = tid / n_points_per_batch;
    const size_t idx = start_idx + (tid % n_points_per_batch);

    const f_t* b_xi = xi + b * (n_points - n0) * GRAPHGP_VECTOR_NCOMP;
    f_t* b_values = values + b * n_points * GRAPHGP_VECTOR_NCOMP;

    const size_t m = k * GRAPHGP_VECTOR_NCOMP;
    const size_t n = (k + 1) * GRAPHGP_VECTOR_NCOMP;

    f_t pts[(MAX_K + 1) * N_DIM];
    f_t vec[(MAX_K + 1) * GRAPHGP_VECTOR_NCOMP];
    f_t mat[((MAX_K + 1) * GRAPHGP_VECTOR_NCOMP * ((MAX_K + 1) * GRAPHGP_VECTOR_NCOMP + 1)) / 2];

    for (size_t i = 0; i < k; ++i) {
        const size_t neighbor_idx = neighbors[(idx - n0) * k + i];
        for (size_t d = 0; d < N_DIM; ++d) {
            pts[i * N_DIM + d] = points[neighbor_idx * N_DIM + d];
        }
        for (size_t c = 0; c < GRAPHGP_VECTOR_NCOMP; ++c) {
            vec[i * GRAPHGP_VECTOR_NCOMP + c] = b_values[neighbor_idx * GRAPHGP_VECTOR_NCOMP + c];
        }
    }

    for (size_t d = 0; d < N_DIM; ++d) {
        pts[k * N_DIM + d] = points[idx * N_DIM + d];
    }
    for (size_t c = 0; c < GRAPHGP_VECTOR_NCOMP; ++c) {
        vec[m + c] = b_xi[(idx - n0) * GRAPHGP_VECTOR_NCOMP + c];
    }

    div_free_rbf_cov_3_block_matrix<MAX_K, N_DIM>(pts, params, mat, k);
    cholesky(mat, (int)n);
    solve_cholesky_forward(mat, vec, (int)m, 1);

    for (size_t c = 0; c < GRAPHGP_VECTOR_NCOMP; ++c) {
        const size_t row = m + c;
        b_values[idx * GRAPHGP_VECTOR_NCOMP + c] = dot(mat + tri(row, 0), vec, (int)(row + 1));
    }
}

template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__host__ void refine_vector(
    cudaStream_t stream,
    const f_t* points,
    const i_t* neighbors,
    const i_t* offsets,
    const f_t* initial_values,
    const f_t* xi,
    const f_t* params,
    f_t* values,
    size_t n0,
    size_t k,
    size_t n_points,
    size_t n_levels,
    size_t n_batches
) {
    batch_copy<<<cld(n_batches * n0 * GRAPHGP_VECTOR_NCOMP, 256), 256, 0, stream>>>(
        initial_values,
        values,
        n_batches,
        n0 * GRAPHGP_VECTOR_NCOMP,
        n_points * GRAPHGP_VECTOR_NCOMP,
        n0 * GRAPHGP_VECTOR_NCOMP
    );

    i_t* offsets_host = (i_t*)malloc(n_levels * sizeof(i_t));
    if (offsets_host == nullptr) throw std::runtime_error("Failed to allocate memory for offsets on host");
    cudaMemcpy(offsets_host, offsets, n_levels * sizeof(i_t), cudaMemcpyDeviceToHost);

    for (size_t level = 1; level < n_levels; ++level) {
        const size_t start_idx = (size_t)offsets_host[level - 1];
        const size_t end_idx = (size_t)offsets_host[level];
        const size_t n_threads = (end_idx - start_idx) * n_batches;
        refine_vector_kernel<MAX_K, N_DIM, i_t, f_t><<<cld(n_threads, 256), 256, 0, stream>>>(
            points,
            neighbors,
            xi,
            params,
            values,
            n0,
            k,
            n_points,
            n_batches,
            start_idx,
            n_threads
        );
    }

    free(offsets_host);
}

template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__global__ void refine_vector_transpose_kernel(
    const f_t* points,
    const i_t* neighbors,
    const f_t* params,
    f_t* values,          // adjoint buffer (B, N, 3)
    f_t* xi,              // output adjoint (B, N - n0, 3)
    size_t n0,
    size_t k,
    size_t n_points,
    size_t n_batches,
    size_t start_idx,
    size_t n_threads
) {
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;

    const size_t n_points_per_batch = n_threads / n_batches;
    const size_t b = tid / n_points_per_batch;
    const size_t idx = start_idx + (tid % n_points_per_batch);

    f_t* b_values = values + b * n_points * GRAPHGP_VECTOR_NCOMP;
    f_t* b_xi = xi + b * (n_points - n0) * GRAPHGP_VECTOR_NCOMP;

    const size_t m = k * GRAPHGP_VECTOR_NCOMP;
    const size_t n = (k + 1) * GRAPHGP_VECTOR_NCOMP;

    f_t pts[(MAX_K + 1) * N_DIM];
    f_t coarse_adj[(MAX_K + 1) * GRAPHGP_VECTOR_NCOMP];
    f_t fine_adj[GRAPHGP_VECTOR_NCOMP];
    f_t mat[((MAX_K + 1) * GRAPHGP_VECTOR_NCOMP * ((MAX_K + 1) * GRAPHGP_VECTOR_NCOMP + 1)) / 2];

    for (size_t i = 0; i < k; ++i) {
        const size_t neighbor_idx = neighbors[(idx - n0) * k + i];
        for (size_t d = 0; d < N_DIM; ++d) {
            pts[i * N_DIM + d] = points[neighbor_idx * N_DIM + d];
        }
    }
    for (size_t d = 0; d < N_DIM; ++d) {
        pts[k * N_DIM + d] = points[idx * N_DIM + d];
    }
    for (size_t c = 0; c < GRAPHGP_VECTOR_NCOMP; ++c) {
        fine_adj[c] = b_values[idx * GRAPHGP_VECTOR_NCOMP + c];
    }

    div_free_rbf_cov_3_block_matrix<MAX_K, N_DIM>(pts, params, mat, k);
    cholesky(mat, (int)n);

    // coarse_adj = L_fc^T @ fine_adj
    for (size_t p = 0; p < m; ++p) {
        f_t s = f_t(0);
        for (size_t c = 0; c < GRAPHGP_VECTOR_NCOMP; ++c) {
            s += mat[tri(m + c, p)] * fine_adj[c];
        }
        coarse_adj[p] = s;
    }
    solve_cholesky_backward(mat, coarse_adj, (int)m, 1);

    for (size_t i = 0; i < k; ++i) {
        const size_t neighbor_idx = neighbors[(idx - n0) * k + i];
        for (size_t c = 0; c < GRAPHGP_VECTOR_NCOMP; ++c) {
            atomicAdd(b_values + neighbor_idx * GRAPHGP_VECTOR_NCOMP + c, coarse_adj[i * GRAPHGP_VECTOR_NCOMP + c]);
        }
    }

    // xi adjoint = L_ff^T @ fine_adj.
    for (size_t c = 0; c < GRAPHGP_VECTOR_NCOMP; ++c) {
        f_t s = f_t(0);
        for (size_t j = c; j < GRAPHGP_VECTOR_NCOMP; ++j) {
            s += mat[tri(m + j, m + c)] * fine_adj[j];
        }
        atomicAdd(b_xi + (idx - n0) * GRAPHGP_VECTOR_NCOMP + c, s);
    }
}

template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__host__ void refine_vector_transpose(
    cudaStream_t stream,
    const f_t* points,
    const i_t* neighbors,
    const i_t* offsets,
    const f_t* params,
    const f_t* values,
    f_t* values_buffer,
    f_t* initial_values,
    f_t* xi,
    size_t n0,
    size_t k,
    size_t n_points,
    size_t n_levels,
    size_t n_batches
) {
    cudaMemcpyAsync(values_buffer, values, n_batches * n_points * GRAPHGP_VECTOR_NCOMP * sizeof(f_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemsetAsync(xi, 0, n_batches * (n_points - n0) * GRAPHGP_VECTOR_NCOMP * sizeof(f_t), stream);

    i_t* offsets_host = (i_t*)malloc(n_levels * sizeof(i_t));
    if (offsets_host == nullptr) throw std::runtime_error("Failed to allocate memory for offsets on host");
    cudaMemcpy(offsets_host, offsets, n_levels * sizeof(i_t), cudaMemcpyDeviceToHost);

    for (size_t level = n_levels; level-- > 1;) {
        const size_t start_idx = (size_t)offsets_host[level - 1];
        const size_t end_idx = (size_t)offsets_host[level];
        const size_t n_threads = (end_idx - start_idx) * n_batches;
        refine_vector_transpose_kernel<MAX_K, N_DIM, i_t, f_t><<<cld(n_threads, 256), 256, 0, stream>>>(
            points,
            neighbors,
            params,
            values_buffer,
            xi,
            n0,
            k,
            n_points,
            n_batches,
            start_idx,
            n_threads
        );
    }

    batch_copy<<<cld(n_batches * n0 * GRAPHGP_VECTOR_NCOMP, 256), 256, 0, stream>>>(
        values_buffer,
        initial_values,
        n_batches,
        n_points * GRAPHGP_VECTOR_NCOMP,
        n0 * GRAPHGP_VECTOR_NCOMP,
        n0 * GRAPHGP_VECTOR_NCOMP
    );

    free(offsets_host);
}

template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__global__ void refine_vector_inv_kernel(
    const f_t* points,
    const i_t* neighbors,
    const f_t* values,
    const f_t* params,
    f_t* xi,
    size_t n0,
    size_t k,
    size_t n_points,
    size_t n_batches,
    size_t start_idx,
    size_t n_threads
) {
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;

    const size_t n_points_per_batch = n_threads / n_batches;
    const size_t b = tid / n_points_per_batch;
    const size_t idx = start_idx + (tid % n_points_per_batch);

    const f_t* b_values = values + b * n_points * GRAPHGP_VECTOR_NCOMP;
    f_t* b_xi = xi + b * (n_points - n0) * GRAPHGP_VECTOR_NCOMP;

    const size_t m = k * GRAPHGP_VECTOR_NCOMP;
    const size_t n = (k + 1) * GRAPHGP_VECTOR_NCOMP;

    f_t pts[(MAX_K + 1) * N_DIM];
    f_t vec[(MAX_K + 1) * GRAPHGP_VECTOR_NCOMP];
    f_t rhs[GRAPHGP_VECTOR_NCOMP];
    f_t mat[((MAX_K + 1) * GRAPHGP_VECTOR_NCOMP * ((MAX_K + 1) * GRAPHGP_VECTOR_NCOMP + 1)) / 2];

    for (size_t i = 0; i < k; ++i) {
        const size_t neighbor_idx = neighbors[(idx - n0) * k + i];
        for (size_t d = 0; d < N_DIM; ++d) {
            pts[i * N_DIM + d] = points[neighbor_idx * N_DIM + d];
        }
        for (size_t c = 0; c < GRAPHGP_VECTOR_NCOMP; ++c) {
            vec[i * GRAPHGP_VECTOR_NCOMP + c] = b_values[neighbor_idx * GRAPHGP_VECTOR_NCOMP + c];
        }
    }
    for (size_t d = 0; d < N_DIM; ++d) {
        pts[k * N_DIM + d] = points[idx * N_DIM + d];
    }

    div_free_rbf_cov_3_block_matrix<MAX_K, N_DIM>(pts, params, mat, k);
    cholesky(mat, (int)n);
    solve_cholesky_forward(mat, vec, (int)m, 1);

    for (size_t c = 0; c < GRAPHGP_VECTOR_NCOMP; ++c) {
        f_t mean = f_t(0);
        for (size_t p = 0; p < m; ++p) {
            mean += mat[tri(m + c, p)] * vec[p];
        }
        rhs[c] = b_values[idx * GRAPHGP_VECTOR_NCOMP + c] - mean;
    }

    // Solve L_ff z = rhs where L_ff is the lower-right 3x3 block of mat.
    for (size_t i = 0; i < GRAPHGP_VECTOR_NCOMP; ++i) {
        f_t s = f_t(0);
        for (size_t j = 0; j < i; ++j) {
            s += mat[tri(m + i, m + j)] * rhs[j];
        }
        rhs[i] = (rhs[i] - s) / mat[tri(m + i, m + i)];
    }

    for (size_t c = 0; c < GRAPHGP_VECTOR_NCOMP; ++c) {
        b_xi[(idx - n0) * GRAPHGP_VECTOR_NCOMP + c] = rhs[c];
    }
}

template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__host__ void refine_vector_inv(
    cudaStream_t stream,
    const f_t* points,
    const i_t* neighbors,
    const i_t* offsets,
    const f_t* values,
    const f_t* params,
    f_t* initial_values,
    f_t* xi,
    size_t n0,
    size_t k,
    size_t n_points,
    size_t n_levels,
    size_t n_batches
) {
    batch_copy<<<cld(n_batches * n0 * GRAPHGP_VECTOR_NCOMP, 256), 256, 0, stream>>>(
        values,
        initial_values,
        n_batches,
        n_points * GRAPHGP_VECTOR_NCOMP,
        n0 * GRAPHGP_VECTOR_NCOMP,
        n0 * GRAPHGP_VECTOR_NCOMP
    );

    i_t* offsets_host = (i_t*)malloc(n_levels * sizeof(i_t));
    if (offsets_host == nullptr) throw std::runtime_error("Failed to allocate memory for offsets on host");
    cudaMemcpy(offsets_host, offsets, n_levels * sizeof(i_t), cudaMemcpyDeviceToHost);

    for (size_t level = n_levels; level-- > 1;) {
        const size_t start_idx = (size_t)offsets_host[level - 1];
        const size_t end_idx = (size_t)offsets_host[level];
        const size_t n_threads = (end_idx - start_idx) * n_batches;
        refine_vector_inv_kernel<MAX_K, N_DIM, i_t, f_t><<<cld(n_threads, 256), 256, 0, stream>>>(
            points,
            neighbors,
            values,
            params,
            xi,
            n0,
            k,
            n_points,
            n_batches,
            start_idx,
            n_threads
        );
    }

    free(offsets_host);
}

template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__global__ void refine_vector_logdet_kernel(
    const f_t* points,
    const i_t* neighbors,
    const f_t* params,
    f_t* logdet,
    size_t n0,
    size_t k,
    size_t n_batches,
    size_t start_idx,
    size_t n_threads
) {
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;

    const size_t n_points_per_batch = n_threads / n_batches;
    const size_t b = tid / n_points_per_batch;
    const size_t idx = start_idx + (tid % n_points_per_batch);

    const size_t m = k * GRAPHGP_VECTOR_NCOMP;
    const size_t n = (k + 1) * GRAPHGP_VECTOR_NCOMP;

    f_t pts[(MAX_K + 1) * N_DIM];
    f_t mat[((MAX_K + 1) * GRAPHGP_VECTOR_NCOMP * ((MAX_K + 1) * GRAPHGP_VECTOR_NCOMP + 1)) / 2];

    for (size_t i = 0; i < k; ++i) {
        const size_t neighbor_idx = neighbors[(idx - n0) * k + i];
        for (size_t d = 0; d < N_DIM; ++d) {
            pts[i * N_DIM + d] = points[neighbor_idx * N_DIM + d];
        }
    }
    for (size_t d = 0; d < N_DIM; ++d) {
        pts[k * N_DIM + d] = points[idx * N_DIM + d];
    }

    div_free_rbf_cov_3_block_matrix<MAX_K, N_DIM>(pts, params, mat, k);
    cholesky(mat, (int)n);

    f_t s = f_t(0);
    for (size_t c = 0; c < GRAPHGP_VECTOR_NCOMP; ++c) {
        s += log(mat[tri(m + c, m + c)]);
    }
    atomicAdd(logdet + b, s);
}

template <size_t MAX_K, size_t N_DIM, typename i_t, typename f_t>
__host__ void refine_vector_logdet(
    cudaStream_t stream,
    const f_t* points,
    const i_t* neighbors,
    const i_t* offsets,
    const f_t* params,
    f_t* logdet,
    size_t n0,
    size_t k,
    size_t n_levels,
    size_t n_batches
) {
    cudaMemsetAsync(logdet, 0, n_batches * sizeof(f_t), stream);

    i_t* offsets_host = (i_t*)malloc(n_levels * sizeof(i_t));
    if (offsets_host == nullptr) throw std::runtime_error("Failed to allocate memory for offsets on host");
    cudaMemcpy(offsets_host, offsets, n_levels * sizeof(i_t), cudaMemcpyDeviceToHost);

    for (size_t level = 1; level < n_levels; ++level) {
        const size_t start_idx = (size_t)offsets_host[level - 1];
        const size_t end_idx = (size_t)offsets_host[level];
        const size_t n_threads = (end_idx - start_idx) * n_batches;
        refine_vector_logdet_kernel<MAX_K, N_DIM, i_t, f_t><<<cld(n_threads, 256), 256, 0, stream>>>(
            points,
            neighbors,
            params,
            logdet,
            n0,
            k,
            n_batches,
            start_idx,
            n_threads
        );
    }

    free(offsets_host);
}
