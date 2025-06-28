// refine.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cublas_v2.h>
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
    size_t n_points,
    size_t n_cov,
    size_t n_batches, // number of batches, affects cov_vals, xi, and values
    size_t start_idx, // = offsets[level]
    size_t n_threads // = (end_idx - start_idx) * n_batches
) {
    // compute global index of the point to refine
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_threads) return;
    size_t n_points_per_batch = n_threads / n_batches;
    size_t b = tid / n_points_per_batch; // batch index
    size_t idx = start_idx + (tid % n_points_per_batch); // point index within batch

    // batched memory access
    const float *b_cov_vals = cov_vals + b * n_cov;
    const float *b_xi = xi + b * n_points;
    float *b_values = values + b * n_points;

    // define working variables, these should fit on register
    float pts[(K_COARSE + 1) * N_DIM]; // fine point + K coarse points
    float vec[K_COARSE + 1];
    float mat[((K_COARSE + 1) * (K_COARSE + 2)) / 2]; // lower triangular matrix for joint covariance

    // load coarse points, coarse values, fine points, and fine xi
    for (int i = 0; i < K_COARSE; ++i) {
        size_t neighbor_idx = neighbors[idx * K_COARSE + i];
        vec[K_COARSE - 1 - i] = b_values[neighbor_idx]; // order is decreasing distance to fine point
        for (int j = 0; j < N_DIM; ++j) {
            pts[(K_COARSE - 1 - i) * N_DIM + j] = points[neighbor_idx * N_DIM + j];
        }
    }
    for (int j = 0; j < N_DIM; ++j) {
        pts[K_COARSE * N_DIM + j] = points[idx * N_DIM + j];
    }
    vec[K_COARSE] = b_xi[idx]; // fine xi at the end of coarse values vector

    // build joint covariance matrix
    cov_lookup_matrix_triangular<K_COARSE, N_DIM>(pts, cov_bins, b_cov_vals, mat, n_cov); // Kcc
    cov_lookup_matrix_full<1, K_COARSE, N_DIM>(pts + K_COARSE * N_DIM, pts, cov_bins, b_cov_vals, mat + tri(K_COARSE, 0), n_cov); // Kfc
    mat[tri(K_COARSE, K_COARSE)] = cov_lookup(0.0f, cov_bins, b_cov_vals, n_cov); // Kff

    // factorize and generate fine value
    cholesky<K_COARSE + 1>(mat); // L
    solve_cholesky_forward<K_COARSE, 1>(mat, vec); // x = Lcc^-1 @ v
    b_values[idx] = dot<K_COARSE + 1>(mat + tri(K_COARSE, 0), vec); // v = L @ x
}



template <int K_COARSE, int N_DIM>
__host__ void refine(
    cudaStream_t stream,
    const float* points,
    const uint32_t* neighbors,
    const uint32_t* offsets,
    const float* cov_bins,
    const float* cov_vals,
    const float* initial_cholesky,
    const float* xi,
    float* values,
    size_t n_points,
    size_t n_levels,
    size_t n_cov,
    size_t n_batches // batch dim only affects cov_vals, initial_cholesky, xi, and output values
) {
    size_t n_threads;
    size_t threads_per_block = 256;
    size_t n_blocks;

    // copy offsets to host
    uint32_t *offsets_host;
    offsets_host = (uint32_t*)malloc(n_levels * sizeof(uint32_t));
    if (offsets_host == nullptr) throw std::runtime_error("Failed to allocate memory for offsets on host");
    cudaMemcpy(offsets_host, offsets, n_levels * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // multiply initial_cholesky by xi to get initial values
    n_threads = n_batches * offsets_host[0];
    n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
    batched_matvec_kernel<<<n_blocks, threads_per_block, 0, stream>>>(
        initial_cholesky, xi, values, n_batches, offsets_host[0], n_points
    );

    // iteratively refine levels
    for (size_t level = 0; level < n_levels; ++level) {
        uint32_t start_idx = offsets_host[level];
        uint32_t end_idx = level + 1 < n_levels ? offsets_host[level + 1] : n_points;
        n_threads = (end_idx - start_idx) * n_batches;
        n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
        refine_kernel<K_COARSE, N_DIM><<<n_blocks, threads_per_block, 0, stream>>>(
            points, neighbors, cov_bins, cov_vals, xi, values, n_points, n_cov, n_batches, start_idx, n_threads
        );
    }

    free(offsets_host);
}


/* CuBLAS matrix multiply code if I figure out how to link it */

// __global__ void fill_batch_pointers(
//     const float* A, // (B, N0, N0)
//     const float* X, // (B, N)
//     const float* Y, // (B, N)
//     float** Aarray, // (B,)
//     float** Xarray, // (B,)
//     float** Yarray, // (B,)
//     size_t n_batches,
//     size_t n_initial,
//     size_t n_points // total points
// ) {
//     size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= n_batches) return;
//     Aarray[tid] = const_cast<float*>(A + tid * n_initial * n_initial);
//     Xarray[tid] = const_cast<float*>(X + tid * n_points);
//     Yarray[tid] = const_cast<float*>(Y + tid * n_points);
// }

//     // set up cuBLAS handle
//     cublasHandle_t handle;
//     cublasCreate(&handle);
//     cublasSetStream(handle, stream);

//     // create device arrays for batch pointers
//     float **Aarray, **Xarray, **Yarray;
//     cudaMalloc(&Aarray, n_batches * sizeof(float*));
//     cudaMalloc(&Xarray, n_batches * sizeof(float*));
//     cudaMalloc(&Yarray, n_batches * sizeof(float*));
//     n_threads = n_batches;
//     n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
//     fill_batch_pointers<<<n_blocks, threads_per_block, 0, stream>>>(
//         initial_cholesky, xi, values,
//         Aarray, Xarray, Yarray,
//         n_batches, n_initial, n_points
//     );

//     // cuBLAS matrix multiply to compute initial values
//     const float alpha = 1.0f;
//     const float beta = 0.0f;
//     cublasSgemv(
//         handle, CUBLAS_OP_T,
//         n_initial, n_initial,
//         &alpha,
//         initial_cholesky, n_initial,
//         xi, 1,
//         &beta,
//         values, 1
//     );

//     // clean up cuBLAS
//     cudaStreamSynchronize(stream);
//     cudaFree(Aarray);
//     cudaFree(Xarray);
//     cudaFree(Yarray);
//     cublasDestroy(handle);