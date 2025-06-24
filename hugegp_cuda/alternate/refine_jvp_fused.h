// refine_jvp.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "common.h"
#include "linalg.h"
#include "covariance.h"

// This is a fused version, otherwise you could just call the forward pass twice f(x) and f(dx) since it's linear.
// This version will definitely be faster, but it's more delicate to integrate into jax's autodiff system.

template <int K_COARSE, int N_DIM>
__global__ void refine_jvp_linear_kernel(
    const float* points, // (N, d)
    const uint32_t* neighbors, // (N, k)
    const float* cov_bins, // (R,)
    const float* cov_vals, // (R,)
    const float* xi, // (N,)
    const float* xi_tangent, // (N,)
    float* values, // (N,)
    float* values_tangent, // (N,)
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
    float vec1t[K_COARSE]; // for tangents
    float vec2[K_COARSE];
    float mat1[(K_COARSE * (K_COARSE + 1)) / 2]; // lower triangular matrix for covariance

    // load fine point, coarse points, and coarse values
    for (int i = 0; i < N_DIM; ++i) {
        fine_point[i] = points[idx * N_DIM + i];
    }
    for (int i = 0; i < K_COARSE; ++i) {
        size_t neighbor_idx = neighbors[idx * K_COARSE + i];
        vec1[i] = values[neighbor_idx]; // assumed to be generated already
        vec1t[i] = values_tangent[neighbor_idx]; // assumed to be generated already
        for (int j = 0; j < N_DIM; ++j) {
            neighbor_points[i * N_DIM + j] = points[neighbor_idx * N_DIM + j];
        }
    }

    // load covariance matrices
    cov_lookup_matrix_triangular<K_COARSE, N_DIM>(neighbor_points, cov_bins, cov_vals, mat1, n_cov); // Kcc
    cov_lookup_matrix_full<1, K_COARSE, N_DIM>(fine_point, neighbor_points, cov_bins, cov_vals, vec2, n_cov); // Kfc
    float variance = cov_lookup(0.0f, cov_bins, cov_vals, n_cov); // Kff

    // compute conditional mean
    cholesky<K_COARSE>(mat1); // L
    solve_cholesky<K_COARSE, 1>(mat1, vec1); // Kcc^-1 @ v
    solve_cholesky<K_COARSE, 1>(mat1, vec1t); // Kcc^-1 @ vt
    float mean = dot<K_COARSE>(vec2, vec1); // Kfc @ (Kcc^-1 @ v)
    float mean_tangent = dot<K_COARSE>(vec2, vec1t); // Kfc @ (Kcc^-1 @ vt)

    // compute conditional variance
    vec_copy<K_COARSE>(vec2, vec1); // Kcf = Kfc
    solve_cholesky<K_COARSE, 1>(mat1, vec1); // Kcc^-1 @ Kcf
    variance -= dot<K_COARSE>(vec2, vec1); // Kff - Kfc @ (Kcc^-1 @ Kcf)
    variance = fmaxf(variance, 0.0f); // better not to rely on this and add jitter to cov_vals[0]

    // write conditional value
    values[idx] = mean + sqrtf(variance) * xi[idx];
    values_tangent[idx] = mean_tangent + sqrtf(variance) * xi_tangent[idx];
}

template <int K_COARSE, int N_DIM>
__host__ void refine_jvp_linear(
    cudaStream_t stream,
    const float* points,
    const uint32_t* offsets,
    const uint32_t* neighbors,
    const float* cov_bins,
    const float* cov_vals,
    const float* initial_values,
    const float* xi,
    const float* initial_values_tangent,
    const float* xi_tangent,
    float* values,
    float* values_tangent,
    size_t n_points,
    size_t n_levels,
    size_t n_cov
) {
    size_t n_threads;
    size_t threads_per_block = 256;
    size_t n_blocks;

    // copy offsets to host
    uint32_t *offsets_host;
    offsets_host = (uint32_t*)malloc(n_levels * sizeof(uint32_t));
    if (offsets_host == nullptr) throw std::runtime_error("Failed to allocate memory for offsets on host");
    cudaMemcpy(offsets_host, offsets, n_levels * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // copy initial values to output buffer
    cudaMemcpyAsync(values, initial_values, offsets_host[0] * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(values_tangent, initial_values_tangent, offsets_host[0] * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // iteratively refine levels
    for (size_t level = 0; level < n_levels; ++level) {
        uint32_t start_idx = offsets_host[level];
        uint32_t end_idx = level + 1 < n_levels ? offsets_host[level + 1] : n_points;
        n_threads = end_idx - start_idx;
        n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
        refine_jvp_linear_kernel<K_COARSE, N_DIM><<<n_blocks, threads_per_block, 0, stream>>>(
            points, neighbors, cov_bins, cov_vals, xi, xi_tangent, values, values_tangent, n_cov, start_idx, n_threads
        );
    }

    free(offsets_host);
}


// template <int K_COARSE, int N_DIM>
// __global__ void refine_jvp_cov_kernel(
//     const float* points, // (N, d)
//     const uint32_t* neighbors, // (N, k)
//     const float* cov_bins, // (R,)
//     const float* cov_vals, // (R,)
//     const float* xi, // (N,)
//     const float* cov_vals_tangent, // (R,)
//     float* values, // (N,)
//     float* values_tangent, // (N,)
//     size_t n_cov,
//     size_t start_idx,
//     size_t n_threads
// ) {
//     // compute global index of the point to refine
//     size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= n_threads) return;
//     size_t idx = start_idx + tid;

//     // define working variables, these should fit on register
//     float fine_point[N_DIM];
//     float neighbor_points[K_COARSE * N_DIM];
//     float vec1[K_COARSE];
//     float vec2[K_COARSE];
//     float vec3[K_COARSE];
//     float vec4[K_COARSE];
//     float mat1[(K_COARSE * (K_COARSE + 1)) / 2]; // lower triangular matrix for covariance
//     float mat2[(K_COARSE * (K_COARSE + 1)) / 2]; // lower triangular matrix for covariance_tangent

//     // load fine point, coarse points, and coarse values
//     for (int i = 0; i < N_DIM; ++i) {
//         fine_point[i] = points[idx * N_DIM + i];
//     }
//     for (int i = 0; i < K_COARSE; ++i) {
//         size_t neighbor_idx = neighbors[idx * K_COARSE + i];
//         vec1[i] = values_tangent[neighbor_idx]; // assumed to be generated already
//         for (int j = 0; j < N_DIM; ++j) {
//             neighbor_points[i * N_DIM + j] = points[neighbor_idx * N_DIM + j];
//         }
//     }

//     // compute mean tangent
//     cov_lookup_matrix_triangular_2<K_COARSE, N_DIM>(neighbor_points, cov_bins, cov_vals, cov_vals_tangent, mat1, mat2, n_cov); // Kcc and dKcc
//     cov_lookup_matrix_full_2<1, K_COARSE, N_DIM>(fine_point, neighbor_points, cov_bins, cov_vals, cov_vals_tangent, vec2, vec3, n_cov); // Kfc and dKfc
//     cholesky<K_COARSE>(mat1); // L
//     solve_cholesky<K_COARSE, 1>(mat1, vec1); // Kcc^-1 @ vt
//     float mean_tangent = dot<K_COARSE>(vec2, vec1); // Kfc @ (Kcc^-1 @ vt)
//     matmul<K_COARSE, K_COARSE, 1>(mat2, vec1, vec4); // dKcc @ (Kcc^-1 @ vt)
//     solve_cholesky<K_COARSE, 1>(mat1, vec4); // Kcc^-1 @ (dKcc @ (Kcc^-1 @ vt))
//     mean_tangent -= dot<K_COARSE>(vec2, vec4); // Kfc @ (Kcc^-1 @ (dKcc @ (Kcc^-1 @ vt)))

//     // compute conditional variance
//     float variance = cov_lookup(0.0f, cov_bins, cov_vals, n_cov); // Kff
//     vec_copy<K_COARSE>(vec2, vec1); // Kcf = Kfc
//     solve_cholesky<K_COARSE, 1>(mat1, vec1); // Kcc^-1 @ Kcf
//     variance -= dot<K_COARSE>(vec2, vec1); // Kff - Kfc @ (Kcc^-1 @ Kcf)
//     variance = fmaxf(variance, 0.0f); // better not to rely on this and add jitter to cov_vals[0]
    
//     // if variance is zero, we are done
//     if (variance == 0.0f) values_tangent[idx] = mean_tangent;
    
//     // compute noise tangent
//     float noise_tangent = cov_lookup(0.0f, cov_bins, cov_vals_tangent, n_cov); // dKff
//     solve_cholesky<K_COARSE, 1>(mat1, vec3); // Kcc^-1 @ dKcf
//     noise_tangent -= 2 * dot<K_COARSE>(vec2, vec3); // dKff - 2 * Kfc @ (Kcc^-1 @ dKcf)
//     matmul<K_COARSE, K_COARSE, 1>(mat2, vec1, vec4); // dKcc @ (Kcc^-1 @ Kcf)
//     solve_cholesky<K_COARSE, 1>(mat1, vec4); // Kcc^-1 @ (dKcc @ (Kcc^-1 @ Kcf))
//     noise_tangent += dot<K_COARSE>(vec2, vec4); // dKff - 2 * Kfc @ (Kcc^-1 @ dKcf) + Kfc @ (Kcc^-1 @ (dKcc @ (Kcc^-1 @ Kcf)))
//     noise_tangent /= (2 * sqrtf(variance));

//     // write conditional value
//     values_tangent[idx] = mean_tangent + noise_tangent * xi[idx];
// }

// template <int K_COARSE, int N_DIM>
// __host__ void refine_jvp_cov(
//     cudaStream_t stream,
//     const float* points,
//     const uint32_t* offsets,
//     const uint32_t* neighbors,
//     const float* cov_bins,
//     const float* cov_vals,
//     const float* initial_values,
//     const float* xi,
//     const float* cov_vals_tangent,
//     float* values_tangent,
//     size_t n_points,
//     size_t n_levels,
//     size_t n_cov
// ) {
//     size_t n_threads;
//     size_t threads_per_block = 256;
//     size_t n_blocks;

//     // copy offsets to host
//     uint32_t *offsets_host;
//     offsets_host = (uint32_t*)malloc(n_levels * sizeof(uint32_t));
//     if (offsets_host == nullptr) throw std::runtime_error("Failed to allocate memory for offsets on host");
//     cudaMemcpy(offsets_host, offsets, n_levels * sizeof(uint32_t), cudaMemcpyDeviceToHost);

//     // set initial level values_tangent to zero
//     cudaMemsetAsync(values_tangent, 0, offsets_host[0] * sizeof(float), stream);

//     // iteratively refine levels
//     for (size_t level = 0; level < n_levels; ++level) {
//         uint32_t start_idx = offsets_host[level];
//         uint32_t end_idx = level + 1 < n_levels ? offsets_host[level + 1] : n_points;
//         n_threads = end_idx - start_idx;
//         n_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
//         refine_jvp_cov_kernel<K_COARSE, N_DIM><<<n_blocks, threads_per_block, 0, stream>>>(
//             points, neighbors, cov_bins, cov_vals, xi, values, cov_vals_tangent, n_cov, start_idx, n_threads
//         );
//     }

//     free(offsets_host);
// }