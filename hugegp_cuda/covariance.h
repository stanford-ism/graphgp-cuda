// covariance.h
#pragma once

#include <cmath>
#include <cuda_runtime.h>

#include "common.h"

__forceinline__ __device__ float cov_lookup(
    float r,
    const float *cov_bins,
    const float *cov_vals,
    size_t n_cov
) {
    size_t idx = searchsorted(cov_bins, r, n_cov);
    if (idx == 0) return cov_vals[0]; // not ideal, should have cov_bins[0] == 0.0f
    if (idx == n_cov) return cov_vals[n_cov - 1]; // outside bounds, return last value (should probably set to 0.0f)

    // inside bounds, interpolate
    float r0 = cov_bins[idx - 1];
    float r1 = cov_bins[idx];
    float c0 = cov_vals[idx - 1];
    float c1 = cov_vals[idx];
    if (r0 == r1) return c0; // avoid division by zero in case bins are somehow the same
    return c0 + (c1 - c0) * (r - r0) / (r1 - r0);
}

template <int N_DIM>
__forceinline__ __device__ void cov_lookup_matrix(
    const float *points, // (n, d)
    const float *cov_bins, // (R,)
    const float *cov_vals, // (R,)
    float *out, // (n, n) lower triangular so actually n * (n + 1) / 2 entries
    int n,
    size_t n_cov
) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            float r = compute_distance<N_DIM>(points + (i * N_DIM), points + (j * N_DIM));
            out[tri(i, j)] = cov_lookup(r, cov_bins, cov_vals, n_cov);
        }
    }
}

// compute two covariance matrices at the same time, useful for JVP
template <int N_DIM>
__forceinline__ __device__ void two_cov_lookup_matrix(
    const float *points, // (n, d)
    const float *cov_bins, // (R,)
    const float *cov_vals_1, // (R,) first covariance values
    const float *cov_vals_2,
    float *out_1, // (n, n) lower triangular so actually n * (n + 1) / 2 entries
    float *out_2,
    int n,
    size_t n_cov
) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            float r = compute_distance<N_DIM>(points + (i * N_DIM), points + (j * N_DIM));
            out_1[tri(i, j)] = cov_lookup(r, cov_bins, cov_vals_1, n_cov);
            out_2[tri(i, j)] = cov_lookup(r, cov_bins, cov_vals_2, n_cov);
        }
    }
}

// template <int N_DIM>
// __forceinline__ __device__ void cov_lookup_matrix_full(
//     const float *points_a, // (n, d)
//     const float *points_b, // (m, d)
//     const float *cov_bins, // (R,)
//     const float *cov_vals, // (R,)
//     float *out, // (n, m)
//     size_t n,
//     size_t m,
//     size_t n_cov
// ) {
//     for (int i = 0; i < n; ++i) {
//         for (int j = 0; j < m; ++j) {
//             float r = compute_distance<N_DIM>(points_a + (i * N_DIM), points_b + (j * N_DIM));
//             out[i * m + j] = cov_lookup(r, cov_bins, cov_vals, n_cov);
//         }
//     }
// }



// compute two covariance matrices at the same time, useful for JVP
// template <int n, int m, int N_DIM>
// __forceinline__ __device__ void cov_lookup_matrix_full_2(
//     const float *points_a, // (n, d)
//     const float *points_b, // (m, d)
//     const float *cov_bins, // (R,)
//     const float *cov_vals_1, // (R,)
//     const float *cov_vals_2, // (R,) second covariance values
//     float *out_1, // (n, m)
//     float *out_2, // (n, m)
//     size_t n_cov
// ) {
//     for (int i = 0; i < n; ++i) {
//         for (int j = 0; j < m; ++j) {
//             float r = compute_distance<N_DIM>(points_a + (i * N_DIM), points_b + (j * N_DIM));
//             out_1[i * m + j] = cov_lookup(r, cov_bins, cov_vals_1, n_cov);
//             out_2[i * m + j] = cov_lookup(r, cov_bins, cov_vals_2, n_cov);
//         }
//     }
// }

// __forceinline__ __device__ float test_cov(
//     float r
// ) {
//     float amplitude = 1.0f; // amplitude of the kernel
//     float cutoff = 0.2f; // cutoff scale
//     float slope = -1.0f; // slope
//     float result = amplitude * powf(1.0f + (r * r) / (cutoff * cutoff), slope);
//     if (r == 0.0f) {
//         result += 1e-4f * result; // add small regularization term
//     }
//     return result;
// }


// template <int n, int m, int N_DIM>
// __forceinline__ __device__ void compute_test_cov_matrix(
//     const float* points_a, // (n, d)
//     const float* points_b, // (m, d)
//     float* out // (n, m)
// ) {
//     for (int i = 0; i < n; ++i) {
//         for (int j = 0; j < m; ++j) {
//             float r = compute_distance<N_DIM>(points_a + (i * N_DIM), points_b + (j * N_DIM));
//             out[i * m + j] = test_cov(r);
//         }
//     }
// }

