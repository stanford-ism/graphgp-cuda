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
    if (idx == 0) return cov_vals[0];
    if (idx == n_cov) return cov_vals[n_cov - 1];

    // if inside bounds, interpolate
    float r0 = cov_bins[idx - 1];
    float r1 = cov_bins[idx];
    float c0 = cov_vals[idx - 1];
    float c1 = cov_vals[idx];
    return c0 + (c1 - c0) * (r - r0) / (r1 - r0);
}

template <int n, int m, int N_DIM>
__forceinline__ __device__ float compute_cov_lookup_matrix(
    const float *points_a, // (d,)
    const float *points_b, // (d,)
    const float *cov_bins, // (R,)
    const float *cov_vals, // (R,)
    float *out, // (n, m)
    size_t n_cov
) {
    #pragma unroll
    for (int i = 0; i < n; ++i) {
        #pragma unroll
        for (int j = 0; j < m; ++j) {
            float r = compute_distance<N_DIM>(points_a + (i * N_DIM), points_b + (j * N_DIM));
            out[i * m + j] = cov_lookup(r, cov_bins, cov_vals, n_cov);
        }
    }
}

__forceinline__ __device__ float test_cov(
    float r
) {
    float amplitude = 1.0f; // amplitude of the kernel
    float cutoff = 0.2f; // cutoff scale
    float slope = -1.0f; // slope
    float result = amplitude * powf(1.0f + (r * r) / (cutoff * cutoff), slope);
    if (r == 0.0f) {
        result += 1e-4f * result; // add small regularization term
    }
    return result;
}


template <int n, int m, int N_DIM>
__forceinline__ __device__ void compute_test_cov_matrix(
    const float* points_a, // (n, d)
    const float* points_b, // (m, d)
    float* out // (n, m)
) {
    #pragma unroll
    for (int i = 0; i < n; ++i) {
        #pragma unroll
        for (int j = 0; j < m; ++j) {
            float r = compute_distance<N_DIM>(points_a + (i * N_DIM), points_b + (j * N_DIM));
            out[i * m + j] = test_cov(r);
        }
    }
}