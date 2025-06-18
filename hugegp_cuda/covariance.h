// covariance.h
#pragma once

#include <cmath>
#include <cuda_runtime.h>

#include "common.h"

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