// linalg.h
#pragma once

#include <cuda_runtime.h>

// vector dot product
template <int n>
__forceinline__ __device__ float dot(
    const float* a, // (n,)
    const float* b // (n,)
) {
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// compute the Cholesky decomposition L L.T = A
template <int n>
__forceinline__ __device__ void cholesky(
    const float* A, // (n, n)
    float* L // (n, n)
) {
    #pragma unroll
    for (int i = 0; i < n; ++i) {
        #pragma unroll
        for (int j = 0; j < i; ++j) {
            float sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < j; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
        }
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < i; ++k) {
            sum += L[i * n + k] * L[i * n + k];
        }
        L[i * n + i] = sqrtf(A[i * n + i] - sum);
    }
}

// solve A X = B given L, the Cholesky decomposition of A
template <int n, int m>
__forceinline__ __device__ void solve_cholesky(
    const float* L, // (n, n)
    const float* B, // (n, m)
    float* X // (n, m)
) {
    
    // Forward substitution
    #pragma unroll
    for (int i = 0; i < n; ++i) {
        #pragma unroll
        for (int j = 0; j < m; ++j) {
            X[i * m + j] = B[i * m + j];
            #pragma unroll
            for (int k = 0; k < i; ++k) {
                X[i * m + j] -= L[i * n + k] * X[k * m + j];
            }
            X[i * m + j] /= L[i * n + i];
        }
    }

    // Backward substitution
    #pragma unroll
    for (int i = n; i-- > 0;) {
        #pragma unroll
        for (int k = i + 1; k < n; ++k) {
            #pragma unroll
            for (int j = 0; j < m; ++j) {
                X[i * m + j] -= L[k * n + i] * X[k * m + j];
            }
        }
        #pragma unroll
        for (int j = 0; j < m; ++j) {
            X[i * m + j] /= L[i * n + i];
        }
    }
}

// convenience version of solve_cholesky for vectors
template <int n>
__forceinline__ __device__ void solve_cholesky(
    const float* L, // (n, n)
    const float* B, // (n,)
    float* x // (n,)
) {
    solve_cholesky<n, 1>(L, B, x);
}