// linalg.h
#pragma once

#include <cuda_runtime.h>

// convenience copy function
template <int n>
__forceinline__ __device__ void vec_copy(
    const float* a, // (n,)
    float* b // (n,)
) {
    for (int i = 0; i < n; ++i) {
        b[i] = a[i];
    }
}


// vector dot product
template <int n>
__forceinline__ __device__ float dot(
    const float* a, // (n,)
    const float* b // (n,)
) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// multiply C = A B
template <int n, int p, int m>
__forceinline__ __device__ void matmul(
    const float* A, // (n, p)
    const float* B, // (p, m)
    float* C // (n, m)
) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            C[i * m + j] = 0.0f;
            for (int k = 0; k < p; ++k) {
                C[i * m + j] += A[i * p + k] * B[k * m + j];
            }
        }
    }
}

// compute the Cholesky decomposition L L.T = A, assuming triangular matrix order and modifying A in place
template <int n>
__forceinline__ __device__ void cholesky(
    float* A // (n, n) lower triangular so actually n * (n + 1) / 2 entries
) {
    for (int i = 0; i < n; ++i) {
        // off-diagonal elements
        for (int j = 0; j < i; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < j; ++k) {
                sum += A[tri(i, k)] * A[tri(j, k)];
            }
            A[tri(i, j)] = (A[tri(i, j)] - sum) / A[tri(j, j)];
        }
        // diagonal elements
        float sum = 0.0f;
        for (int j = 0; j < i; ++j) {
            sum += A[tri(i, j)] * A[tri(i, j)];
        }
        A[tri(i, i)] = sqrtf(A[tri(i, i)] - sum);
    }
}

// solve A X = B given L, the Cholesky decomposition of A, assuming triangular matrix order and modifying B in place
template <int n, int m>
__forceinline__ __device__ void solve_cholesky(
    const float* L, // (n, n)
    float* B // (n, m)
) {

    // Forward substitution
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < i; ++k) {
                sum += L[tri(i, k)] * B[k * m + j];
            }
            B[i * m + j] = (B[i * m + j] - sum) / L[tri(i, i)];
        }
    }

    // Backward substitution
    for (int i = n; i-- > 0;) {
        for (int j = 0; j < m; ++j) {
            float sum = 0.0f;
            for (int k = i + 1; k < n; ++k) {
                sum += L[tri(k, i)] * B[k * m + j];
            }
            B[i * m + j] = (B[i * m + j] - sum) / L[tri(i, i)];
        }
    }
}


// compute the Cholesky decomposition L L.T = A, assuming a full matrix and storing L separately
template <int n>
__forceinline__ __device__ void cholesky_full_pure(
    const float* A, // (n, n)
    float* L // (n, n)
) {
    for (int i = 0; i < n; ++i) {
        // off-diagonal elements
        for (int j = 0; j < i; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < j; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
        }
        // diagonal elements
        float sum = 0.0f;
        for (int j = 0; j < i; ++j) {
            sum += L[i * n + j] * L[i * n + j];
        }
        L[i * n + i] = sqrtf(A[i * n + i] - sum);
    }
}

// solve A X = B given L, the Cholesky decomposition of A, assuming a full matrix and storing X separately
template <int n, int m>
__forceinline__ __device__ void solve_cholesky_full_pure(
    const float* L, // (n, n)
    const float* B, // (n, m)
    float* X // (n, m)
) {
    
    // Forward substitution
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            X[i * m + j] = B[i * m + j];
            for (int k = 0; k < i; ++k) {
                X[i * m + j] -= L[i * n + k] * X[k * m + j];
            }
            X[i * m + j] /= L[i * n + i];
        }
    }

    // Backward substitution
    for (int i = n; i-- > 0;) {
        for (int k = i + 1; k < n; ++k) {
            for (int j = 0; j < m; ++j) {
                X[i * m + j] -= L[k * n + i] * X[k * m + j];
            }
        }
        for (int j = 0; j < m; ++j) {
            X[i * m + j] /= L[i * n + i];
        }
    }
}

// apparently more numerically stable?

// template <int n>
// __forceinline__ __device__ float dot(const float* a, const float* b) {
//     float sum = 0.0f;
//     float c = 0.0f; // compensation
//     #pragma unroll
//     for (int i = 0; i < n; ++i) {
//         float prod = a[i] * b[i];
//         float y = prod - c;
//         float t = sum + y;
//         c = (t - sum) - y;
//         sum = t;
//     }
//     return sum;
// }