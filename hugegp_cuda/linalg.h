// linalg.h
#pragma once

#include <cuda_runtime.h>

// convenience copy function
__forceinline__ __device__ void vec_copy(
    const float* a, // (n,)
    float* b, // (n,)
    int n
) {
    for (int i = 0; i < n; ++i) {
        b[i] = a[i];
    }
}

// vector dot product
__forceinline__ __device__ float dot(
    const float* a, // (n,)
    const float* b, // (n,)
    int n
) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// multiply C = A B
__forceinline__ __device__ void matmul(
    const float* A, // (n, p)
    const float* B, // (p, m)
    float* C, // (n, m)
    int n,
    int p,
    int m
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

// multiply C = L B
__forceinline__ __device__ void matmul_tri(
    const float* L, // (n, n) lower triangular
    const float* B, // (n, m)
    float* C, // (n, m)
    int n,
    int m
) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            C[i * m + j] = 0.0f;
            for (int k = 0; k <= i; ++k) {
                C[i * m + j] += L[tri(i, k)] * B[k * m + j];
            }
        }
    }
}

// compute the Cholesky decomposition L L.T = A, assuming triangular matrix order and modifying A in place
__forceinline__ __device__ void cholesky(
    float* A, // (n, n) lower triangular so actually n * (n + 1) / 2 entries
    int n
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

// compute L, dA -> dL in-place, where A is SPD and all are lower-triangular
__forceinline__ __device__ void cholesky_jvp(
    const float *L,
    float *dA,
    int n
) {
    for (int i = 0; i < n; ++i) {
        // off-diagonal elements
        for (int j = 0; j < i; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < j; ++k) {
                sum += L[tri(i, k)] * dA[tri(j, k)] + dA[tri(i, k)] * L[tri(j, k)];
            }
            sum += L[tri(i, j)] * dA[tri(j, j)];
            dA[tri(i, j)] = (dA[tri(i, j)] - sum) / L[tri(j, j)];
        }
        // diagonal element
        float sum = 0.0f;
        for (int j = 0; j < i; ++j) {
            sum += L[tri(i, j)] * dA[tri(i, j)];
        }
        dA[tri(i, i)] = ((dA[tri(i, i)] / 2.0f) - sum) / L[tri(i, i)];
    }
}

// compute L, dL -> dA in-place, where all are lower-triangular
__forceinline__ __device__ void cholesky_vjp(
    const float *L,
    float *dL,
    int n
) {
    for (int i = n; i-- > 0;) {
        for (int j = i + 1; j-- > 0;) {
            float sum = 0.0f;
            for (int k = j + 1; k <= i; ++k) {
                sum += dL[tri(i, k)] * L[tri(k, j)];
            }
            for (int k = i + 1; k < n; ++k) {
                // same as above just access lower triangular for dL_ik when k > i
                sum += dL[tri(k, i)] * L[tri(k, j)];
            }
            dL[tri(i, j)] = ((dL[tri(i, j)] / 2.0f) - sum) * L[tri(j, j)];
        }
    }
}

__forceinline__ __device__ void solve_cholesky_forward(
    const float* L, // (n, n)
    float* B, // (n, m)
    int n,
    int m
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
}

// solve L.T X = B given lower triangular L
__forceinline__ __device__ void solve_cholesky_backward(
    const float* L, // (n, n)
    float* B, // (n, m)
    int n,
    int m
) {
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

// solve A X = B given L, the Cholesky decomposition of A, assuming triangular matrix order and modifying B in place
__forceinline__ __device__ void solve_cholesky(
    const float* L, // (n, n)
    float* B, // (n, m)
    int n,
    int m
) {
    solve_cholesky_forward(L, B, n, m);
    solve_cholesky_backward(L, B, n, m);
}




__global__ void batched_matvec_kernel(
    const float* A, // (B, n, n)
    const float* x, // (B, p)
    float* y, // (B, p)
    size_t n_batches,
    size_t n,
    size_t p // p >= n, will only use first n entries
) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_batches * n) return;

    size_t b = tid / n; // batch index
    size_t i = tid % n; // row index in y

    float sum = 0.0f;
    for (size_t j = 0; j < n; ++j) {
        sum += A[b * n * n + i * n + j] * x[b * p + j];
    }
    y[b * p + i] = sum;
}

__global__ void batched_transpose_matvec_kernel(
    const float* A, // (B, n, n)
    const float* x, // (B, p)
    float* y, // (B, p)
    size_t n_batches,
    size_t n,
    size_t p // p >= n, will only use first n entries
) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_batches * n) return;

    size_t b = tid / n; // batch index
    size_t i = tid % n; // row index in y

    float sum = 0.0f;
    for (size_t j = 0; j < n; ++j) {
        sum += A[b * n * n + j * n + i] * x[b * p + j]; // only difference is i <-> j from the above
    }
    y[b * p + i] = sum;
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