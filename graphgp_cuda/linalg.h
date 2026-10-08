// linalg.h
#pragma once

#include <cfloat>
#include <cuda_runtime.h>

// Pivots of the Cholesky decomposition below n * eps * A_ii are below floating point resolution and are dropped: L_ii
// and the column below it are set to zero, so that point carries no information to later rows. Every division by a
// zero pivot is replaced by zero, in the solves and in the derivatives (which hold the set of dropped pivots fixed).

template <typename f_t>
__forceinline__ __device__ f_t machine_eps();

template <>
__forceinline__ __device__ float machine_eps<float>() { return FLT_EPSILON; }

template <>
__forceinline__ __device__ double machine_eps<double>() { return DBL_EPSILON; }

// convenience copy function
template <typename f_t>
__forceinline__ __device__ void vec_copy(
    const f_t* a, // (n,)
    f_t* b, // (n,)
    int n
) {
    for (int i = 0; i < n; ++i) {
        b[i] = a[i];
    }
}

// vector dot product
template <typename f_t>
__forceinline__ __device__ f_t dot(
    const f_t* a, // (n,)
    const f_t* b, // (n,)
    int n
) {
    f_t sum = f_t(0);
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// multiply C = A B
template <typename f_t>
__forceinline__ __device__ void matmul(
    const f_t* A, // (n, p)
    const f_t* B, // (p, m)
    f_t* C, // (n, m)
    int n,
    int p,
    int m
) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            C[i * m + j] = f_t(0);
            for (int k = 0; k < p; ++k) {
                C[i * m + j] += A[i * p + k] * B[k * m + j];
            }
        }
    }
}

// multiply C = L B
template <typename f_t>
__forceinline__ __device__ void matmul_tri(
    const f_t* L, // (n, n) lower triangular
    const f_t* B, // (n, m)
    f_t* C, // (n, m)
    int n,
    int m
) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            C[i * m + j] = f_t(0);
            for (int k = 0; k <= i; ++k) {
                C[i * m + j] += L[tri(i, k)] * B[k * m + j];
            }
        }
    }
}

// compute the Cholesky decomposition L L.T = A, assuming triangular matrix order and modifying A in place
// pivots below n * eps * A_ii are dropped (L_ii = 0 and zero column below)
template <typename f_t>
__forceinline__ __device__ void cholesky(
    f_t* A, // (n, n) lower triangular so actually n * (n + 1) / 2 entries
    int n
) {
    for (int i = 0; i < n; ++i) {
        // off-diagonal elements
        for (int j = 0; j < i; ++j) {
            f_t sum = f_t(0);
            for (int k = 0; k < j; ++k) {
                sum += A[tri(i, k)] * A[tri(j, k)];
            }
            A[tri(i, j)] = (A[tri(j, j)] > f_t(0)) ? (A[tri(i, j)] - sum) / A[tri(j, j)] : f_t(0);
        }
        // diagonal elements
        f_t sum = f_t(0);
        for (int j = 0; j < i; ++j) {
            sum += A[tri(i, j)] * A[tri(i, j)];
        }
        f_t d = A[tri(i, i)] - sum;
        A[tri(i, i)] = (d > f_t(n) * machine_eps<f_t>() * A[tri(i, i)]) ? sqrt(d) : f_t(0);
    }
}

// compute L, dA -> dL in-place, where A is SPD and all are lower-triangular
template <typename f_t>
__forceinline__ __device__ void cholesky_jvp(
    const f_t *L,
    f_t *dA,
    int n
) {
    for (int i = 0; i < n; ++i) {
        // off-diagonal elements
        for (int j = 0; j < i; ++j) {
            f_t sum = f_t(0);
            for (int k = 0; k < j; ++k) {
                sum += L[tri(i, k)] * dA[tri(j, k)] + dA[tri(i, k)] * L[tri(j, k)];
            }
            sum += L[tri(i, j)] * dA[tri(j, j)];
            dA[tri(i, j)] = (L[tri(j, j)] > f_t(0)) ? (dA[tri(i, j)] - sum) / L[tri(j, j)] : f_t(0);
        }
        // diagonal element
        f_t sum = f_t(0);
        for (int j = 0; j < i; ++j) {
            sum += L[tri(i, j)] * dA[tri(i, j)];
        }
        dA[tri(i, i)] = (L[tri(i, i)] > f_t(0)) ? ((dA[tri(i, i)] / f_t(2)) - sum) / L[tri(i, i)] : f_t(0);
    }
}

// compute L, dL -> dA in-place, where all are lower-triangular
template <typename f_t>
__forceinline__ __device__ void cholesky_vjp(
    const f_t *L,
    f_t *dL,
    int n
) {
    for (int i = n; i-- > 0;) {
        for (int j = i + 1; j-- > 0;) {
            f_t sum = f_t(0);
            for (int k = j + 1; k <= i; ++k) {
                sum += dL[tri(i, k)] * L[tri(k, j)];
            }
            for (int k = i + 1; k < n; ++k) {
                // same as above just access lower triangular for dL_ik when k > i
                sum += dL[tri(k, i)] * L[tri(k, j)];
            }
            dL[tri(i, j)] = (L[tri(j, j)] > f_t(0)) ? ((dL[tri(i, j)] / f_t(2)) - sum) / L[tri(j, j)] : f_t(0);
        }
    }
}

// solve L X = B given lower triangular L
template <typename f_t>
__forceinline__ __device__ void solve_cholesky_forward(
    const f_t* L, // (n, n)
    f_t* B, // (n, m)
    int n,
    int m
) {

    // Forward substitution
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            f_t sum = f_t(0);
            for (int k = 0; k < i; ++k) {
                sum += L[tri(i, k)] * B[k * m + j];
            }
            B[i * m + j] = (L[tri(i, i)] > f_t(0)) ? (B[i * m + j] - sum) / L[tri(i, i)] : f_t(0);
        }
    }
}

// solve L.T X = B given lower triangular L
template <typename f_t>
__forceinline__ __device__ void solve_cholesky_backward(
    const f_t* L, // (n, n)
    f_t* B, // (n, m)
    int n,
    int m
) {
    // Backward substitution
    for (int i = n; i-- > 0;) {
        for (int j = 0; j < m; ++j) {
            f_t sum = f_t(0);
            for (int k = i + 1; k < n; ++k) {
                sum += L[tri(k, i)] * B[k * m + j];
            }
            B[i * m + j] = (L[tri(i, i)] > f_t(0)) ? (B[i * m + j] - sum) / L[tri(i, i)] : f_t(0);
        }
    }
}

// solve A X = B given L, the Cholesky decomposition of A, assuming triangular matrix order and modifying B in place
template <typename f_t>
__forceinline__ __device__ void solve_cholesky(
    const f_t* L, // (n, n)
    f_t* B, // (n, m)
    int n,
    int m
) {
    solve_cholesky_forward(L, B, n, m);
    solve_cholesky_backward(L, B, n, m);
}

// multiply L B in-place
template <typename f_t>
__forceinline__ __device__ void apply_cholesky(
    const f_t* L, // (n, n)
    f_t* B, // (n, m)
    int n,
    int m
) {
    for (int i = n; i-- > 0;) {
        for (int j = 0; j < m; ++j) {
            f_t sum = f_t(0);
            for (int k = 0; k <= i; ++k) {
                sum += L[tri(i, k)] * B[k * m + j];
            }
            B[i * m + j] = sum;
        }
    }
}