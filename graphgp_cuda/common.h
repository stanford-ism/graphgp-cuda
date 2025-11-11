// common.h
#pragma once

#include <cmath>
#include <cuda_runtime.h>

// divide and round up
template <typename T>
__forceinline__ __host__ size_t cld(size_t a, T b) {
    return (a + b - 1) / b;
}

// lower triangular matrix index
__forceinline__ __device__ size_t tri(size_t i, size_t j) {
    return (i * (i + 1)) / 2 + j;
}

__forceinline__ __host__ __device__ uint32_t floored_log2(uint32_t x) {
    return (x > 0) ? 31 - __builtin_clz(x) : 0;  // returns 0 for x = 0
}

__forceinline__ __host__ __device__ uint64_t floored_log2(uint64_t x) {
    return (x > 0) ? 63 - __builtin_clzll(x) : 0;  // returns 0 for x = 0
}

template <typename T>
__global__ void arange_kernel(T* a, size_t n) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) a[tid] = (T)tid;
}

template <typename f_t>
__forceinline__ __device__ f_t compute_square_distance(
    const f_t* point_a, // (d,)
    const f_t* point_b, // (d,)
    size_t n_dim
) {
    f_t dist = f_t(0);
    for (size_t i = 0; i < n_dim; ++i) {
        f_t diff = point_a[i] - point_b[i];
        dist += diff * diff;
    }
    return dist;
}

// binary search for insertion index
template <typename f_t>
__forceinline__ __device__ size_t searchsorted(const f_t* a, f_t v, size_t n) {
    size_t left = 0;
    size_t right = n;
    while (left < right) {
        size_t mid = left + (right - left) / 2;
        if (a[mid] < v) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

// copy N elements from (B, N1) to (B, N2), where N <= min(N1, N2)
template <typename T>
__global__ void batch_copy(const T *src, T *dest, size_t n_batches, size_t n_src, size_t n_dest, size_t n) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * n_batches) return;
    size_t b = tid / n; // batch index
    size_t i = tid % n; // index within batch
    dest[b * n_dest + i] = src[b * n_src + i];
}

// compute indices to undo a permutation
template <typename i_t>
__global__ void compute_inverse_permutation(const i_t* permutation, i_t* inv_permutation, size_t n) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    inv_permutation[permutation[tid]] = tid;
}

// copy a row from (N, d) into (N,)
template <typename T>
__global__ void copy_row(const T* src, T* dest, size_t n_dim, size_t d, size_t n) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    dest[tid] = src[tid * n_dim + d];
}

// copy from (N,) into a row of (N, d)
template <typename T>
__global__ void copy_row_back(const T* src, T* dest, size_t n_dim, size_t d, size_t n) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    dest[tid * n_dim + d] = src[tid];
}

// permute a row of values into 1d temp array (DANGER: must have all permutation values >= shift)
template <typename T, typename i_t>
__global__ void permute_row(const T* src, const i_t* permutation, T* dest, size_t n_dim, size_t d, size_t shift, size_t n) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    dest[tid] = src[(permutation[tid] - shift) * n_dim + d];
}

// permute a row of values into 1d temp array (DANGER: must have all permutation values >= shift)
template <typename T, typename i_t>
__global__ void permute_row_with_dims(const T* src, const i_t* permutation, const i_t* dims, T* dest, size_t n_dim, size_t shift, size_t n) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    dest[tid] = src[(permutation[tid] - shift) * n_dim + dims[tid]];
}

template <typename T, typename i_t>
__global__ void permute_rows(const T* values_in, const i_t* permutation, T* values_out, size_t n_dim, size_t n) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * n_dim) return;
    size_t row = tid / n_dim;
    size_t col = tid % n_dim;
    values_out[tid] = values_in[permutation[row] * n_dim + col];
}

template <typename T, typename i_t>
__host__ void permute(cudaStream_t stream, T* values, T* temp, const i_t* permutation, size_t n_dim, size_t shift, size_t n) {
    for (int d = 0; d < n_dim; ++d) {
        permute_row<<<cld(n, 256), 256, 0, stream>>>(values, permutation, temp, n_dim, d, shift, n);
        copy_row_back<<<cld(n, 256), 256, 0, stream>>>(temp, values, n_dim, d, n);
    }
}

