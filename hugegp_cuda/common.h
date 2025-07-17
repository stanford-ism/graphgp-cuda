// common.h
#pragma once

#include <cmath>
#include <cuda_runtime.h>

__forceinline__ __device__ int tri(int i, int j) {
    return (i * (i + 1)) / 2 + j;
}

// __global__ void add_to_indices(const float *a, const uint32_t *indices, float *out, size_t n) {
//     size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= n) return;
//     out[indices[tid]] += a[tid];
// }

__forceinline__ __device__ size_t searchsorted(const float* a, float v, size_t n) {
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

template <int N_DIM>
__forceinline__ __device__ float compute_distance(
    const float* point_a, // (d,)
    const float* point_b // (d,)
) {
    float dist = 0.0f;
    for (size_t i = 0; i < N_DIM; ++i) {
        float diff = point_a[i] - point_b[i];
        dist += diff * diff;
    }
    return sqrtf(dist);
}

template <int N_DIM>
__forceinline__ __device__ float compute_square_distance(
    const float* point_a, // (d,)
    const float* point_b // (d,)
) {
    float dist = 0.0f;
    for (int i = 0; i < N_DIM; ++i) {
        float diff = point_a[i] - point_b[i];
        dist += diff * diff;
    }
    return dist;
}

// concatenate (B, N0) and (B, N1) into (B, N0 + N1)
template <typename T>
__global__ void batch_concat(T *dest, const T *src1, const T *src2, size_t n_batches, size_t n0, size_t n1) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t n = n0 + n1;
    size_t b = tid / n; // batch index
    size_t i = tid % n; // index within batch
    if (b >= n_batches) return;
    if (i < n0) {
        dest[b * n + i] = src1[b * n0 + i];
    } else {
        dest[b * n + i] = src2[b * n1 + (i - n0)];
    }
}

// extract (B, N0) from (B, N) where N0 <= N
template <typename T>
__global__ void batch_extract(T *dest, const T *src, size_t n_batches, size_t n0, size_t n) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t b = tid / n0; // batch index
    size_t i = tid % n0; // index within batch
    if (b >= n_batches) return;
    dest[b * n0 + i] = src[b * n + i];
}


// copy from (B, N0) to (B, N) where N0 <= N
template <typename T>
__global__ void batch_copy(T *dest, const T *src,  size_t n_batches, size_t n0, size_t n) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t b = tid / n0; // batch index
    size_t i = tid % n0; // index within batch
    if (b >= n_batches) return;
    dest[b * n + i] = src[b * n0 + i];
}

// set first N0 of (B, N) to value
template <typename T>
__global__ void batch_memset(T *dest, T value, size_t n_batches, size_t n0, size_t n) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t b = tid / n0; // batch index
    size_t i = tid % n0; // index within batch
    if (b >= n_batches) return;
    dest[b * n + i] = value;
}

// // out[indices] = a, NOT out = a[indices]
// template <typename T>
// __global__ void restore_order(const T* a, const uint32_t* indices, T* out, size_t n) {
//     size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= n) return;
//     out[indices[tid]] = a[tid];
// }

__host__ __device__ uint32_t floored_log2(uint32_t x) {
    return (x > 0) ? 31 - __builtin_clz(x) : 0;  // returns 0 for x = 0
}

__host__ __device__ uint64_t floored_log2(uint64_t x) {
    return (x > 0) ? 63 - __builtin_clzll(x) : 0;  // returns 0 for x = 0
}