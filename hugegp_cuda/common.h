// common.h
#pragma once

#include <cmath>
#include <cuda_runtime.h>

__forceinline__ __device__ int tri(int i, int j) {
    return (i * (i + 1)) / 2 + j;
}

__forceinline__ __device__ int searchsorted(const float* a, float v, int n) {
    int left = 0;
    int right = n;
    while (left < right) {
        int mid = left + (right - left) / 2;
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
    for (int i = 0; i < N_DIM; ++i) {
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


// copy N elements from (B, N1) to (B, N2), where N <= min(N1, N2)
template <typename T>
__global__ void batch_copy(T *dest, const T *src,  int n_batches, int n_dest, int n_src, int n_copy) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int b = tid / n_copy; // batch index
    int i = tid % n_copy; // index within batch
    if (b >= n_batches) return;
    dest[b * n_dest + i] = src[b * n_src + i];
}

// // set first N0 of (B, N) to value
// template <typename T>
// __global__ void batch_memset(T *dest, T value, int n_batches, int n0, int n) {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     int b = tid / n0; // batch index
//     int i = tid % n0; // index within batch
//     if (b >= n_batches) return;
//     dest[b * n + i] = value;
// }

// template <typename T>
// __global__ void fill_kernel(T* a, T value, size_t n) {
//     size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= n) return;
//     a[i] = value;
// }

template <typename T>
__global__ void arange_kernel(T* a, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = (T)i;
}

__host__ __device__ int floored_log2(int x) {
    if (x <= 0) return 0;  // define behavior for nonpositive
    return 31 - __builtin_clz(static_cast<uint32_t>(x));
}

__host__ __device__ uint32_t floored_log2(uint32_t x) {
    return (x > 0) ? 31 - __builtin_clz(x) : 0;  // returns 0 for x = 0
}

__host__ __device__ uint64_t floored_log2(uint64_t x) {
    return (x > 0) ? 63 - __builtin_clzll(x) : 0;  // returns 0 for x = 0
}