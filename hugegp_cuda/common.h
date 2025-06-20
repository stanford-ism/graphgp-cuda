// common.h
#pragma once

#include <cmath>
#include <cuda_runtime.h>

__global__ void add_to_indices(const float *a, const uint32_t *indices, float *out, size_t n) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    out[indices[tid]] += a[tid];
}

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
    #pragma unroll
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
    #pragma unroll
    for (int i = 0; i < N_DIM; ++i) {
        float diff = point_a[i] - point_b[i];
        dist += diff * diff;
    }
    return dist;
}

// out[indices] = a, NOT out = a[indices]
template <typename T>
__global__ void restore_order(const T* a, const uint32_t* indices, T* out, size_t n) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    out[indices[tid]] = a[tid];
}

__host__ __device__ uint32_t floored_log2(uint32_t x) {
    return (x > 0) ? 31 - __builtin_clz(x) : 0;  // returns 0 for x = 0
}

__host__ __device__ uint64_t floored_log2(uint64_t x) {
    return (x > 0) ? 63 - __builtin_clzll(x) : 0;  // returns 0 for x = 0
}