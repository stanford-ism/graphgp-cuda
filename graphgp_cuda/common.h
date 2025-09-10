// common.h
#pragma once

#include <cmath>
#include <cuda_runtime.h>

#define CUDA_LAUNCH(kernel, n_threads, stream, ...) do {                         \
    int threads_per_block = 256;                                                 \
    int n_blocks  = (n_threads + threads_per_block - 1) / threads_per_block;     \
    kernel<<<n_blocks, threads_per_block, 0, stream>>>(__VA_ARGS__, n_threads);  \
} while(0)

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

__forceinline__ __device__ float compute_distance(
    const float* point_a, // (d,)
    const float* point_b, // (d,)
    int n_dim
) {
    float dist = 0.0f;
    for (int i = 0; i < n_dim; ++i) {
        float diff = point_a[i] - point_b[i];
        dist += diff * diff;
    }
    return sqrtf(dist);
}

__forceinline__ __device__ float compute_square_distance(
    const float* point_a, // (d,)
    const float* point_b, // (d,)
    int n_dim
) {
    float dist = 0.0f;
    for (int i = 0; i < n_dim; ++i) {
        float diff = point_a[i] - point_b[i];
        dist += diff * diff;
    }
    return dist;
}

template <typename T>
__global__ void apply_permutation(T* dest, const T* src, const int* permutation, int n_threads) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_threads) return;
    dest[tid] = src[permutation[tid]];
}

__global__ void compute_inverse_permutation(const int* permutation, int* inv_permutation, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    inv_permutation[permutation[tid]] = tid;
}

// copy N elements from (B, N1) to (B, N2), where N <= min(N1, N2)
template <typename T>
__global__ void batch_copy(T *dest, const T *src, int n_batches, int n_dest, int n_src, int n_threads) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_threads) return;
    int n_copy = n_threads / n_batches;
    int b = tid / n_copy; // batch index
    int i = tid % n_copy; // index within batch
    if (b >= n_batches) return;
    dest[b * n_dest + i] = src[b * n_src + i];
}

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