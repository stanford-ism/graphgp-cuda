// sort.h
#pragma once

#include <cuda_runtime.h>

// EXTREMELY SIMPLISTIC SORT
// Uses logic from https://github.com/ingowald/cudaBitonic/blob/master/cubit/cubit.h
// Should certainly replace with this eventually, just want something simple I can easily modify

__forceinline__ __device__ void sort_two(float* keys, int i, int j) {
    float k1 = keys[i];
    float k2 = keys[j];
    if (k1 > k2) {
        keys[i] = k2;
        keys[j] = k1;
    }
}

__global__ void sort_up(float* keys, int stride, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int s = tid & -stride;
    int l = tid + s;
    int r = l ^ (2 * stride - 1);
    if (r < n) sort_two(keys, l, r);
}

__global__ void sort_down(float* keys, int stride, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int s = tid & -stride;
    int l = tid + s;
    int r = l + stride;
    if (r < n) sort_two(keys, l, r);
}

__host__ void bitonic_sort(cudaStream_t stream, float* keys, int n) {
    int threads_per_block = 256;
    int n_blocks = (n + threads_per_block - 1) / threads_per_block;
    for (int u = 2; u <= n; u *= 2) {
        sort_up<<<n_blocks, threads_per_block, 0, stream>>>(keys, u, n);
        for (int d = u/2; d > 0; d /= 2) {
            sort_down<<<n_blocks, threads_per_block, 0, stream>>>(keys, d, n);
        }
    }
}