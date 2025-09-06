// sort.h
#pragma once

#include <cuda_runtime.h>

// EXTREMELY SIMPLISTIC SORT
// Uses logic from https://github.com/ingowald/cudaBitonic/blob/master/cubit/cubit.h
// Should certainly replace with this eventually, just want something simple I can easily modify

template <int N_DIM>
__forceinline__ __device__ void compare_swap_points(int* tags, float* points, int* split_dims, int* indices, float* ranges, int dim, int a, int b) {
    if (dim == -1) dim = split_dims[a]; // split_dims[b] should be the same
    int ta = tags[a];
    int tb = tags[b];
    float pa = points[a * N_DIM + dim];
    float pb = points[b * N_DIM + dim];

    // compare primarily by tags, secondarily by points, and bring split_dims and indices along
    if ((ta > tb) || ((ta == tb) && (pa > pb)) || ((ta == tb) && (pa == pb) && (a > b))) {
        tags[a] = tb;
        tags[b] = ta;

        // swap points
        for (int d = 0; d < N_DIM; d++) {
            pa = points[a * N_DIM + d];
            pb = points[b * N_DIM + d];
            points[a * N_DIM + d] = pb;
            points[b * N_DIM + d] = pa;
        }

        // swap split_dims
        ta = split_dims[a];
        tb = split_dims[b];
        split_dims[a] = tb;
        split_dims[b] = ta;

        // swap indices
        ta = indices[a];
        tb = indices[b];
        indices[a] = tb;
        indices[b] = ta;

        // swap ranges
        pa = ranges[a];
        pb = ranges[b];
        ranges[a] = pb;
        ranges[b] = pa;
    }
}

template <int N_DIM>
__global__ void sort_points_up(int* tags, float* points, int* split_dims, int* indices, float* ranges, int dim, int stride, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int s = tid & -stride;
    int l = tid + s;
    int r = l ^ (2 * stride - 1);
    if (r < n) compare_swap_points<N_DIM>(tags, points, split_dims, indices, ranges, dim, l, r);
}

template <int N_DIM>
__global__ void sort_points_down(int* tags, float* points, int* split_dims, int* indices, float* ranges, int dim, int stride, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int s = tid & -stride;
    int l = tid + s;
    int r = l + stride;
    if (r < n) compare_swap_points<N_DIM>(tags, points, split_dims, indices, ranges, dim, l, r);
}

template <int N_DIM>
__host__ void sort_points(
    cudaStream_t stream,
    int* tags,
    float* points,
    int* split_dims,
    int* indices,
    float* ranges,
    int dim, // if dim == -1, we use split_dims
    int n_points
) {
    int threads_per_block = 256;
    int n_blocks = (n_points + threads_per_block - 1) / threads_per_block;
    for (int u = 1; u <= n_points; u *= 2) {
        sort_points_up<N_DIM><<<n_blocks, threads_per_block, 0, stream>>>(tags, points, split_dims, indices, ranges, dim, u, n_points);
        for (int d = u/2; d > 0; d /= 2) {
            sort_points_down<N_DIM><<<n_blocks, threads_per_block, 0, stream>>>(tags, points, split_dims, indices, ranges, dim, d, n_points);
        }
    }
}




// ================= REFERENCE NON-POWER-OF-TWO BITONIC SORT ===================

template <typename T>
__forceinline__ __device__ void compare_swap(T* keys, int i, int j) {
    T k1 = keys[i];
    T k2 = keys[j];
    if (k1 > k2) {
        keys[i] = k2;
        keys[j] = k1;
    }
}

template <typename T>
__global__ void sort_up(T* keys, int stride, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int s = tid & -stride;
    int l = tid + s;
    int r = l ^ (2 * stride - 1);
    if ((r < n) && (l < n)) compare_swap(keys, l, r);
}

template <typename T>
__global__ void sort_down(T* keys, int stride, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int s = tid & -stride;
    int l = tid + s;
    int r = l + stride;
    if ((r < n) && (l < n)) compare_swap(keys, l, r);
}

template <typename T>
__host__ void bitonic_sort(cudaStream_t stream, T* keys, int n) {
    int threads_per_block = 256;
    int n_blocks = (n + threads_per_block - 1) / threads_per_block;
    for (int u = 1; u < n; u += u) {
        sort_up<<<n_blocks, threads_per_block, 0, stream>>>(keys, u, n);
        for (int d = u/2; d > 0; d /= 2) {
            sort_down<<<n_blocks, threads_per_block, 0, stream>>>(keys, d, n);
        }
    }
}