// sort.h
#pragma once

#include <cuda_runtime.h>
#include "common.h"

// EXTREMELY SIMPLISTIC SORT
// Uses logic from https://github.com/ingowald/cudaBitonic/blob/master/cubit/cubit.h
// Should certainly replace with this eventually, just want something simple I can easily modify

// ======================= SORT FOR BUILDING TREE ================================

__forceinline__ __device__ void compare_swap_points(int* tags, float* points, int* split_dims, int* indices, float* ranges, int dim, int n_dim, int a, int b) {
    if (dim == -1) dim = split_dims[a]; // split_dims[b] should be the same
    int ta = tags[a];
    int tb = tags[b];
    float pa = points[a * n_dim + dim];
    float pb = points[b * n_dim + dim];

    // compare primarily by tags, secondarily by points, and bring split_dims and indices along
    if ((ta > tb) || ((ta == tb) && (pa > pb))) {
        tags[a] = tb;
        tags[b] = ta;

        // swap points
        for (int d = 0; d < n_dim; d++) {
            pa = points[a * n_dim + d];
            pb = points[b * n_dim + d];
            points[a * n_dim + d] = pb;
            points[b * n_dim + d] = pa;
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

__global__ void sort_points_up(int* tags, float* points, int* split_dims, int* indices, float* ranges, int dim, int n_dim, int stride, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int s = tid & -stride;
    int l = tid + s;
    int r = l ^ (2 * stride - 1);
    if (r < n) compare_swap_points(tags, points, split_dims, indices, ranges, dim, n_dim, l, r);
}

__global__ void sort_points_down(int* tags, float* points, int* split_dims, int* indices, float* ranges, int dim, int n_dim, int stride, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int s = tid & -stride;
    int l = tid + s;
    int r = l + stride;
    if (r < n) compare_swap_points(tags, points, split_dims, indices, ranges, dim, n_dim, l, r);
}

__host__ void sort_points(
    cudaStream_t stream,
    int* tags,
    float* points,
    int* split_dims,
    int* indices,
    float* ranges,
    int dim, // if dim == -1, we use split_dims
    int n_dim,
    int n_points
) {
    for (int u = 1; u <= n_points; u *= 2) {
        CUDA_LAUNCH(sort_points_up, n_points, stream, tags, points, split_dims, indices, ranges, dim, n_dim, u);
        for (int d = u/2; d > 0; d /= 2) {
            CUDA_LAUNCH(sort_points_down, n_points, stream, tags, points, split_dims, indices, ranges, dim, n_dim, d);
        }
    }
}


// ========================= SORT BY DEPTH =====================

__forceinline__ __device__ void compare_swap_by_depth(
    int* depths,
    float* points,
    int* indices,
    int* neighbors,
    int* permutation,
    int n_dim,
    int k,
    int a,
    int b
) {
    int da = depths[a];
    int db = depths[b];
    float pa;
    float pb;

    // compare by depths and bring other arrays along
    if ((da > db) || ((da == db) && (permutation[a] > permutation[b]))) {
        depths[a] = db;
        depths[b] = da;

        // swap points
        for (int d = 0; d < n_dim; d++) {
            pa = points[a * n_dim + d];
            pb = points[b * n_dim + d];
            points[a * n_dim + d] = pb;
            points[b * n_dim + d] = pa;
        }

        // swap indices
        da = indices[a];
        db = indices[b];
        indices[a] = db;
        indices[b] = da;

        // swap neighbors
        for (int j = 0; j < k; ++j) {
            int da = neighbors[a * k + j];
            int db = neighbors[b * k + j];
            neighbors[a * k + j] = db;
            neighbors[b * k + j] = da;
        }

        // swap permutation
        da = permutation[a];
        db = permutation[b];
        permutation[a] = db;
        permutation[b] = da;
    }
}

__global__ void sort_by_depth_up(int* depths, float* points, int* indices, int* neighbors, int* permutation, int n_dim, int k, int stride, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int s = tid & -stride;
    int l = tid + s;
    int r = l ^ (2 * stride - 1);
    if (r < n) compare_swap_by_depth(depths, points, indices, neighbors, permutation, n_dim, k, l, r);
}

__global__ void sort_by_depth_down(int* depths, float* points, int* indices, int* neighbors, int* permutation, int n_dim, int k, int stride, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int s = tid & -stride;
    int l = tid + s;
    int r = l + stride;
    if (r < n) compare_swap_by_depth(depths, points, indices, neighbors, permutation, n_dim, k, l, r);
}

__host__ void sort_by_depth(
    cudaStream_t stream,
    float* points,
    int* indices,
    int* neighbors,
    int* depths,
    int* permutation,
    int n_dim,
    int k,
    int n // typically (n_points - n0) and the arrays above are offset
) {
    for (int u = 1; u <= n; u *= 2) {
        CUDA_LAUNCH(sort_by_depth_up, n, stream, depths, points, indices, neighbors, permutation, n_dim, k, u);
        for (int d = u/2; d > 0; d /= 2) {
            CUDA_LAUNCH(sort_by_depth_down, n, stream, depths, points, indices, neighbors, permutation, n_dim, k, d);
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
    for (int u = 1; u < n; u += u) {
        CUDA_LAUNCH(sort_up<T>, n, stream, keys, u);
        for (int d = u/2; d > 0; d /= 2) {
            CUDA_LAUNCH(sort_down<T>, n, stream, keys, d);
        }
    }
}