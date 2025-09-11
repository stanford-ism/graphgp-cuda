// ======================================================================== //
// Copyright 2022-2022 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

// Rewritten to be easier to modify. Doesn't seem to be much slower.

// sort.h
#pragma once

#include <cuda_runtime.h>
#include "common.h"
#include "cubit/cubit.h"

// EXTREMELY SIMPLISTIC SORT
// Uses logic from https://github.com/ingowald/cudaBitonic/blob/master/cubit/cubit.h
// Should certainly replace with this eventually, just want something simple I can easily modify

template<typename T>
inline __device__ T max_value();
template<>
__forceinline__ __device__ uint32_t max_value() { return UINT_MAX; };
template<>
__forceinline__ __device__ int32_t max_value() { return INT_MAX; };
template<>
__forceinline__ __device__ uint64_t max_value() { return ULONG_MAX; };
template<>
__forceinline__ __device__ float max_value() { return INFINITY; };
template<>
__forceinline__ __device__ double max_value() { return INFINITY; };


// ================= REFERENCE NON-POWER-OF-TWO BITONIC SORT ===================

template <typename T>
__forceinline__ __device__ void shared_swap(T* keys, int a, int b) {
    T ka = keys[a];
    T kb = keys[b];
    bool swap = (ka > kb);
    keys[a] = swap ? kb : ka;
    keys[b] = swap ? ka : kb;
}

template <typename T1, typename T2>
__forceinline__ __device__ void shared_swap(T1* keys1, T2* keys2, int a, int b) {
    T1 k1a = keys1[a];
    T1 k1b = keys1[b];
    T2 k2a = keys2[a];
    T2 k2b = keys2[b];
    bool swap = (k1a > k1b) || (k1a == k1b && k2a > k2b);
    keys1[a] = swap ? k1b : k1a;
    keys1[b] = swap ? k1a : k1b;
    keys2[a] = swap ? k2b : k2a;
    keys2[b] = swap ? k2a : k2b;
}

template <typename T>
__forceinline__ __device__ void global_swap(T* keys, int a, int b) {
    T k1 = keys[a];
    T k2 = keys[b];
    if (k1 > k2) {
        keys[a] = k2;
        keys[b] = k1;
    }
}

template <typename T1, typename T2>
__forceinline__ __device__ void global_swap(T1* keys1, T2* keys2, int a, int b) {
    T1 k1a = keys1[a];
    T1 k1b = keys1[b];
    T2 k2a = keys2[a];
    T2 k2b = keys2[b];
    if ((k1a > k1b) || (k1a == k1b && k2a > k2b)) {
        keys1[a] = k1b;
        keys1[b] = k1a;
        keys2[a] = k2b;
        keys2[b] = k2a;
    }
}

template <typename T>
__global__ void global_sort_up(T* keys, int u, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int s = tid & -u;
    int l = tid + s;
    int r = l ^ (2 * u - 1);
    if (r < n) global_swap(keys, l, r);
}

template <typename T>
__global__ void global_sort_down(T* keys, int d, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int s = tid & -d;
    int l = tid + s;
    int r = l + d;
    if (r < n) global_swap(keys, l, r);
}

template <typename T1, typename T2>
__global__ void global_sort_up(T1* keys1, T2* keys2, int u, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int s = tid & -u;
    int l = tid + s;
    int r = l ^ (2 * u - 1);
    if (r < n) global_swap(keys1, keys2, l, r);
}

template <typename T1, typename T2>
__global__ void global_sort_down(T1* keys1, T2* keys2, int d, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int s = tid & -d;
    int l = tid + s;
    int r = l + d;
    if (r < n) global_swap(keys1, keys2, l, r);
}

template <typename T, int BLOCK_THREADS, bool UP>
__global__ void block_sort(T* g_keys, int n) {
    int i = threadIdx.x;
    const int n_shared = 2 * BLOCK_THREADS; // must be power of two
    int idx = i + blockIdx.x * n_shared;

    // build power of two array of keys
    __shared__ float keys[n_shared];
    if (idx < n) {
        keys[i] = g_keys[idx];
    }
    else {
        keys[i] = INFINITY;
    }
    if (idx + BLOCK_THREADS < n) {
        keys[i + BLOCK_THREADS] = g_keys[idx + BLOCK_THREADS];
    }
    else {
        keys[i + BLOCK_THREADS] = INFINITY;
    }
    __syncthreads();

    // bitonic sorting network, stride at most BLOCK_THREADS to cover n_shared values
    int s, l, r;
    if constexpr (UP) {
        for (int u = 1; u <= BLOCK_THREADS; u += u) {
            s = i & -u; l = i + s; r = l ^ (2 * u - 1);
            shared_swap(keys, l, r);
            __syncthreads();
            for (int d = u/2; d >= 1; d /= 2) {
                s = i & -d; l = i + s; r = l + d;
                shared_swap(keys, l, r);
                __syncthreads();
            }
        }
    } else {
        for (int d = BLOCK_THREADS; d >= 1; d /= 2) {
            s = i & -d; l = i + s; r = l + d;
            shared_swap(keys, l, r);
            __syncthreads();
        }
    }

    // write back results
    if (idx < n) {
        g_keys[idx] = keys[i];
    }
    if (idx + BLOCK_THREADS < n) {
        g_keys[idx + BLOCK_THREADS] = keys[i + BLOCK_THREADS];
    }
}


template <typename T1, typename T2, int BLOCK_THREADS, bool UP>
__global__ void block_sort(T1* g_keys1, T2* g_keys2, int n) {
    int i = threadIdx.x;
    const int n_shared = 2 * BLOCK_THREADS; // must be power of two
    int idx = i + blockIdx.x * n_shared;

    // build power of two array of keys
    __shared__ float keys1[n_shared];
    __shared__ float keys2[n_shared];
    if (idx < n) {
        keys1[i] = g_keys1[idx];
        keys2[i] = g_keys2[idx];
    }
    else {
        keys1[i] = INFINITY;
        keys2[i] = INFINITY;
    }
    if (idx + BLOCK_THREADS < n) {
        keys1[i + BLOCK_THREADS] = g_keys1[idx + BLOCK_THREADS];
        keys2[i + BLOCK_THREADS] = g_keys2[idx + BLOCK_THREADS];
    }
    else {
        keys1[i + BLOCK_THREADS] = INFINITY;
        keys2[i + BLOCK_THREADS] = INFINITY;
    }
    __syncthreads();

    // bitonic sorting network, stride at most BLOCK_THREADS to cover n_shared values
    int s, l, r;
    if constexpr (UP) {
        for (int u = 1; u <= BLOCK_THREADS; u += u) {
            s = i & -u; l = i + s; r = l ^ (2 * u - 1);
            shared_swap(keys1, keys2, l, r);
            __syncthreads();
            for (int d = u/2; d >= 1; d /= 2) {
                s = i & -d; l = i + s; r = l + d;
                shared_swap(keys1, keys2, l, r);
                __syncthreads();
            }
        }
    } else {
        for (int d = BLOCK_THREADS; d >= 1; d /= 2) {
            s = i & -d; l = i + s; r = l + d;
            shared_swap(keys1, keys2, l, r);
            __syncthreads();
        }
    }

    // write back results
    if (idx < n) {
        g_keys1[idx] = keys1[i];
        g_keys2[idx] = keys2[i];
    }
    if (idx + BLOCK_THREADS < n) {
        g_keys1[idx + BLOCK_THREADS] = keys1[i + BLOCK_THREADS];
        g_keys2[idx + BLOCK_THREADS] = keys2[i + BLOCK_THREADS];
    }
}

template <typename T>
__host__ void sort(cudaStream_t stream, T* keys, int n) {

    const int block_threads = 1024; // must be power of two
    const int n_shared = 2 * block_threads;
    int n_blocks = (n + n_shared - 1) / n_shared;

    // block sort handles maximum stride of block_threads, or n_shared / 2
    block_sort<T, block_threads, true><<<n_blocks, block_threads, 0, stream>>>(keys, n);
    for (int u = n_shared; u < n; u += u) {
        CUDA_LAUNCH(global_sort_up, n, stream, keys, u);
        for (int d = u/2; d >= n_shared; d /= 2) {
            CUDA_LAUNCH(global_sort_down, n, stream, keys, d);
        }
        block_sort<T, block_threads, false><<<n_blocks, block_threads, 0, stream>>>(keys, n);
    }

    // swap in cubit
    // cubit::sort(keys, n, stream);
}

template <typename T1, typename T2>
__host__ void sort(cudaStream_t stream, T1* keys1, T2* keys2, int n) {

    const int block_threads = 1024; // must be power of two
    const int n_shared = 2 * block_threads;
    int n_blocks = (n + n_shared - 1) / n_shared;

    // block sort handles maximum stride of block_threads, or n_shared / 2
    block_sort<T1, T2, block_threads, true><<<n_blocks, block_threads, 0, stream>>>(keys1, keys2, n);
    for (int u = n_shared; u < n; u += u) {
        CUDA_LAUNCH(global_sort_up, n, stream, keys1, keys2, u);
        for (int d = u/2; d >= n_shared; d /= 2) {
            CUDA_LAUNCH(global_sort_down, n, stream, keys1, keys2, d);
        }
        block_sort<T1, T2, block_threads, false><<<n_blocks, block_threads, 0, stream>>>(keys1, keys2, n);
    }
}








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


// // ========================= SORT BY DEPTH =====================

// __forceinline__ __device__ void compare_swap_by_depth(
//     int* depths,
//     float* points,
//     int* indices,
//     int* neighbors,
//     int* permutation,
//     int n_dim,
//     int k,
//     int a,
//     int b
// ) {
//     int da = depths[a];
//     int db = depths[b];
//     float pa;
//     float pb;

//     // compare by depths and bring other arrays along
//     if ((da > db) || ((da == db) && (permutation[a] > permutation[b]))) {
//         depths[a] = db;
//         depths[b] = da;

//         // swap points
//         for (int d = 0; d < n_dim; d++) {
//             pa = points[a * n_dim + d];
//             pb = points[b * n_dim + d];
//             points[a * n_dim + d] = pb;
//             points[b * n_dim + d] = pa;
//         }

//         // swap indices
//         da = indices[a];
//         db = indices[b];
//         indices[a] = db;
//         indices[b] = da;

//         // swap neighbors
//         for (int j = 0; j < k; ++j) {
//             int da = neighbors[a * k + j];
//             int db = neighbors[b * k + j];
//             neighbors[a * k + j] = db;
//             neighbors[b * k + j] = da;
//         }

//         // swap permutation
//         da = permutation[a];
//         db = permutation[b];
//         permutation[a] = db;
//         permutation[b] = da;
//     }
// }

// __global__ void sort_by_depth_up(int* depths, float* points, int* indices, int* neighbors, int* permutation, int n_dim, int k, int stride, int n) {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= n) return;
//     int s = tid & -stride;
//     int l = tid + s;
//     int r = l ^ (2 * stride - 1);
//     if (r < n) compare_swap_by_depth(depths, points, indices, neighbors, permutation, n_dim, k, l, r);
// }

// __global__ void sort_by_depth_down(int* depths, float* points, int* indices, int* neighbors, int* permutation, int n_dim, int k, int stride, int n) {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= n) return;
//     int s = tid & -stride;
//     int l = tid + s;
//     int r = l + stride;
//     if (r < n) compare_swap_by_depth(depths, points, indices, neighbors, permutation, n_dim, k, l, r);
// }

// __host__ void sort_by_depth(
//     cudaStream_t stream,
//     float* points,
//     int* indices,
//     int* neighbors,
//     int* depths,
//     int* permutation,
//     int n_dim,
//     int k,
//     int n // typically (n_points - n0) and the arrays above are offset
// ) {
//     for (int u = 1; u <= n; u *= 2) {
//         CUDA_LAUNCH(sort_by_depth_up, n, stream, depths, points, indices, neighbors, permutation, n_dim, k, u);
//         for (int d = u/2; d > 0; d /= 2) {
//             CUDA_LAUNCH(sort_by_depth_down, n, stream, depths, points, indices, neighbors, permutation, n_dim, k, d);
//         }
//     }
// }

// ================= 2 KEY SORT ===================

// template <typename Ta, typename Tb>
// __forceinline__ __device__ void shared_swap(Ta* keys_a, Tb* keys_b, int i, int j) {
//     Ta ka1 = keys_a[i];
//     Ta ka2 = keys_a[j];
//     Tb kb1 = keys_b[i];
//     Tb kb2 = keys_b[j];
//     bool swap = (ka1 > ka2) || (ka1 == ka2 && kb1 > kb2);
//     keys_a[i] = swap ? ka2 : ka1;
//     keys_a[j] = swap ? ka1 : ka2;
//     keys_b[i] = swap ? kb2 : kb1;
//     keys_b[j] = swap ? kb1 : kb2;
// }

// // template <typename Ta, typename Tb>
// // __forceinline__ __device__ void global_swap(Ta* keys_a, Tb* keys_b, int i, int j) {
// //     Ta ka1 = keys_a[i];
// //     Ta ka2 = keys_a[j];
// //     Tb kb1 = keys_b[i];
// //     Tb kb2 = keys_b[j];
// //     if ((ka1 > ka2) || (ka1 == ka2 && kb1 > kb2)) {
// //         keys_a[i] = ka2;
// //         keys_b[i] = kb2;
// //         keys_a[j] = ka1;
// //         keys_b[j] = kb1;
// //     }
// // }

// template <typename Ta, typename Tb>
// __global__ void global_sort_up(Ta* keys_a, Tb* keys_b, int u, int n) {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= n) return;
//     int s = tid & -u;
//     int l = tid + s;
//     int r = l ^ (2 * u - 1);
//     if (r < n) shared_swap(keys_a, keys_b, l, r);
// }

// template <typename Ta, typename Tb>
// __global__ void global_sort_down(Ta* keys_a, Tb* keys_b, int d, int n) {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= n) return;
//     int s = tid & -d;
//     int l = tid + s;
//     int r = l + d;
//     if (r < n) shared_swap(keys_a, keys_b, l, r);
// }

// template <typename Ta, typename Tb, int N_SHARED>
// __global__ void block_sort(Ta* keys_a, Tb* keys_b, int n) {
//     int i = threadIdx.x;
//     int block_threads = blockDim.x; // must be N_SHARED / 2
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
//     // build power of two array of keys
//     __shared__ Ta s_keys_a[N_SHARED];
//     __shared__ Tb s_keys_b[N_SHARED];
//     if (tid < n) {
//         s_keys_a[i] = keys_a[tid];
//         s_keys_b[i] = keys_b[tid];
//     }
//     else {
//         s_keys_a[i] = max_value<Ta>();
//         s_keys_b[i] = max_value<Tb>();
//     }
//     if (tid + block_threads < n) {
//         s_keys_a[i + block_threads] = keys_a[tid + block_threads];
//         s_keys_b[i + block_threads] = keys_b[tid + block_threads];
//     }
//     else {
//         s_keys_a[i + block_threads] = max_value<Ta>();
//         s_keys_b[i + block_threads] = max_value<Tb>();
//     }
//     __syncthreads();

//     // bitonic sorting network
//     int s, l, r;
//     for (int u = 1; u < N_SHARED; u += u) {
//         s = i & -u; l = i + s; r = l ^ (2 * u - 1);
//         shared_swap(s_keys_a, s_keys_b, l, r);
//         __syncthreads();
//         for (int d = u/2; d > 0; d /= 2) {
//             s = i & -d; l = i + s; r = l + d;
//             shared_swap(s_keys_a, s_keys_b, l, r);
//             __syncthreads();
//         }
//     }

//     // write back results
//     if (tid < n) {
//         keys_a[tid] = s_keys_a[i];
//         keys_b[tid] = s_keys_b[i];
//     }
//     if (tid + block_threads < n) {
//         keys_a[tid + block_threads] = s_keys_a[i + block_threads];
//         keys_b[tid + block_threads] = s_keys_b[i + block_threads];
//     }
// }

// template <typename Ta, typename Tb, int N_SHARED>
// __global__ void block_sort_down(Ta* keys_a, Tb* keys_b, int n) {
//     int i = threadIdx.x;
//     int block_threads = blockDim.x; // must be N_SHARED / 2
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
//     // build power of two array of keys
//     __shared__ Ta s_keys_a[N_SHARED];
//     __shared__ Tb s_keys_b[N_SHARED];
//     if (tid < n) {
//         s_keys_a[i] = keys_a[tid];
//         s_keys_b[i] = keys_b[tid];
//     }
//     else {
//         s_keys_a[i] = max_value<Ta>();
//         s_keys_b[i] = max_value<Tb>();
//     }
//     if (tid + block_threads < n) {
//         s_keys_a[i + block_threads] = keys_a[tid + block_threads];
//         s_keys_b[i + block_threads] = keys_b[tid + block_threads];
//     }
//     else {
//         s_keys_a[i + block_threads] = max_value<Ta>();
//         s_keys_b[i + block_threads] = max_value<Tb>();
//     }
//     __syncthreads();

//     // bitonic sorting network
//     int s, l, r;
//     for (int d = N_SHARED/2; d > 0; d /= 2) {
//         s = i & -d; l = i + s; r = l + d;
//         shared_swap(s_keys_a, s_keys_b, l, r);
//         __syncthreads();
//     }

//     // write back results
//     if (tid < n) {
//         keys_a[tid] = s_keys_a[i];
//         keys_b[tid] = s_keys_b[i];
//     }
//     if (tid + block_threads < n) {
//         keys_a[tid + block_threads] = s_keys_a[i + block_threads];
//         keys_b[tid + block_threads] = s_keys_b[i + block_threads];
//     }
// }

// template <typename Ta, typename Tb>
// __host__ void sort(cudaStream_t stream, Ta* keys_a, Tb* keys_b, int n) {
//     const int n_shared = 128; // must be power of two
//     const int n_threads_per_block = n_shared / 2;
//     int n_blocks = (n + n_threads_per_block - 1) / n_threads_per_block;

//     block_sort<Ta, Tb, n_shared><<<n_blocks, n_threads_per_block, 0, stream>>>(keys_a, keys_b, n);
//     for (int u = n_shared; u < n; u += u) {
//         CUDA_LAUNCH(global_sort_up, n, stream, keys_a, keys_b, u);
//         for (int d = u/2; d > n_shared/2; d /= 2) {
//             CUDA_LAUNCH(global_sort_down, n, stream, keys_a, keys_b, d);
//         }
//         block_sort_down<Ta, Tb, n_shared><<<n_blocks, n_threads_per_block, 0, stream>>>(keys_a, keys_b, n);
//     }
// }


// // ================= REFERENCE NON-POWER-OF-TWO BITONIC SORT ===================

// __forceinline__ __device__ void shared_swap(float* keys, int i, int j) {
//     float k1 = keys[i];
//     float k2 = keys[j];
//     bool swap = (k1 > k2);
//     keys[i] = swap ? k2 : k1;
//     keys[j] = swap ? k1 : k2;
// }

// __forceinline__ __device__ void global_swap(float* keys, int i, int j) {
//     float k1 = keys[i];
//     float k2 = keys[j];
//     if (k1 > k2) {
//         keys[i] = k2;
//         keys[j] = k1;
//     }
// }

// __global__ void global_sort_up(float* keys, int u, int n) {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= n) return;
//     int s = tid & -u;
//     int l = tid + s;
//     int r = l ^ (2 * u - 1);
//     if (r < n) global_swap(keys, l, r);
// }

// __global__ void global_sort_down(float* keys, int d, int n) {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= n) return;
//     int s = tid & -d;
//     int l = tid + s;
//     int r = l + d;
//     if (r < n) global_swap(keys, l, r);
// }


// template <int BLOCK_THREADS, bool UP>
// __global__ void block_sort(float* global_keys, int n) {
//     int i = threadIdx.x;
//     const int n_shared = 2 * BLOCK_THREADS; // must be power of two
//     int idx = i + blockIdx.x * n_shared;

//     // build power of two array of keys
//     __shared__ float keys[n_shared];
//     if (idx < n) keys[i] = global_keys[idx];
//     else keys[i] = INFINITY;
//     if (idx + BLOCK_THREADS < n) keys[i + BLOCK_THREADS] = global_keys[idx + BLOCK_THREADS];
//     else keys[i + BLOCK_THREADS] = INFINITY;
//     __syncthreads();

//     // bitonic sorting network, stride at most BLOCK_THREADS to cover n_shared values
//     int s, l, r;
//     if constexpr (UP) {
//         for (int u = 1; u <= BLOCK_THREADS; u += u) {
//             s = i & -u; l = i + s; r = l ^ (2 * u - 1);
//             shared_swap(keys, l, r);
//             __syncthreads();
//             for (int d = u/2; d >= 1; d /= 2) {
//                 s = i & -d; l = i + s; r = l + d;
//                 shared_swap(keys, l, r);
//                 __syncthreads();
//             }
//         }
//     } else {
//         for (int d = BLOCK_THREADS; d >= 1; d /= 2) {
//             s = i & -d; l = i + s; r = l + d;
//             shared_swap(keys, l, r);
//             __syncthreads();
//         }
//     }

//     // write back results
//     if (idx < n) global_keys[idx] = keys[i];
//     if (idx + BLOCK_THREADS < n) global_keys[idx + BLOCK_THREADS] = keys[i + BLOCK_THREADS];
// }

// __host__ void sort(cudaStream_t stream, float* keys, int n) {

//     const int block_threads = 1024; // must be power of two
//     const int n_shared = 2 * block_threads;
//     int n_blocks = (n + n_shared - 1) / n_shared;

//     // block sort handles maximum stride of block_threads, or n_shared / 2
//     block_sort<block_threads, true><<<n_blocks, block_threads, 0, stream>>>(keys, n);
//     for (int u = n_shared; u < n; u += u) {
//         CUDA_LAUNCH(global_sort_up, n, stream, keys, u);
//         for (int d = u/2; d >= n_shared; d /= 2) {
//             CUDA_LAUNCH(global_sort_down, n, stream, keys, d);
//         }
//         block_sort<block_threads, false><<<n_blocks, block_threads, 0, stream>>>(keys, n);
//     }

//     // swap in cubit
//     // cubit::sort(keys, n, stream);
// }