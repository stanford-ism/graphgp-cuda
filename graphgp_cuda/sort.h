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

// Rewritten to include up to 4-key sort and be easier to modify.
// Doesn't seem to be substantially slower than inlining everything and without syncthreads below 32.

// sort.h
#pragma once

#include <cuda_runtime.h>
#include "common.h"
#include "cubit/cubit.h"

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

template <typename T1, typename T2, typename T3>
__forceinline__ __device__ void shared_swap(T1* keys1, T2* keys2, T3* keys3, int a, int b) {
    T1 k1a = keys1[a];
    T1 k1b = keys1[b];
    T2 k2a = keys2[a];
    T2 k2b = keys2[b];
    T3 k3a = keys3[a];
    T3 k3b = keys3[b];
    bool swap = (k1a > k1b) || (k1a == k1b && k2a > k2b) || (k1a == k1b && k2a == k2b && k3a > k3b);
    keys1[a] = swap ? k1b : k1a;
    keys1[b] = swap ? k1a : k1b;
    keys2[a] = swap ? k2b : k2a;
    keys2[b] = swap ? k2a : k2b;
    keys3[a] = swap ? k3b : k3a;
    keys3[b] = swap ? k3a : k3b;
}

template <typename T1, typename T2, typename T3, typename T4>
__forceinline__ __device__ void shared_swap(T1* keys1, T2* keys2, T3* keys3, T4* keys4, int a, int b) {
    T1 k1a = keys1[a];
    T1 k1b = keys1[b];
    T2 k2a = keys2[a];
    T2 k2b = keys2[b];
    T3 k3a = keys3[a];
    T3 k3b = keys3[b];
    T4 k4a = keys4[a];
    T4 k4b = keys4[b];
    bool swap = (k1a > k1b) || (k1a == k1b && k2a > k2b) || (k1a == k1b && k2a == k2b && k3a > k3b) || (k1a == k1b && k2a == k2b && k3a == k3b && k4a > k4b);
    keys1[a] = swap ? k1b : k1a;
    keys1[b] = swap ? k1a : k1b;
    keys2[a] = swap ? k2b : k2a;
    keys2[b] = swap ? k2a : k2b;
    keys3[a] = swap ? k3b : k3a;
    keys3[b] = swap ? k3a : k3b;
    keys4[a] = swap ? k4b : k4a;
    keys4[b] = swap ? k4a : k4b;
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

template <typename T1, typename T2, typename T3>
__forceinline__ __device__ void global_swap(T1* keys1, T2* keys2, T3* keys3, int a, int b) {
    T1 k1a = keys1[a];
    T1 k1b = keys1[b];
    T2 k2a = keys2[a];
    T2 k2b = keys2[b];
    T3 k3a = keys3[a];
    T3 k3b = keys3[b];
    if ((k1a > k1b) || (k1a == k1b && k2a > k2b) || (k1a == k1b && k2a == k2b && k3a > k3b)) {
        keys1[a] = k1b;
        keys1[b] = k1a;
        keys2[a] = k2b;
        keys2[b] = k2a;
        keys3[a] = k3b;
        keys3[b] = k3a;
    }
}

template <typename T1, typename T2, typename T3, typename T4>
__forceinline__ __device__ void global_swap(T1* keys1, T2* keys2, T3* keys3, T4* keys4, int a, int b) {
    T1 k1a = keys1[a];
    T1 k1b = keys1[b];
    T2 k2a = keys2[a];
    T2 k2b = keys2[b];
    T3 k3a = keys3[a];
    T3 k3b = keys3[b];
    T4 k4a = keys4[a];
    T4 k4b = keys4[b];
    if ((k1a > k1b) || (k1a == k1b && k2a > k2b) || (k1a == k1b && k2a == k2b && k3a > k3b) || (k1a == k1b && k2a == k2b && k3a == k3b && k4a > k4b)) {
        keys1[a] = k1b;
        keys1[b] = k1a;
        keys2[a] = k2b;
        keys2[b] = k2a;
        keys3[a] = k3b;
        keys3[b] = k3a;
        keys4[a] = k4b;
        keys4[b] = k4a;
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

template <typename T1, typename T2>
__global__ void global_sort_up(T1* keys1, T2* keys2, int u, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int s = tid & -u;
    int l = tid + s;
    int r = l ^ (2 * u - 1);
    if (r < n) global_swap(keys1, keys2, l, r);
}

template <typename T1, typename T2, typename T3>
__global__ void global_sort_up(T1* keys1, T2* keys2, T3* keys3, int u, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int s = tid & -u;
    int l = tid + s;
    int r = l ^ (2 * u - 1);
    if (r < n) global_swap(keys1, keys2, keys3, l, r);
}

template <typename T1, typename T2, typename T3, typename T4>
__global__ void global_sort_up(T1* keys1, T2* keys2, T3* keys3, T4* keys4, int u, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int s = tid & -u;
    int l = tid + s;
    int r = l ^ (2 * u - 1);
    if (r < n) global_swap(keys1, keys2, keys3, keys4, l, r);
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
__global__ void global_sort_down(T1* keys1, T2* keys2, int d, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int s = tid & -d;
    int l = tid + s;
    int r = l + d;
    if (r < n) global_swap(keys1, keys2, l, r);
}

template <typename T1, typename T2, typename T3>
__global__ void global_sort_down(T1* keys1, T2* keys2, T3* keys3, int d, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int s = tid & -d;
    int l = tid + s;
    int r = l + d;
    if (r < n) global_swap(keys1, keys2, keys3, l, r);
}

template <typename T1, typename T2, typename T3, typename T4>
__global__ void global_sort_down(T1* keys1, T2* keys2, T3* keys3, T4* keys4, int d, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int s = tid & -d;
    int l = tid + s;
    int r = l + d;
    if (r < n) global_swap(keys1, keys2, keys3, keys4, l, r);
}

template <typename T, int BLOCK_THREADS, bool UP>
__global__ void block_sort(T* g_keys, int n) {
    int i = threadIdx.x;
    const int n_shared = 2 * BLOCK_THREADS; // must be power of two
    int idx = i + blockIdx.x * n_shared;

    // build power of two array of keys
    __shared__ T keys[n_shared];
    if (idx < n) {
        keys[i] = g_keys[idx];
    }
    else {
        keys[i] = max_value<T>();
    }
    if (idx + BLOCK_THREADS < n) {
        keys[i + BLOCK_THREADS] = g_keys[idx + BLOCK_THREADS];
    }
    else {
        keys[i + BLOCK_THREADS] = max_value<T>();
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
    __shared__ T1 keys1[n_shared];
    __shared__ T2 keys2[n_shared];
    if (idx < n) {
        keys1[i] = g_keys1[idx];
        keys2[i] = g_keys2[idx];
    }
    else {
        keys1[i] = max_value<T1>();
        keys2[i] = max_value<T2>();
    }
    if (idx + BLOCK_THREADS < n) {
        keys1[i + BLOCK_THREADS] = g_keys1[idx + BLOCK_THREADS];
        keys2[i + BLOCK_THREADS] = g_keys2[idx + BLOCK_THREADS];
    }
    else {
        keys1[i + BLOCK_THREADS] = max_value<T1>();
        keys2[i + BLOCK_THREADS] = max_value<T2>();
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

template <typename T1, typename T2, typename T3, int BLOCK_THREADS, bool UP>
__global__ void block_sort(T1* g_keys1, T2* g_keys2, T3* g_keys3, int n) {
    int i = threadIdx.x;
    const int n_shared = 2 * BLOCK_THREADS; // must be power of two
    int idx = i + blockIdx.x * n_shared;

    // build power of two array of keys
    __shared__ T1 keys1[n_shared];
    __shared__ T2 keys2[n_shared];
    __shared__ T3 keys3[n_shared];
    if (idx < n) {
        keys1[i] = g_keys1[idx];
        keys2[i] = g_keys2[idx];
        keys3[i] = g_keys3[idx];
    }
    else {
        keys1[i] = max_value<T1>();
        keys2[i] = max_value<T2>();
        keys3[i] = max_value<T3>();
    }
    if (idx + BLOCK_THREADS < n) {
        keys1[i + BLOCK_THREADS] = g_keys1[idx + BLOCK_THREADS];
        keys2[i + BLOCK_THREADS] = g_keys2[idx + BLOCK_THREADS];
        keys3[i + BLOCK_THREADS] = g_keys3[idx + BLOCK_THREADS];
    }
    else {
        keys1[i + BLOCK_THREADS] = max_value<T1>();
        keys2[i + BLOCK_THREADS] = max_value<T2>();
        keys3[i + BLOCK_THREADS] = max_value<T3>();
    }
    __syncthreads();

    // bitonic sorting network, stride at most BLOCK_THREADS to cover n_shared values
    int s, l, r;
    if constexpr (UP) {
        for (int u = 1; u <= BLOCK_THREADS; u += u) {
            s = i & -u; l = i + s; r = l ^ (2 * u - 1);
            shared_swap(keys1, keys2, keys3, l, r);
            __syncthreads();
            for (int d = u/2; d >= 1; d /= 2) {
                s = i & -d; l = i + s; r = l + d;
                shared_swap(keys1, keys2, keys3, l, r);
                __syncthreads();
            }
        }
    } else {
        for (int d = BLOCK_THREADS; d >= 1; d /= 2) {
            s = i & -d; l = i + s; r = l + d;
            shared_swap(keys1, keys2, keys3, l, r);
            __syncthreads();
        }
    }

    // write back results
    if (idx < n) {
        g_keys1[idx] = keys1[i];
        g_keys2[idx] = keys2[i];
        g_keys3[idx] = keys3[i];
    }
    if (idx + BLOCK_THREADS < n) {
        g_keys1[idx + BLOCK_THREADS] = keys1[i + BLOCK_THREADS];
        g_keys2[idx + BLOCK_THREADS] = keys2[i + BLOCK_THREADS];
        g_keys3[idx + BLOCK_THREADS] = keys3[i + BLOCK_THREADS];
    }
}

template <typename T1, typename T2, typename T3, typename T4, int BLOCK_THREADS, bool UP>
__global__ void block_sort(T1* g_keys1, T2* g_keys2, T3* g_keys3, T4* g_keys4, int n) {
    int i = threadIdx.x;
    const int n_shared = 2 * BLOCK_THREADS; // must be power of two
    int idx = i + blockIdx.x * n_shared;

    // build power of two array of keys
    __shared__ T1 keys1[n_shared];
    __shared__ T2 keys2[n_shared];
    __shared__ T3 keys3[n_shared];
    __shared__ T4 keys4[n_shared];
    if (idx < n) {
        keys1[i] = g_keys1[idx];
        keys2[i] = g_keys2[idx];
        keys3[i] = g_keys3[idx];
        keys4[i] = g_keys4[idx];
    }
    else {
        keys1[i] = max_value<T1>();
        keys2[i] = max_value<T2>();
        keys3[i] = max_value<T3>();
        keys4[i] = max_value<T4>();
    }
    if (idx + BLOCK_THREADS < n) {
        keys1[i + BLOCK_THREADS] = g_keys1[idx + BLOCK_THREADS];
        keys2[i + BLOCK_THREADS] = g_keys2[idx + BLOCK_THREADS];
        keys3[i + BLOCK_THREADS] = g_keys3[idx + BLOCK_THREADS];
        keys4[i + BLOCK_THREADS] = g_keys4[idx + BLOCK_THREADS];
    }
    else {
        keys1[i + BLOCK_THREADS] = max_value<T1>();
        keys2[i + BLOCK_THREADS] = max_value<T2>();
        keys3[i + BLOCK_THREADS] = max_value<T3>();
        keys4[i + BLOCK_THREADS] = max_value<T4>();
    }
    __syncthreads();

    // bitonic sorting network, stride at most BLOCK_THREADS to cover n_shared values
    int s, l, r;
    if constexpr (UP) {
        for (int u = 1; u <= BLOCK_THREADS; u += u) {
            s = i & -u; l = i + s; r = l ^ (2 * u - 1);
            shared_swap(keys1, keys2, keys3, keys4, l, r);
            __syncthreads();
            for (int d = u/2; d >= 1; d /= 2) {
                s = i & -d; l = i + s; r = l + d;
                shared_swap(keys1, keys2, keys3, keys4, l, r);
                __syncthreads();
            }
        }
    } else {
        for (int d = BLOCK_THREADS; d >= 1; d /= 2) {
            s = i & -d; l = i + s; r = l + d;
            shared_swap(keys1, keys2, keys3, keys4, l, r);
            __syncthreads();
        }
    }

    // write back results
    if (idx < n) {
        g_keys1[idx] = keys1[i];
        g_keys2[idx] = keys2[i];
        g_keys3[idx] = keys3[i];
        g_keys4[idx] = keys4[i];
    }
    if (idx + BLOCK_THREADS < n) {
        g_keys1[idx + BLOCK_THREADS] = keys1[i + BLOCK_THREADS];
        g_keys2[idx + BLOCK_THREADS] = keys2[i + BLOCK_THREADS];
        g_keys3[idx + BLOCK_THREADS] = keys3[i + BLOCK_THREADS];
        g_keys4[idx + BLOCK_THREADS] = keys4[i + BLOCK_THREADS];
    }
}

template <typename T>
__host__ void sort(T* keys, int n, cudaStream_t stream) {

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
}

template <typename T1, typename T2>
__host__ void sort(T1* keys1, T2* keys2, int n, cudaStream_t stream) {

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

template <typename T1, typename T2, typename T3>
__host__ void sort(T1* keys1, T2* keys2, T3* keys3, int n, cudaStream_t stream) {

    const int block_threads = 1024; // must be power of two
    const int n_shared = 2 * block_threads;
    int n_blocks = (n + n_shared - 1) / n_shared;

    // block sort handles maximum stride of block_threads, or n_shared / 2
    block_sort<T1, T2, T3, block_threads, true><<<n_blocks, block_threads, 0, stream>>>(keys1, keys2, keys3, n);
    for (int u = n_shared; u < n; u += u) {
        CUDA_LAUNCH(global_sort_up, n, stream, keys1, keys2, keys3, u);
        for (int d = u/2; d >= n_shared; d /= 2) {
            CUDA_LAUNCH(global_sort_down, n, stream, keys1, keys2, keys3, d);
        }
        block_sort<T1, T2, T3, block_threads, false><<<n_blocks, block_threads, 0, stream>>>(keys1, keys2, keys3, n);
    }
}

template <typename T1, typename T2, typename T3, typename T4>
__host__ void sort(T1* keys1, T2* keys2, T3* keys3, T4* keys4, int n, cudaStream_t stream) {

    const int block_threads = 1024; // must be power of two
    const int n_shared = 2 * block_threads;
    int n_blocks = (n + n_shared - 1) / n_shared;

    // block sort handles maximum stride of block_threads, or n_shared / 2
    block_sort<T1, T2, T3, T4, block_threads, true><<<n_blocks, block_threads, 0, stream>>>(keys1, keys2, keys3, keys4, n);
    for (int u = n_shared; u < n; u += u) {
        CUDA_LAUNCH(global_sort_up, n, stream, keys1, keys2, keys3, keys4, u);
        for (int d = u/2; d >= n_shared; d /= 2) {
            CUDA_LAUNCH(global_sort_down, n, stream, keys1, keys2, keys3, keys4, d);
        }
        block_sort<T1, T2, T3, T4, block_threads, false><<<n_blocks, block_threads, 0, stream>>>(keys1, keys2, keys3, keys4, n);
    }
}
