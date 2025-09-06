// covariance.h
#pragma once

#include <cmath>
#include <cuda_runtime.h>

#include "common.h"

__forceinline__ __device__ float cov_lookup(
    float r,
    const float *cov_bins,
    const float *cov_vals,
    int n_cov
) {
    int idx = searchsorted(cov_bins, r, n_cov);
    if (idx == 0) return cov_vals[0]; // not ideal, should have cov_bins[0] == 0.0f
    if (idx == n_cov) return cov_vals[n_cov - 1]; // outside bounds, return last value (should probably set to 0.0f)

    // inside bounds, interpolate
    float r0 = cov_bins[idx - 1];
    float r1 = cov_bins[idx];
    float c0 = cov_vals[idx - 1];
    float c1 = cov_vals[idx];
    if (r0 == r1) return c0; // avoid division by zero in case bins are somehow the same
    return c0 + (c1 - c0) * (r - r0) / (r1 - r0);
}

__forceinline__ __device__ void cov_lookup_matrix(
    const float *points, // (n, d)
    const float *cov_bins, // (R,)
    const float *cov_vals, // (R,)
    float *out, // (n, n) lower triangular so actually n * (n + 1) / 2 entries
    int n_points,
    int n_dim,
    int n_cov
) {
    for (int i = 0; i < n_points; ++i) {
        for (int j = 0; j <= i; ++j) {
            float r = compute_distance(points + (i * n_dim), points + (j * n_dim), n_dim);
            out[tri(i, j)] = cov_lookup(r, cov_bins, cov_vals, n_cov);
        }
    }
}

// atomic write cov tangent to appropriate bin
__forceinline__ __device__ void cov_lookup_vjp(
    float r,
    float v,
    const float *cov_bins,
    float *cov_vals_tangent,
    int n_cov
) {
    int idx = searchsorted(cov_bins, r, n_cov);

    if (idx == 0) {
        atomicAdd(cov_vals_tangent + 0, v);
        return;
    }
    if (idx == n_cov) {
        atomicAdd(cov_vals_tangent + n_cov - 1, v);
        return;
    }

    // inside bounds, interpolate
    float r0 = cov_bins[idx - 1];
    float r1 = cov_bins[idx];
    if (r0 == r1) {
        atomicAdd(cov_vals_tangent + idx - 1, v);
        return;
    }
    atomicAdd(cov_vals_tangent + idx - 1, v * (r1 - r) / (r1 - r0));
    atomicAdd(cov_vals_tangent + idx, v * (r - r0) / (r1 - r0));
}

__forceinline__ __device__ void cov_lookup_matrix_vjp(
    const float *points, // (n, d)
    const float *dA, // (n, n) lower triangular so actually n * (n + 1) / 2 entries
    const float *cov_bins, // (R,)
    float *cov_vals_tangent, // (R,)
    int n_points,
    int n_dim,
    int n_cov
) {
    for (int i = 0; i < n_points; ++i) {
        for (int j = 0; j <= i; ++j) {
            float r = compute_distance(points + (i * n_dim), points + (j * n_dim), n_dim);
            float sym = (i == j) ? 1.0f : 2.0f; // off diagonal must be added twice
            cov_lookup_vjp(r, sym * dA[tri(i, j)], cov_bins, cov_vals_tangent, n_cov);
        }
    }
}