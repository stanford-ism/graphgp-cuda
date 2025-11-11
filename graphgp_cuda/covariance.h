// covariance.h
#pragma once

#include <cmath>
#include <cuda_runtime.h>

#include "common.h"

template <typename f_t>
__forceinline__ __device__ f_t cov_lookup(
    f_t r,
    const f_t *cov_bins,
    const f_t *cov_vals,
    size_t n_cov
) {
    size_t idx = searchsorted(cov_bins, r, n_cov);
    if (idx == 0) return cov_vals[0]; // not ideal, should have cov_bins[0] == 0.0f
    if (idx == n_cov) return cov_vals[n_cov - 1]; // outside bounds, return last value (should probably set to 0.0f)

    // inside bounds, interpolate
    f_t r0 = cov_bins[idx - 1];
    f_t r1 = cov_bins[idx];
    f_t c0 = cov_vals[idx - 1];
    f_t c1 = cov_vals[idx];
    if (r0 == r1) return c0; // avoid division by zero in case bins are somehow the same
    return c0 + (c1 - c0) * (r - r0) / (r1 - r0);
}

template <typename f_t>
__forceinline__ __device__ void cov_lookup_matrix(
    const f_t *points, // (n, d)
    const f_t *cov_bins, // (R,)
    const f_t *cov_vals, // (R,)
    f_t *out, // (n, n) lower triangular so actually n * (n + 1) / 2 entries
    size_t n_points,
    size_t n_dim,
    size_t n_cov
) {
    for (size_t i = 0; i < n_points; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            f_t r = sqrt(compute_square_distance(points + (i * n_dim), points + (j * n_dim), n_dim));
            out[tri(i, j)] = cov_lookup(r, cov_bins, cov_vals, n_cov);
        }
    }
}

// atomic write cov tangent to appropriate bin
template <typename f_t>
__forceinline__ __device__ void cov_lookup_vjp(
    f_t r,
    f_t v,
    const f_t *cov_bins,
    f_t *cov_vals_tangent,
    size_t n_cov
) {
    size_t idx = searchsorted(cov_bins, r, n_cov);

    if (idx == 0) {
        atomicAdd(cov_vals_tangent + 0, v);
        return;
    }
    if (idx == n_cov) {
        atomicAdd(cov_vals_tangent + n_cov - 1, v);
        return;
    }

    // inside bounds, interpolate
    f_t r0 = cov_bins[idx - 1];
    f_t r1 = cov_bins[idx];
    if (r0 == r1) {
        atomicAdd(cov_vals_tangent + idx - 1, v);
        return;
    }
    atomicAdd(cov_vals_tangent + idx - 1, v * (r1 - r) / (r1 - r0));
    atomicAdd(cov_vals_tangent + idx, v * (r - r0) / (r1 - r0));
}

template <typename f_t>
__forceinline__ __device__ void cov_lookup_matrix_vjp(
    const f_t *points, // (n, d)
    const f_t *dA, // (n, n) lower triangular so actually n * (n + 1) / 2 entries
    const f_t *cov_bins, // (R,)
    f_t *cov_vals_tangent, // (R,)
    size_t n_points,
    size_t n_dim,
    size_t n_cov
) {
    for (size_t i = 0; i < n_points; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            f_t r = sqrt(compute_square_distance(points + (i * n_dim), points + (j * n_dim), n_dim));
            f_t sym = (i == j) ? f_t(1.0) : f_t(2.0); // off diagonal must be added twice
            cov_lookup_vjp(r, sym * dA[tri(i, j)], cov_bins, cov_vals_tangent, n_cov);
        }
    }
}