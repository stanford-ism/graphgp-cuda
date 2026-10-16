// covariance.h
#pragma once

#include <cmath>
#include <cuda_runtime.h>

#include "common.h"

// Isotropic Gaussian mixture covariance, C(r) = sum_m amps[m] * exp(-rates[m] * r^2), with rates = 1 / (2 scales^2)
// precomputed on the host. A zero scale has rate = inf and acts as a nugget: exp(-inf * r^2) = 0 for r > 0, and the
// r == 0 case is handled separately (every component contributes its full amplitude there).
template <typename f_t>
__forceinline__ __device__ f_t cov_mixture(
    f_t r2,
    const f_t *cov_rates, // (M,)
    const f_t *cov_amps, // (M,)
    size_t n_scales
) {
    f_t cov = f_t(0);
    if (r2 == f_t(0)) {
        for (size_t m = 0; m < n_scales; ++m) cov += cov_amps[m];
    } else {
        for (size_t m = 0; m < n_scales; ++m) cov += cov_amps[m] * exp(-cov_rates[m] * r2);
    }
    return cov;
}

template <typename f_t>
__forceinline__ __device__ void cov_matrix(
    const f_t *points, // (n, d)
    const f_t *cov_rates, // (M,)
    const f_t *cov_amps, // (M,)
    f_t *out, // (n, n) lower triangular so actually n * (n + 1) / 2 entries
    size_t n_points,
    size_t n_dim,
    size_t n_scales
) {
    for (size_t i = 0; i < n_points; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            f_t r2 = compute_square_distance(points + (i * n_dim), points + (j * n_dim), n_dim);
            out[tri(i, j)] = cov_mixture(r2, cov_rates, cov_amps, n_scales);
        }
    }
}

// Accumulate the amplitude cotangents, cov_amps_tangent[m] += sum_ij dA_ij * exp(-rates[m] * r_ij^2), counting the
// off-diagonal entries twice. Loops over components outermost so each thread needs a single accumulator and issues
// one atomic per component. The squared distances are cached in scratch (n * (n + 1) / 2 entries).
template <typename f_t>
__forceinline__ __device__ void cov_matrix_vjp(
    const f_t *points, // (n, d)
    const f_t *dA, // (n, n) lower triangular so actually n * (n + 1) / 2 entries
    const f_t *cov_rates, // (M,)
    f_t *cov_amps_tangent, // (M,)
    f_t *scratch, // (n, n) lower triangular
    size_t n_points,
    size_t n_dim,
    size_t n_scales
) {
    for (size_t i = 0; i < n_points; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            scratch[tri(i, j)] = compute_square_distance(points + (i * n_dim), points + (j * n_dim), n_dim);
        }
    }
    for (size_t m = 0; m < n_scales; ++m) {
        f_t rate = cov_rates[m];
        f_t acc = f_t(0);
        for (size_t i = 0; i < n_points; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                f_t r2 = scratch[tri(i, j)];
                f_t gaussian = (r2 == f_t(0)) ? f_t(1) : exp(-rate * r2);
                f_t sym = (i == j) ? f_t(1) : f_t(2); // off diagonal must be added twice
                acc += sym * dA[tri(i, j)] * gaussian;
            }
        }
        atomicAdd(cov_amps_tangent + m, acc);
    }
}
