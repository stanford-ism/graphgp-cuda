// bindings.cu
#pragma once

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <cuda_runtime.h>
#include <cstdint>

#include "refine.h"
#include "refine_transpose.h"
#include "neighbors.h"

namespace ffi = xla::ffi;

ffi::Error refine_xla_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> points, // (N, d) in topological order
    ffi::Buffer<ffi::U32> offsets, // (L,) first entry is number of initial points
    ffi::Buffer<ffi::U32> neighbors, // (N, k) in original order, initial points not used
    ffi::Buffer<ffi::F32> cov_bins, // (R,)
    ffi::Buffer<ffi::F32> cov_vals, // (R,)
    ffi::Buffer<ffi::F32> initial_values, // (N0,)
    ffi::Buffer<ffi::F32> xi, // (N,) the first N0 are not used
    ffi::ResultBuffer<ffi::F32> values // (N,)
) {
    size_t n_points = points.dimensions()[0];
    size_t n_dim = points.dimensions()[1];
    size_t n_levels = offsets.dimensions()[0];
    size_t n_cov = cov_bins.dimensions()[0];
    size_t k = neighbors.dimensions()[1];

    if ((k < 2) || (k > 8)) return ffi::Error::InvalidArgument("only compiled for 2 <= k <= 8");
    if (n_dim > 3) return ffi::Error::InvalidArgument("only compiled for 2 <= n_dim <= 3");

    using func_type = void(*)(cudaStream_t, const float*, const uint32_t*, const uint32_t*, const float*, const float*, const float*, const float*, float*, size_t, size_t, size_t);
    func_type dispatch = nullptr;
    if (n_dim == 2) {
        if      (k == 2) dispatch = refine<2, 2>;
        else if (k == 3) dispatch = refine<3, 2>;
        else if (k == 4) dispatch = refine<4, 2>;
        else if (k == 5) dispatch = refine<5, 2>;
        else if (k == 6) dispatch = refine<6, 2>;
        else if (k == 7) dispatch = refine<7, 2>;
        else if (k == 8) dispatch = refine<8, 2>;
    } else if (n_dim == 3) {
        if      (k == 2) dispatch = refine<2, 3>;
        else if (k == 3) dispatch = refine<3, 3>;
        else if (k == 4) dispatch = refine<4, 3>;
        else if (k == 5) dispatch = refine<5, 3>;
        else if (k == 6) dispatch = refine<6, 3>;
        else if (k == 7) dispatch = refine<7, 3>;
        else if (k == 8) dispatch = refine<8, 3>;
    }

    dispatch(stream, points.typed_data(), offsets.typed_data(), neighbors.typed_data(), cov_bins.typed_data(), cov_vals.typed_data(), initial_values.typed_data(), xi.typed_data(), values->typed_data(), n_points, n_levels, n_cov);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    refine_xla, refine_xla_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>() // points
        .Arg<ffi::Buffer<ffi::U32>>() // offsets
        .Arg<ffi::Buffer<ffi::U32>>() // neighbors
        .Arg<ffi::Buffer<ffi::F32>>() // cov_bins
        .Arg<ffi::Buffer<ffi::F32>>() // cov_vals
        .Arg<ffi::Buffer<ffi::F32>>() // initial_values
        .Arg<ffi::Buffer<ffi::F32>>() // xi
        .Ret<ffi::Buffer<ffi::F32>>() // values
);


ffi::Error refine_transpose_xla_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> points, // (N, d) in topological order
    ffi::Buffer<ffi::U32> offsets, // (L,) first entry is number of initial points
    ffi::Buffer<ffi::U32> neighbors, // (N, k) in original order, initial points not used
    ffi::Buffer<ffi::F32> cov_bins, // (R,)
    ffi::Buffer<ffi::F32> cov_vals, // (R,)
    ffi::Buffer<ffi::F32> values_tangent, // (N,)
    ffi::ResultBuffer<ffi::F32> initial_values_tangent, // (N,) to use as a temporary buffer, only first N0 entries make sense
    ffi::ResultBuffer<ffi::F32> xi_tangent // (N,)
) {
    size_t n_points = points.dimensions()[0];
    size_t n_dim = points.dimensions()[1];
    size_t n_levels = offsets.dimensions()[0];
    size_t n_cov = cov_bins.dimensions()[0];
    size_t k = neighbors.dimensions()[1];

    if ((k < 2) || (k > 8)) return ffi::Error::InvalidArgument("only compiled for 2 <= k <= 5");
    if (n_dim > 3) return ffi::Error::InvalidArgument("only compiled for n_dim <= 3");

    if (initial_values_tangent->dimensions()[0] != n_points) {
        return ffi::Error::InvalidArgument("initial_values_tangent must have the same number of points as points because we use it as a temporary buffer");
    }
    
    using func_type = void(*)(cudaStream_t, const float*, const uint32_t*, const uint32_t*, const float*, const float*, const float*, float*, float*, size_t, size_t, size_t);
    func_type dispatch = nullptr;
    if (n_dim == 2) {
        if      (k == 2) dispatch = refine_transpose<2, 2>;
        else if (k == 3) dispatch = refine_transpose<3, 2>;
        else if (k == 4) dispatch = refine_transpose<4, 2>;
        else if (k == 5) dispatch = refine_transpose<5, 2>;
        else if (k == 6) dispatch = refine_transpose<6, 2>;
        else if (k == 7) dispatch = refine_transpose<7, 2>;
        else if (k == 8) dispatch = refine_transpose<8, 2>;
    } else if (n_dim == 3) {
        if      (k == 2) dispatch = refine_transpose<2, 3>;
        else if (k == 3) dispatch = refine_transpose<3, 3>;
        else if (k == 4) dispatch = refine_transpose<4, 3>;
        else if (k == 5) dispatch = refine_transpose<5, 3>;
        else if (k == 6) dispatch = refine_transpose<6, 3>;
        else if (k == 7) dispatch = refine_transpose<7, 3>;
        else if (k == 8) dispatch = refine_transpose<8, 3>;
    }

    dispatch(stream, points.typed_data(), offsets.typed_data(), neighbors.typed_data(), cov_bins.typed_data(), cov_vals.typed_data(), values_tangent.typed_data(), initial_values_tangent->typed_data(), xi_tangent->typed_data(), n_points, n_levels, n_cov);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    refine_transpose_xla, refine_transpose_xla_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>() // points
        .Arg<ffi::Buffer<ffi::U32>>() // offsets
        .Arg<ffi::Buffer<ffi::U32>>() // neighbors
        .Arg<ffi::Buffer<ffi::F32>>() // cov_bins
        .Arg<ffi::Buffer<ffi::F32>>() // cov_vals
        .Arg<ffi::Buffer<ffi::F32>>() // values_tangent
        .Ret<ffi::Buffer<ffi::F32>>() // initial_values_tangent
        .Ret<ffi::Buffer<ffi::F32>>() // xi_tangent

);

ffi::Error query_coarse_neighbors_xla_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> points, // (N, d)
    ffi::Buffer<ffi::S8> split_dims, // (N,) split dimensions
    ffi::ResultBuffer<ffi::U32> neighbors // (N, k)
) {
    size_t n_points = points.dimensions()[0];
    size_t n_dim = points.dimensions()[1];
    size_t k = neighbors->dimensions()[1];

    if (k > 16) return ffi::Error::InvalidArgument("only compiled for k <= 16");

    if (n_dim == 1) {
        query_coarse_neighbors<16, 1>(stream, points.typed_data(), split_dims.typed_data(), neighbors->typed_data(), n_points, k);
    } else if (n_dim == 2) {
        query_coarse_neighbors<16, 2>(stream, points.typed_data(), split_dims.typed_data(), neighbors->typed_data(), n_points, k);
    } else if (n_dim == 3) {
        query_coarse_neighbors<16, 3>(stream, points.typed_data(), split_dims.typed_data(), neighbors->typed_data(), n_points, k);
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    query_coarse_neighbors_xla, query_coarse_neighbors_xla_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>() // points
        .Arg<ffi::Buffer<ffi::S8>>() // split_dims
        .Ret<ffi::Buffer<ffi::U32>>() // neighbors
);
