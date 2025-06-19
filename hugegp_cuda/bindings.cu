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
    ffi::Buffer<ffi::F32> xi, // (N,)
    ffi::Buffer<ffi::U32> neighbors, // (N, k) in original order, initial points not used
    ffi::Buffer<ffi::U32> level_offsets, // (L,) first entry is number of initial points
    ffi::Buffer<ffi::F32> initial_values, // (N0,)
    ffi::Buffer<ffi::F32> cov_r, // (R,)
    ffi::Buffer<ffi::F32> cov, // (R,)
    ffi::ResultBuffer<ffi::F32> values // (N,)
) {
    size_t n_points = points.dimensions()[0];
    size_t n_dim = points.dimensions()[1];
    size_t n_levels = level_offsets.dimensions()[0];
    size_t n_cov = cov_r.dimensions()[0];
    size_t k = neighbors.dimensions()[1];

    if ((k < 2) || (k > 5)) {
        return ffi::Error::InvalidArgument("only compiled for k in [2, 5]");
    }
    if (n_dim > 3) {
        return ffi::Error::InvalidArgument("only compiled for n_dim <= 3");
    }

    if (n_dim == 1) {
        if      (k == 2) refine<2, 1>(stream, points.typed_data(), xi.typed_data(), neighbors.typed_data(), level_offsets.typed_data(), initial_values.typed_data(), cov_r.typed_data(), cov.typed_data(), values->typed_data(), n_points, n_levels, n_cov);
        else if (k == 3) refine<3, 1>(stream, points.typed_data(), xi.typed_data(), neighbors.typed_data(), level_offsets.typed_data(), initial_values.typed_data(), cov_r.typed_data(), cov.typed_data(), values->typed_data(), n_points, n_levels, n_cov);
        else if (k == 4) refine<4, 1>(stream, points.typed_data(), xi.typed_data(), neighbors.typed_data(), level_offsets.typed_data(), initial_values.typed_data(), cov_r.typed_data(), cov.typed_data(), values->typed_data(), n_points, n_levels, n_cov);
        else if (k == 5) refine<5, 1>(stream, points.typed_data(), xi.typed_data(), neighbors.typed_data(), level_offsets.typed_data(), initial_values.typed_data(), cov_r.typed_data(), cov.typed_data(), values->typed_data(), n_points, n_levels, n_cov);
    } else if (n_dim == 2) {
        if      (k == 2) refine<2, 2>(stream, points.typed_data(), xi.typed_data(), neighbors.typed_data(), level_offsets.typed_data(), initial_values.typed_data(), cov_r.typed_data(), cov.typed_data(), values->typed_data(), n_points, n_levels, n_cov);
        else if (k == 3) refine<3, 2>(stream, points.typed_data(), xi.typed_data(), neighbors.typed_data(), level_offsets.typed_data(), initial_values.typed_data(), cov_r.typed_data(), cov.typed_data(), values->typed_data(), n_points, n_levels, n_cov);
        else if (k == 4) refine<4, 2>(stream, points.typed_data(), xi.typed_data(), neighbors.typed_data(), level_offsets.typed_data(), initial_values.typed_data(), cov_r.typed_data(), cov.typed_data(), values->typed_data(), n_points, n_levels, n_cov);
        else if (k == 5) refine<5, 2>(stream, points.typed_data(), xi.typed_data(), neighbors.typed_data(), level_offsets.typed_data(), initial_values.typed_data(), cov_r.typed_data(), cov.typed_data(), values->typed_data(), n_points, n_levels, n_cov);
    } else if (n_dim == 3) {
        if      (k == 2) refine<2, 3>(stream, points.typed_data(), xi.typed_data(), neighbors.typed_data(), level_offsets.typed_data(), initial_values.typed_data(), cov_r.typed_data(), cov.typed_data(), values->typed_data(), n_points, n_levels, n_cov);
        else if (k == 3) refine<3, 3>(stream, points.typed_data(), xi.typed_data(), neighbors.typed_data(), level_offsets.typed_data(), initial_values.typed_data(), cov_r.typed_data(), cov.typed_data(), values->typed_data(), n_points, n_levels, n_cov);
        else if (k == 4) refine<4, 3>(stream, points.typed_data(), xi.typed_data(), neighbors.typed_data(), level_offsets.typed_data(), initial_values.typed_data(), cov_r.typed_data(), cov.typed_data(), values->typed_data(), n_points, n_levels, n_cov);
        else if (k == 5) refine<5, 3>(stream, points.typed_data(), xi.typed_data(), neighbors.typed_data(), level_offsets.typed_data(), initial_values.typed_data(), cov_r.typed_data(), cov.typed_data(), values->typed_data(), n_points, n_levels, n_cov);
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    refine_xla, refine_xla_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>() // points
        .Arg<ffi::Buffer<ffi::F32>>() // xi
        .Arg<ffi::Buffer<ffi::U32>>() // neighbors
        .Arg<ffi::Buffer<ffi::U32>>() // level_offsets
        .Arg<ffi::Buffer<ffi::F32>>() // initial_values
        .Arg<ffi::Buffer<ffi::F32>>() // cov_r
        .Arg<ffi::Buffer<ffi::F32>>() // cov_eval
        .Ret<ffi::Buffer<ffi::F32>>() // values
);


ffi::Error refine_transpose_xla_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> points, // (N, d) in topological order
    ffi::Buffer<ffi::U32> neighbors, // (N, k) in original order, initial points not used
    ffi::Buffer<ffi::U32> level_offsets, // (L,) first entry is number of initial points
    ffi::Buffer<ffi::F32> cov_r, // (R,)
    ffi::Buffer<ffi::F32> cov, // (R,)
    ffi::Buffer<ffi::F32> values_tangent, // (N,)
    ffi::ResultBuffer<ffi::F32> xi_tangent, // (N,)
    ffi::ResultBuffer<ffi::F32> initial_values_tangent // (N0,)
) {
    size_t n_points = points.dimensions()[0];
    size_t n_dim = points.dimensions()[1];
    size_t n_levels = level_offsets.dimensions()[0];
    size_t n_cov = cov_r.dimensions()[0];
    size_t k = neighbors.dimensions()[1];

    if ((k < 2) || (k > 5)) {
        return ffi::Error::InvalidArgument("only compiled for k in [2, 5]");
    }
    if (n_dim > 3) {
        return ffi::Error::InvalidArgument("only compiled for n_dim <= 3");
    }

    if (n_dim != 2) {
        return ffi::Error::InvalidArgument("only compiled for n_dim == 2");
    }

    if (k != 4) {
        return ffi::Error::InvalidArgument("only compiled for k == 4");
    }

    refine_transpose<4, 2>(stream, points.typed_data(), neighbors.typed_data(), level_offsets.typed_data(), cov_r.typed_data(), cov.typed_data(), values_tangent.typed_data(), xi_tangent->typed_data(), initial_values_tangent->typed_data(), n_points, n_levels, n_cov);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    refine_transpose_xla, refine_transpose_xla_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>() // points
        .Arg<ffi::Buffer<ffi::U32>>() // neighbors
        .Arg<ffi::Buffer<ffi::U32>>() // level_offsets
        .Arg<ffi::Buffer<ffi::F32>>() // cov_r
        .Arg<ffi::Buffer<ffi::F32>>() // cov
        .Arg<ffi::Buffer<ffi::F32>>() // values_tangent
        .Ret<ffi::Buffer<ffi::F32>>() // xi_tangent
        .Ret<ffi::Buffer<ffi::F32>>() // initial_values_tangent
);



ffi::Error query_coarse_neighbors_xla_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> points, // (N, d)
    ffi::Buffer<ffi::S8> split_dims, // (N,) split dimensions
    ffi::ResultBuffer<ffi::U32> neighbors, // (N, k)
    int k
) {
    size_t n_points = points.dimensions()[0];
    size_t n_dim = points.dimensions()[1];

    if (k > 8) {
        return ffi::Error::InvalidArgument("only compiled for k <= 8");
    }

    if (n_dim > 3) {
        return ffi::Error::InvalidArgument("only compiled for n_dim <= 3");
    }

    if (n_dim == 1) {
        query_coarse_neighbors<8, 1>(stream, points.typed_data(), split_dims.typed_data(), neighbors->typed_data(), k, n_points);
    } else if (n_dim == 2) {
        query_coarse_neighbors<8, 2>(stream, points.typed_data(), split_dims.typed_data(), neighbors->typed_data(), k, n_points);
    } else if (n_dim == 3) {
        query_coarse_neighbors<8, 3>(stream, points.typed_data(), split_dims.typed_data(), neighbors->typed_data(), k, n_points);
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
        .Attr<int>("k")
);
