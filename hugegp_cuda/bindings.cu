// bindings.cu
#pragma once

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <cuda_runtime.h>
#include <cstdint>

#include "refine.h"
#include "neighbors.h"

namespace ffi = xla::ffi;

ffi::Error refine_static_xla_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> points, // (N, d)
    ffi::Buffer<ffi::F32> xi, // (N,)
    ffi::Buffer<ffi::U32> indices, // (N,) puts points in topological order
    ffi::Buffer<ffi::U32> neighbors, // (N, k_coarse) in original order, initial points not used
    ffi::Buffer<ffi::U32> level_offsets, // (L,) first entry is number of initial points
    ffi::Buffer<ffi::F32> cov_distances, // (R,)
    ffi::Buffer<ffi::F32> cov_values, // (R,)
    ffi::Buffer<ffi::F32> initial_values, // (N0,)
    ffi::ResultBuffer<ffi::F32> values // (N,)
) {
    size_t n_points = points.dimensions()[0];
    size_t n_levels = level_offsets.dimensions()[0];
    size_t n_cov = cov_distances.dimensions()[0];
    size_t k_coarse = neighbors.dimensions()[1];

    if (k_coarse != 4) {
        return ffi::Error::InvalidArgument("only compiled for k_coarse == 4");
    }

    if (k_coarse == 2) {
        refine_static<2, 2>(stream, points.typed_data(), xi.typed_data(), indices.typed_data(), neighbors.typed_data(), level_offsets.typed_data(), cov_distances.typed_data(), cov_values.typed_data(), initial_values.typed_data(), values->typed_data(), n_points, n_levels, n_cov);
    } else if (k_coarse == 3) {
        refine_static<3, 2>(stream, points.typed_data(), xi.typed_data(), indices.typed_data(), neighbors.typed_data(), level_offsets.typed_data(), cov_distances.typed_data(), cov_values.typed_data(), initial_values.typed_data(), values->typed_data(), n_points, n_levels, n_cov);
    } else if (k_coarse == 4) {
        refine_static<4, 2>(stream, points.typed_data(), xi.typed_data(), indices.typed_data(), neighbors.typed_data(), level_offsets.typed_data(), cov_distances.typed_data(), cov_values.typed_data(), initial_values.typed_data(), values->typed_data(), n_points, n_levels, n_cov);
    } else if (k_coarse == 5) {
        refine_static<5, 2>(stream, points.typed_data(), xi.typed_data(), indices.typed_data(), neighbors.typed_data(), level_offsets.typed_data(), cov_distances.typed_data(), cov_values.typed_data(), initial_values.typed_data(), values->typed_data(), n_points, n_levels, n_cov);
    }

    // refine_static<4, 2>(
    //     stream,
    //     points.typed_data(),
    //     xi.typed_data(),
    //     indices.typed_data(),
    //     neighbors.typed_data(),
    //     level_offsets.typed_data(),
    //     cov_distances.typed_data(),
    //     cov_values.typed_data(),
    //     initial_values.typed_data(),
    //     values->typed_data(),
    //     n_points,
    //     n_levels,
    //     n_cov
    // );

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    refine_static_xla, refine_static_xla_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>() // points
        .Arg<ffi::Buffer<ffi::F32>>() // xi
        .Arg<ffi::Buffer<ffi::U32>>() // indices
        .Arg<ffi::Buffer<ffi::U32>>() // neighbors
        .Arg<ffi::Buffer<ffi::U32>>() // level_offsets
        .Arg<ffi::Buffer<ffi::F32>>() // cov_distances
        .Arg<ffi::Buffer<ffi::F32>>() // cov_values
        .Arg<ffi::Buffer<ffi::F32>>() // initial_values
        .Ret<ffi::Buffer<ffi::F32>>() // values
);


ffi::Error query_coarse_neighbors_xla_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> points, // (N, d)
    ffi::Buffer<ffi::U32> indices, // (N,) kd tree order
    ffi::Buffer<ffi::S8> split_dims, // (N,) split dimensions
    ffi::ResultBuffer<ffi::U32> neighbors, // (N, k_coarse) in original order
    int k
) {
    size_t n_points = points.dimensions()[0];
    size_t n_dim = points.dimensions()[1];

    if (k > 8) {
        return ffi::Error::InvalidArgument("only compiled for k <= 8");
    }

    query_coarse_neighbors<8, 2>(
        stream,
        points.typed_data(),
        indices.typed_data(),
        split_dims.typed_data(),
        neighbors->typed_data(),
        k,
        n_points
    );

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    query_coarse_neighbors_xla, query_coarse_neighbors_xla_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>() // points
        .Arg<ffi::Buffer<ffi::U32>>() // indices
        .Arg<ffi::Buffer<ffi::S8>>() // split_dims
        .Ret<ffi::Buffer<ffi::U32>>() // neighbors
        .Attr<int>("k")
);
