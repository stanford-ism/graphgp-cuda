// bindings.cu
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <cuda_runtime.h>
#include <cstdint>

#include "refine.h"
#include "refine_linear_transpose.h"

#include "neighbors.h"

using xla::ffi::Buffer;
using xla::ffi::ResultBuffer;
using xla::ffi::Error;
using xla::ffi::PlatformStream;
using xla::ffi::Ffi;
using xla::ffi::F32;
using xla::ffi::U32;
using xla::ffi::S8;

#define DISPATCH(DEST, FUNC) \
    if (n_dim == 2) { \
        if      (k == 1) DEST = FUNC<1, 2>; \
        else if (k == 2) DEST = FUNC<2, 2>; \
        else if (k == 3) DEST = FUNC<3, 2>; \
        else if (k == 4) DEST = FUNC<4, 2>; \
        else if (k == 5) DEST = FUNC<5, 2>; \
        else if (k == 6) DEST = FUNC<6, 2>; \
        else if (k == 7) DEST = FUNC<7, 2>; \
        else if (k == 8) DEST = FUNC<8, 2>; \
        else throw std::runtime_error("only compiled for 1 <= k <= 8"); \
    } else if (n_dim == 3) { \
        if      (k == 1) DEST = FUNC<1, 3>; \
        else if (k == 2) DEST = FUNC<2, 3>; \
        else if (k == 3) DEST = FUNC<3, 3>; \
        else if (k == 4) DEST = FUNC<4, 3>; \
        else if (k == 5) DEST = FUNC<5, 3>; \
        else if (k == 6) DEST = FUNC<6, 3>; \
        else if (k == 7) DEST = FUNC<7, 3>; \
        else if (k == 8) DEST = FUNC<8, 3>; \
        else throw std::runtime_error("only compiled for 1 <= k <= 8"); \
    } else { \
        throw std::runtime_error("only compiled for 2 <= n_dim <= 3"); \
    }

Error refine_xla(
    cudaStream_t stream,
    Buffer<F32> points, // (N, d) in topological order
    Buffer<U32> offsets, // (L,) first entry is number of initial points
    Buffer<U32> neighbors, // (N, k) in original order, initial points not used
    Buffer<F32> cov_bins, // (R,)
    Buffer<F32> cov_vals, // (B1, B2, ..., R)
    Buffer<F32> initial_values, // (B1, B2, ..., N0)
    Buffer<F32> xi, // (B1, B2, ..., N) the first N0 are not used
    ResultBuffer<F32> values // (B1, B2, ..., N)
) {
    size_t n_points = points.dimensions()[0];
    size_t n_dim = points.dimensions()[1];
    size_t n_levels = offsets.dimensions()[0];
    size_t n_cov = cov_bins.dimensions()[0];
    size_t k = neighbors.dimensions()[1];

    // handle both unbatched and arbitrarily batched cases
    size_t n_batches = 1;
    size_t n_batch_dims = cov_vals.dimensions().size() - 1;
    for (size_t i = 0; i < n_batch_dims; ++i) {
        n_batches *= cov_vals.dimensions()[i];
    }

    decltype(&refine<1,1>) dispatch = nullptr;
    DISPATCH(dispatch, refine);
    dispatch(
        stream,
        points.typed_data(),
        offsets.typed_data(),
        neighbors.typed_data(), 
        cov_bins.typed_data(),
        cov_vals.typed_data(),
        initial_values.typed_data(),
        xi.typed_data(),
        values->typed_data(),
        n_points,
        n_levels,
        n_cov,
        n_batches
    );

    return Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    refine_bind, refine_xla,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<F32>>() // points
        .Arg<Buffer<U32>>() // offsets
        .Arg<Buffer<U32>>() // neighbors
        .Arg<Buffer<F32>>() // cov_bins
        .Arg<Buffer<F32>>() // cov_vals
        .Arg<Buffer<F32>>() // initial_values
        .Arg<Buffer<F32>>() // xi
        .Ret<Buffer<F32>>() // values
);



Error refine_linear_transpose_xla(
    cudaStream_t stream,
    Buffer<F32> points, // (N, d) in topological order
    Buffer<U32> offsets, // (L,) first entry is number of initial points
    Buffer<U32> neighbors, // (N, k) in original order, initial points not used
    Buffer<F32> cov_bins, // (R,)
    Buffer<F32> cov_vals, // (B1, B2, ..., N)
    Buffer<F32> values_tangent, // (B1, B2, ..., N)
    ResultBuffer<F32> values_tangent_buffer, // (B1, B2, ..., N) to use as a temporary buffer
    ResultBuffer<F32> initial_values_tangent, // (B1, B2, ..., N0) copied from values_tangent_buffer
    ResultBuffer<F32> xi_tangent // (B1, B2, ..., N)
) {
    size_t n_points = points.dimensions()[0];
    size_t n_dim = points.dimensions()[1];
    size_t n_levels = offsets.dimensions()[0];
    size_t n_cov = cov_bins.dimensions()[0];
    size_t k = neighbors.dimensions()[1];

    // handle both unbatched and arbitrarily batched cases
    size_t n_batches = 1;
    size_t n_batch_dims = cov_vals.dimensions().size() - 1;
    for (size_t i = 0; i < n_batch_dims; ++i) {
        n_batches *= cov_vals.dimensions()[i];
    }

    decltype(&refine_linear_transpose<1,1>) dispatch = nullptr;
    DISPATCH(dispatch, refine_linear_transpose);
    dispatch(
        stream,
        points.typed_data(),
        offsets.typed_data(), 
        neighbors.typed_data(),
        cov_bins.typed_data(),
        cov_vals.typed_data(),
        values_tangent.typed_data(),
        values_tangent_buffer->typed_data(),
        initial_values_tangent->typed_data(),
        xi_tangent->typed_data(),
        n_points,
        n_levels,
        n_cov,
        n_batches
    );

    return Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    refine_linear_transpose_bind, refine_linear_transpose_xla,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<F32>>() // points
        .Arg<Buffer<U32>>() // offsets
        .Arg<Buffer<U32>>() // neighbors
        .Arg<Buffer<F32>>() // cov_bins
        .Arg<Buffer<F32>>() // cov_vals
        .Arg<Buffer<F32>>() // values_tangent
        .Ret<Buffer<F32>>() // values_tangent_buffer
        .Ret<Buffer<F32>>() // initial_values_tangent
        .Ret<Buffer<F32>>() // xi_tangent
);


// Error refine_jvp_cov_xla(
//     cudaStream_t stream,
//     Buffer<F32> points, // (N, d) in topological order
//     Buffer<U32> offsets, // (L,) first entry is number of initial points
//     Buffer<U32> neighbors, // (N, k) in original order, initial points not used
//     Buffer<F32> cov_bins, // (R,)
//     Buffer<F32> cov_vals, // (R,)
//     Buffer<F32> initial_values, // (N0,)
//     Buffer<F32> xi, // (N,) the first N0 are not used
//     Buffer<F32> cov_vals_tangent, // (R,)
//     ResultBuffer<F32> values_tangent // (N,)
// ) {
//     size_t n_points = points.dimensions()[0];
//     size_t n_dim = points.dimensions()[1];
//     size_t n_levels = offsets.dimensions()[0];
//     size_t n_cov = cov_bins.dimensions()[0];
//     size_t k = neighbors.dimensions()[1];

//     if ((k < 2) || (k > 8)) return Error::InvalidArgument("only compiled for 2 <= k <= 8");
//     if (n_dim > 3) return Error::InvalidArgument("only compiled for 2 <= n_dim <= 3");

//     decltype(&refine_jvp_cov<1,1>) dispatch = nullptr;
//     DISPATCH(dispatch, refine_jvp_cov);
//     dispatch(
//         stream,
//         points.typed_data(),
//         offsets.typed_data(),
//         neighbors.typed_data(), 
//         cov_bins.typed_data(),
//         cov_vals.typed_data(),
//         initial_values.typed_data(),
//         xi.typed_data(),
//         cov_vals_tangent.typed_data(),
//         values_tangent->typed_data(),
//         n_points,
//         n_levels,
//         n_cov
//     );

//     return Error::Success();
// }

// XLA_FFI_DEFINE_HANDLER_SYMBOL(
//     refine_jvp_cov_bind, refine_jvp_cov_xla,
//     Ffi::Bind()
//         .Ctx<PlatformStream<cudaStream_t>>()
//         .Arg<Buffer<F32>>() // points
//         .Arg<Buffer<U32>>() // offsets
//         .Arg<Buffer<U32>>() // neighbors
//         .Arg<Buffer<F32>>() // cov_bins
//         .Arg<Buffer<F32>>() // cov_vals
//         .Arg<Buffer<F32>>() // initial_values
//         .Arg<Buffer<F32>>() // xi
//         .Arg<Buffer<F32>>() // cov_vals_tangent
//         .Ret<Buffer<F32>>() // values_tangent
// );

Error query_coarse_neighbors_xla(
    cudaStream_t stream,
    Buffer<F32> points, // (N, d)
    Buffer<S8> split_dims, // (N,) split dimensions
    ResultBuffer<U32> neighbors // (N, k)
) {
    size_t n_points = points.dimensions()[0];
    size_t n_dim = points.dimensions()[1];
    size_t k = neighbors->dimensions()[1];

    if (k > 16) return Error::InvalidArgument("only compiled for k <= 16");

    if (n_dim == 1) {
        query_coarse_neighbors<16, 1>(stream, points.typed_data(), split_dims.typed_data(), neighbors->typed_data(), n_points, k);
    } else if (n_dim == 2) {
        query_coarse_neighbors<16, 2>(stream, points.typed_data(), split_dims.typed_data(), neighbors->typed_data(), n_points, k);
    } else if (n_dim == 3) {
        query_coarse_neighbors<16, 3>(stream, points.typed_data(), split_dims.typed_data(), neighbors->typed_data(), n_points, k);
    } else {
        return Error::InvalidArgument("only compiled for n_dim <= 3");
    }

    return Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    query_coarse_neighbors_bind, query_coarse_neighbors_xla,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<F32>>() // points
        .Arg<Buffer<S8>>() // split_dims
        .Ret<Buffer<U32>>() // neighbors
);



// Error refine_jvp_linear_xla(
//     cudaStream_t stream,
//     Buffer<F32> points, // (N, d) in topological order
//     Buffer<U32> offsets, // (L,) first entry is number of initial points
//     Buffer<U32> neighbors, // (N, k) in original order, initial points not used
//     Buffer<F32> cov_bins, // (R,)
//     Buffer<F32> cov_vals, // (R,)
//     Buffer<F32> initial_values, // (N0,)
//     Buffer<F32> xi, // (N,) the first N0 are not used
//     Buffer<F32> initial_values_tangent, // (N0,)
//     Buffer<F32> xi_tangent, // (N,) the first N0 are not used
//     ResultBuffer<F32> values, // (N,)
//     ResultBuffer<F32> values_tangent // (N,)
// ) {
//     size_t n_points = points.dimensions()[0];
//     size_t n_dim = points.dimensions()[1];
//     size_t n_levels = offsets.dimensions()[0];
//     size_t n_cov = cov_bins.dimensions()[0];
//     size_t k = neighbors.dimensions()[1];

//     decltype(&refine_jvp_linear<1,1>) dispatch = nullptr;
//     DISPATCH(dispatch, refine_jvp_linear);
//     dispatch(
//         stream,
//         points.typed_data(),
//         offsets.typed_data(),
//         neighbors.typed_data(), 
//         cov_bins.typed_data(),
//         cov_vals.typed_data(),
//         initial_values.typed_data(),
//         xi.typed_data(),
//         initial_values_tangent.typed_data(),
//         xi_tangent.typed_data(),
//         values->typed_data(),
//         values_tangent->typed_data(),
//         n_points,
//         n_levels,
//         n_cov
//     );

//     return Error::Success();
// }

// XLA_FFI_DEFINE_HANDLER_SYMBOL(
//     refine_jvp_linear_bind, refine_jvp_linear_xla,
//     Ffi::Bind()
//         .Ctx<PlatformStream<cudaStream_t>>()
//         .Arg<Buffer<F32>>() // points
//         .Arg<Buffer<U32>>() // offsets
//         .Arg<Buffer<U32>>() // neighbors
//         .Arg<Buffer<F32>>() // cov_bins
//         .Arg<Buffer<F32>>() // cov_vals
//         .Arg<Buffer<F32>>() // initial_values
//         .Arg<Buffer<F32>>() // xi
//         .Arg<Buffer<F32>>() // initial_values_tangent
//         .Arg<Buffer<F32>>() // xi_tangent
//         .Ret<Buffer<F32>>() // values
//         .Ret<Buffer<F32>>() // values_tangent
// );