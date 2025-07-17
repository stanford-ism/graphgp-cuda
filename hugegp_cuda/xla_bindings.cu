// bindings.cu
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <cuda_runtime.h>
#include <cstdint>

#include "refine.h"
#include "refine_linear_transpose.h"
#include "query.h"
#include "query_alt.h"
#include "levels.h"

using xla::ffi::Buffer;
using xla::ffi::ResultBuffer;
using xla::ffi::Error;
using xla::ffi::PlatformStream;
using xla::ffi::Ffi;
using xla::ffi::F32;
using xla::ffi::U32;
using xla::ffi::S8;

#define DISPATCH(DEST, FUNC) \
    if (n_dim == 1) { \
        if      (k <= 4)  DEST = FUNC<4,  1>; \
        else if (k <= 8)  DEST = FUNC<8,  1>; \
        else if (k <= 16) DEST = FUNC<16, 1>; \
        else if (k <= 32) DEST = FUNC<32, 1>; \
        else throw std::runtime_error("only compiled for k <= 32"); \
    } else if (n_dim == 2) { \
        if      (k <= 4)  DEST = FUNC<4,  2>; \
        else if (k <= 8)  DEST = FUNC<8,  2>; \
        else if (k <= 16) DEST = FUNC<16, 2>; \
        else if (k <= 32) DEST = FUNC<32, 2>; \
        else throw std::runtime_error("only compiled for k <= 32"); \
    } else if (n_dim == 3) { \
        if      (k <= 4)  DEST = FUNC<4,  3>; \
        else if (k <= 8)  DEST = FUNC<8,  3>; \
        else if (k <= 16) DEST = FUNC<16, 3>; \
        else if (k <= 32) DEST = FUNC<32, 3>; \
        else throw std::runtime_error("only compiled for k <= 32"); \
    } else if (n_dim == 4) { \
        if      (k <= 4)  DEST = FUNC<4,  4>; \
        else if (k <= 8)  DEST = FUNC<8,  4>; \
        else if (k <= 16) DEST = FUNC<16, 4>; \
        else if (k <= 32) DEST = FUNC<32, 4>; \
        else throw std::runtime_error("only compiled for k <= 32"); \
    } else if (n_dim == 5) { \
        if      (k <= 4)  DEST = FUNC<4,  5>; \
        else if (k <= 8)  DEST = FUNC<8,  5>; \
        else if (k <= 16) DEST = FUNC<16, 5>; \
        else if (k <= 32) DEST = FUNC<32, 5>; \
        else throw std::runtime_error("only compiled for k <= 32"); \
    } else if (n_dim == 6) { \
        if      (k <= 4)  DEST = FUNC<4,  6>; \
        else if (k <= 8)  DEST = FUNC<8,  6>; \
        else if (k <= 16) DEST = FUNC<16, 6>; \
        else if (k <= 32) DEST = FUNC<32, 6>; \
    } else { \
        throw std::runtime_error("only compiled for 1 <= n_dim <= 6"); \
    }

Error refine_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> points, // (N, d) in topological order
    Buffer<U32> neighbors, // (N, k) in original order, initial points not used
    Buffer<U32> offsets, // (L,) first entry is number of initial points
    Buffer<F32> cov_bins, // (R,)
    Buffer<F32> cov_vals, // (B1, B2, ..., R)
    Buffer<F32> initial_cholesky, // (B1, B2, ..., N0, N0)
    Buffer<F32> xi, // (B1, B2, ..., N)
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
        neighbors.typed_data(), 
        offsets.typed_data(),
        cov_bins.typed_data(),
        cov_vals.typed_data(),
        initial_cholesky.typed_data(),
        xi.typed_data(),
        values->typed_data(),
        k,
        n_points,
        n_levels,
        n_cov,
        n_batches
    );

    return Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    refine_ffi, refine_ffi_impl,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<F32>>() // points
        .Arg<Buffer<U32>>() // neighbors
        .Arg<Buffer<U32>>() // offsets
        .Arg<Buffer<F32>>() // cov_bins
        .Arg<Buffer<F32>>() // cov_vals
        .Arg<Buffer<F32>>() // initial_cholesky
        .Arg<Buffer<F32>>() // xi
        .Ret<Buffer<F32>>() // values
);

Error refine_nonlinear_jvp_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> points, // (N, d) in topological order
    Buffer<U32> neighbors, // (N, k) in original order, initial points not used
    Buffer<U32> offsets, // (L,) first entry is number of initial points
    Buffer<F32> cov_bins, // (R,)
    Buffer<F32> cov_vals, // (B1, B2, ..., R)
    Buffer<F32> initial_cholesky, // (B1, B2, ..., N0, N0)
    Buffer<F32> xi, // (B1, B2, ..., N)
    Buffer<F32> cov_vals_tangent,
    Buffer<F32> initial_cholesky_tangent,
    Buffer<F32> xi_tangent,
    ResultBuffer<F32> values, // (B1, B2, ..., N)
    ResultBuffer<F32> values_tangent
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
        neighbors.typed_data(), 
        offsets.typed_data(),
        cov_bins.typed_data(),
        cov_vals.typed_data(),
        initial_cholesky.typed_data(),
        xi.typed_data(),
        values->typed_data(),
        k,
        n_points,
        n_levels,
        n_cov,
        n_batches
    );

    return Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    refine_nonlinear_jvp_ffi, refine_nonlinear_jvp_ffi_impl,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<F32>>() // points
        .Arg<Buffer<U32>>() // neighbors
        .Arg<Buffer<U32>>() // offsets
        .Arg<Buffer<F32>>() // cov_bins
        .Arg<Buffer<F32>>() // cov_vals
        .Arg<Buffer<F32>>() // initial_cholesky
        .Arg<Buffer<F32>>() // xi
        .Arg<Buffer<F32>>() // cov_vals_tangent
        .Arg<Buffer<F32>>() // initial_cholesky_tangent
        .Arg<Buffer<F32>>() // xi_tangent
        .Ret<Buffer<F32>>() // values
        .Ret<Buffer<F32>>() // values_tangent
);


Error refine_linear_transpose_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> points, // (N, d) in topological order
    Buffer<U32> neighbors, // (N, k) in original order, initial points not used
    Buffer<U32> offsets, // (L,) first entry is number of initial points
    Buffer<F32> cov_bins, // (R,)
    Buffer<F32> cov_vals, // (B1, B2, ..., N)
    Buffer<F32> initial_cholesky, // (B1, B2, ..., N0, N0)
    Buffer<F32> values_tangent, // (B1, B2, ..., N)
    ResultBuffer<F32> values_tangent_buffer, // (B1, B2, ..., N) to use as a temporary buffer
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
        neighbors.typed_data(),
        offsets.typed_data(),
        cov_bins.typed_data(),
        cov_vals.typed_data(),
        initial_cholesky.typed_data(),
        values_tangent.typed_data(),
        values_tangent_buffer->typed_data(),
        xi_tangent->typed_data(),
        k,
        n_points,
        n_levels,
        n_cov,
        n_batches
    );

    return Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    refine_linear_transpose_ffi, refine_linear_transpose_ffi_impl,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<F32>>() // points
        .Arg<Buffer<U32>>() // neighbors
        .Arg<Buffer<U32>>() // offsets
        .Arg<Buffer<F32>>() // cov_bins
        .Arg<Buffer<F32>>() // cov_vals
        .Arg<Buffer<F32>>() // initial_cholesky
        .Arg<Buffer<F32>>() // values_tangent
        .Ret<Buffer<F32>>() // values_tangent_buffer
        .Ret<Buffer<F32>>() // xi_tangent
);


Error query_preceding_neighbors_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> points, // (N, d)
    Buffer<S8> split_dims, // (N,) split dimensions
    ResultBuffer<U32> neighbors // (N, k)
) {
    size_t n_points = points.dimensions()[0];
    size_t n_dim = points.dimensions()[1];
    size_t k = neighbors->dimensions()[1];

    decltype(&query_preceding_neighbors<1,1>) dispatch = nullptr;
    DISPATCH(dispatch, query_preceding_neighbors);
    dispatch(
        stream,
        points.typed_data(),
        split_dims.typed_data(),
        neighbors->typed_data(),
        n_points,
        k
    );

    return Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    query_preceding_neighbors_ffi, query_preceding_neighbors_ffi_impl,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<F32>>() // points
        .Arg<Buffer<S8>>() // split_dims
        .Ret<Buffer<U32>>() // neighbors
);


Error query_preceding_neighbors_alt_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> points, // (N, d)
    Buffer<S8> split_dims, // (N,) split dimensions
    ResultBuffer<U32> neighbors // (N, k)
) {
    size_t n_points = points.dimensions()[0];
    size_t n_dim = points.dimensions()[1];
    size_t k = neighbors->dimensions()[1];

    decltype(&query_preceding_neighbors_alt<1,1>) dispatch = nullptr;
    DISPATCH(dispatch, query_preceding_neighbors_alt);
    dispatch(
        stream,
        points.typed_data(),
        split_dims.typed_data(),
        neighbors->typed_data(),
        n_points,
        k
    );

    return Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    query_preceding_neighbors_alt_ffi, query_preceding_neighbors_alt_ffi_impl,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<F32>>() // points
        .Arg<Buffer<S8>>() // split_dims
        .Ret<Buffer<U32>>() // neighbors
);

Error compute_levels_ffi_impl(
    cudaStream_t stream,
    Buffer<U32> neighbors, // (N, k)
    ResultBuffer<U32> levels, // (N,)
    uint32_t n_initial
) {
    size_t n_points = neighbors.dimensions()[0];
    size_t k = neighbors.dimensions()[1];
    compute_levels<<<1, 1, 0, stream>>>(
        neighbors.typed_data(),
        levels->typed_data(),
        n_points,
        n_initial,
        k
    );
    return Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    compute_levels_ffi, compute_levels_ffi_impl,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<U32>>() // neighbors
        .Ret<Buffer<U32>>() // levels
        .Attr<uint32_t>("n_initial") // n_initial
);