// bindings.cu
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <cuda_runtime.h>
#include <cstdint>

#include "refine.h"
#include "refine_linear_transpose.h"
#include "refine_nonlinear_jvp.h"
#include "refine_nonlinear_vjp.h"
#include "tree.h"
#include "query.h"
#include "depth.h"
#include "sort.h"

using xla::ffi::Buffer;
using xla::ffi::ResultBuffer;
using xla::ffi::Error;
using xla::ffi::PlatformStream;
using xla::ffi::Ffi;
using xla::ffi::F32;
using xla::ffi::S32;

#define DISPATCH_DIM(DEST, FUNC) \
    if      (n_dim == 1) DEST = FUNC<1>; \
    else if (n_dim == 2) DEST = FUNC<2>; \
    else if (n_dim == 3) DEST = FUNC<3>; \
    else if (n_dim == 4) DEST = FUNC<4>; \
    else if (n_dim == 5) DEST = FUNC<5>; \
    else if (n_dim == 6) DEST = FUNC<6>; \
    else throw std::runtime_error("only compiled for 1 <= n_dim <= 6");

#define DISPATCH_K_DIM(DEST, FUNC) \
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
        else throw std::runtime_error("only compiled for k <= 32"); \
    } else { \
        throw std::runtime_error("only compiled for 1 <= n_dim <= 6"); \
    }

Error refine_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> points, // (N, d)
    Buffer<S32> neighbors, // (N - n0, k)
    Buffer<S32> offsets, // (B,) marking the ends of each batch
    Buffer<F32> cov_bins, // (R,)
    Buffer<F32> cov_vals, // (B1, B2, ..., R)
    Buffer<F32> initial_values, // (B1, B2, ..., n0)
    Buffer<F32> xi, // (B1, B2, ..., N - n0)
    ResultBuffer<F32> values // (B1, B2, ..., N)
) {
    int n_points = points.dimensions()[0];
    int n_dim = points.dimensions()[1];
    int n_levels = offsets.dimensions()[0];
    int n_cov = cov_bins.dimensions()[0];
    int n0 = n_points - neighbors.dimensions()[0];
    int k = neighbors.dimensions()[1];

    // handle both unbatched and arbitrarily batched cases
    size_t n_batches = 1;
    size_t n_batch_dims = cov_vals.dimensions().size() - 1;
    for (size_t i = 0; i < n_batch_dims; ++i) {
        n_batches *= cov_vals.dimensions()[i];
    }

    decltype(&refine<1,1>) dispatch = nullptr;
    DISPATCH_K_DIM(dispatch, refine);
    dispatch(
        stream,
        points.typed_data(),
        neighbors.typed_data(), 
        offsets.typed_data(),
        cov_bins.typed_data(),
        cov_vals.typed_data(),
        initial_values.typed_data(),
        xi.typed_data(),
        values->typed_data(),
        n0,
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
        .Arg<Buffer<S32>>() // neighbors
        .Arg<Buffer<S32>>() // offsets
        .Arg<Buffer<F32>>() // cov_bins
        .Arg<Buffer<F32>>() // cov_vals
        .Arg<Buffer<F32>>() // initial_values
        .Arg<Buffer<F32>>() // xi
        .Ret<Buffer<F32>>() // values
);


Error refine_linear_transpose_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> points, // (N, d) in topological order
    Buffer<S32> neighbors, // (N - n0, k) in original order
    Buffer<S32> offsets, // (L,) marking the ends of each batch
    Buffer<F32> cov_bins, // (R,)
    Buffer<F32> cov_vals, // (B1, B2, ..., R)
    Buffer<F32> values_tangent, // (B1, B2, ..., N)
    ResultBuffer<F32> values_tangent_buffer, // (B1, B2, ..., N) to use as a temporary buffer
    ResultBuffer<F32> initial_values_tangent, // (B1, B2, ..., n0)
    ResultBuffer<F32> xi_tangent // (B1, B2, ..., N - n0)
) {
    int n_points = points.dimensions()[0];
    int n_dim = points.dimensions()[1];
    int n_levels = offsets.dimensions()[0];
    int n_cov = cov_bins.dimensions()[0];
    int n0 = n_points - neighbors.dimensions()[0];
    int k = neighbors.dimensions()[1];

    // handle both unbatched and arbitrarily batched cases
    size_t n_batches = 1;
    size_t n_batch_dims = cov_vals.dimensions().size() - 1;
    for (size_t i = 0; i < n_batch_dims; ++i) {
        n_batches *= cov_vals.dimensions()[i];
    }

    decltype(&refine_linear_transpose<1,1>) dispatch = nullptr;
    DISPATCH_K_DIM(dispatch, refine_linear_transpose);
    dispatch(
        stream,
        points.typed_data(), 
        neighbors.typed_data(),
        offsets.typed_data(),
        cov_bins.typed_data(),
        cov_vals.typed_data(),
        values_tangent.typed_data(),
        values_tangent_buffer->typed_data(),
        initial_values_tangent->typed_data(),
        xi_tangent->typed_data(),
        n0,
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
        .Arg<Buffer<S32>>() // neighbors
        .Arg<Buffer<S32>>() // offsets
        .Arg<Buffer<F32>>() // cov_bins
        .Arg<Buffer<F32>>() // cov_vals
        .Arg<Buffer<F32>>() // values_tangent
        .Ret<Buffer<F32>>() // values_tangent_buffer
        .Ret<Buffer<F32>>() // initial_values_tangent
        .Ret<Buffer<F32>>() // xi_tangent
);

Error refine_nonlinear_jvp_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> points, // (N, d)
    Buffer<S32> neighbors, // (N - n0, k)
    Buffer<S32> offsets, // (B,) marking the ends of each batch
    Buffer<F32> cov_bins, // (R,)
    Buffer<F32> cov_vals, // (B1, B2, ..., R)
    Buffer<F32> initial_values, // (B1, B2, ..., n0)
    Buffer<F32> xi, // (B1, B2, ..., N - n0)
    Buffer<F32> cov_vals_tangent,
    Buffer<F32> initial_values_tangent,
    Buffer<F32> xi_tangent,
    ResultBuffer<F32> values, // (B1, B2, ..., N)
    ResultBuffer<F32> values_tangent
) {
    int n_points = points.dimensions()[0];
    int n_dim = points.dimensions()[1];
    int n_levels = offsets.dimensions()[0];
    int n_cov = cov_bins.dimensions()[0];
    int n0 = n_points - neighbors.dimensions()[0];
    int k = neighbors.dimensions()[1];

    // handle both unbatched and arbitrarily batched cases
    size_t n_batches = 1;
    size_t n_batch_dims = cov_vals.dimensions().size() - 1;
    for (size_t i = 0; i < n_batch_dims; ++i) {
        n_batches *= cov_vals.dimensions()[i];
    }

    decltype(&refine_nonlinear_jvp<1,1>) dispatch = nullptr;
    DISPATCH_K_DIM(dispatch, refine_nonlinear_jvp);
    dispatch(
        stream,
        points.typed_data(),
        neighbors.typed_data(), 
        offsets.typed_data(),
        cov_bins.typed_data(),
        cov_vals.typed_data(),
        initial_values.typed_data(),
        xi.typed_data(),
        cov_vals_tangent.typed_data(),
        initial_values_tangent.typed_data(),
        xi_tangent.typed_data(),
        values->typed_data(),
        values_tangent->typed_data(),
        n0,
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
        .Arg<Buffer<S32>>() // neighbors
        .Arg<Buffer<S32>>() // offsets
        .Arg<Buffer<F32>>() // cov_bins
        .Arg<Buffer<F32>>() // cov_vals
        .Arg<Buffer<F32>>() // initial_values
        .Arg<Buffer<F32>>() // xi
        .Arg<Buffer<F32>>() // cov_vals_tangent
        .Arg<Buffer<F32>>() // initial_values_tangent
        .Arg<Buffer<F32>>() // xi_tangent
        .Ret<Buffer<F32>>() // values
        .Ret<Buffer<F32>>() // values_tangent
);

Error refine_nonlinear_vjp_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> points, // (N, d)
    Buffer<S32> neighbors, // (N - n0, k)
    Buffer<S32> offsets, // (B,) marking the ends of each batch
    Buffer<F32> cov_bins, // (R,)
    Buffer<F32> cov_vals, // (B1, B2, ..., R)
    Buffer<F32> initial_values, // (B1, B2, ..., n0)
    Buffer<F32> xi, // (B1, B2, ..., N - n0)
    Buffer<F32> values, // (B1, B2, ..., N)
    Buffer<F32> values_tangent, // (B1, B2, ..., N)
    ResultBuffer<F32> values_tangent_buffer, // (B1, B2, ..., N) to use as a temporary buffer
    ResultBuffer<F32> cov_vals_tangent, // (B1, B2, ..., R)
    ResultBuffer<F32> initial_values_tangent, // (B1, B2, ..., n0)
    ResultBuffer<F32> xi_tangent // (B1, B2, ..., N - n0)
) {
    int n_points = points.dimensions()[0];
    int n_dim = points.dimensions()[1];
    int n_levels = offsets.dimensions()[0];
    int n_cov = cov_bins.dimensions()[0];
    int n0 = n_points - neighbors.dimensions()[0];
    int k = neighbors.dimensions()[1];

    // handle both unbatched and arbitrarily batched cases
    size_t n_batches = 1;
    size_t n_batch_dims = cov_vals.dimensions().size() - 1;
    for (size_t i = 0; i < n_batch_dims; ++i) {
        n_batches *= cov_vals.dimensions()[i];
    }

    decltype(&refine_nonlinear_vjp<1,1>) dispatch = nullptr;
    DISPATCH_K_DIM(dispatch, refine_nonlinear_vjp);
    dispatch(
        stream,
        points.typed_data(),
        neighbors.typed_data(), 
        offsets.typed_data(),
        cov_bins.typed_data(),
        cov_vals.typed_data(),
        initial_values.typed_data(),
        xi.typed_data(),
        values.typed_data(),
        values_tangent.typed_data(),
        values_tangent_buffer->typed_data(),
        cov_vals_tangent->typed_data(),
        initial_values_tangent->typed_data(),
        xi_tangent->typed_data(),
        n0,
        k,
        n_points,
        n_levels,
        n_cov,
        n_batches
    );

    return Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    refine_nonlinear_vjp_ffi, refine_nonlinear_vjp_ffi_impl,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<F32>>() // points
        .Arg<Buffer<S32>>() // neighbors
        .Arg<Buffer<S32>>() // offsets
        .Arg<Buffer<F32>>() // cov_bins
        .Arg<Buffer<F32>>() // cov_vals
        .Arg<Buffer<F32>>() // initial_values
        .Arg<Buffer<F32>>() // xi
        .Arg<Buffer<F32>>() // values
        .Arg<Buffer<F32>>() // values_tangent
        .Ret<Buffer<F32>>() // values_tangent_buffer
        .Ret<Buffer<F32>>() // cov_vals_tangent
        .Ret<Buffer<F32>>() // initial_values_tangent
        .Ret<Buffer<F32>>() // xi_tangent
);

Error build_tree_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> points_in, // (N, d)
    ResultBuffer<F32> points, // (N, d)
    ResultBuffer<S32> split_dims, // (N,)
    ResultBuffer<S32> indices, // (N,)
    ResultBuffer<S32> tags, // (N,)
    ResultBuffer<F32> ranges // (N,)
) {
    int n_points = points_in.dimensions()[0];
    int n_dim = points_in.dimensions()[1];

    cudaMemcpyAsync(points->typed_data(), points_in.typed_data(), n_points * n_dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    decltype(&build_tree<1>) dispatch = nullptr;
    DISPATCH_DIM(dispatch, build_tree);
    dispatch(
        stream,
        points->typed_data(),
        split_dims->typed_data(),
        indices->typed_data(),
        tags->typed_data(),
        ranges->typed_data(),
        n_points
    );

    return Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    build_tree_ffi, build_tree_ffi_impl,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<F32>>() // points_in
        .Ret<Buffer<F32>>() // points
        .Ret<Buffer<S32>>() // split_dims
        .Ret<Buffer<S32>>() // indices
        .Ret<Buffer<S32>>() // tags
        .Ret<Buffer<F32>>() // ranges
);

Error query_neighbors_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> points, // (N, d)
    Buffer<S32> split_dims, // (N,) split dimensions
    Buffer<S32> query_indices, // (Q,)
    Buffer<S32> max_indices, // (Q,)
    ResultBuffer<S32> neighbors // (Q, k)
) {
    int n_points = points.dimensions()[0];
    int n_queries = query_indices.dimensions()[0];
    int n_dim = points.dimensions()[1];
    int k = neighbors->dimensions()[1];

    decltype(&query_neighbors<1,1>) dispatch = nullptr;
    DISPATCH_K_DIM(dispatch, query_neighbors);
    dispatch(
        stream,
        points.typed_data(),
        split_dims.typed_data(),
        query_indices.typed_data(),
        max_indices.typed_data(),
        neighbors->typed_data(),
        k,
        n_points,
        n_queries
    );

    return Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    query_neighbors_ffi, query_neighbors_ffi_impl,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<F32>>() // points
        .Arg<Buffer<S32>>() // split_dims
        .Arg<Buffer<S32>>() // query_indices
        .Arg<Buffer<S32>>() // max_indices
        .Ret<Buffer<S32>>() // neighbors
);

Error query_preceding_neighbors_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> points, // (N, d)
    Buffer<S32> split_dims, // (N,) split dimensions
    ResultBuffer<S32> neighbors, // (N - n0, k)
    int n0
) {
    int n_points = points.dimensions()[0];
    int n_dim = points.dimensions()[1];
    int k = neighbors->dimensions()[1];

    decltype(&query_preceding_neighbors<1,1>) dispatch = nullptr;
    DISPATCH_K_DIM(dispatch, query_preceding_neighbors);
    dispatch(
        stream,
        points.typed_data(),
        split_dims.typed_data(),
        neighbors->typed_data(),
        n0,
        k,
        n_points
    );

    return Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    query_preceding_neighbors_ffi, query_preceding_neighbors_ffi_impl,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<F32>>() // points
        .Arg<Buffer<S32>>() // split_dims
        .Ret<Buffer<S32>>() // neighbors
        .Attr<int>("n0") // n0
);

Error compute_depths_ffi_impl(
    cudaStream_t stream,
    Buffer<S32> neighbors, // (N - n0, k)
    ResultBuffer<S32> depths, // (N,)
    int n0
) {
    int n_points = neighbors.dimensions()[0] + n0;
    int k = neighbors.dimensions()[1];
    compute_depths<<<1, 1, 0, stream>>>(
        neighbors.typed_data(),
        depths->typed_data(),
        n_points,
        n0,
        k
    );
    return Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    compute_depths_ffi, compute_depths_ffi_impl,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<S32>>() // neighbors
        .Ret<Buffer<S32>>() // depths
        .Attr<int>("n0") // n0
);

Error sort_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> keys,
    ResultBuffer<F32> keys_out
) {
    int n = keys.dimensions()[0];
    cudaMemcpyAsync(keys_out->typed_data(), keys.typed_data(), n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    bitonic_sort(stream, keys_out->typed_data(), n);
    return Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sort_ffi, sort_ffi_impl,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<F32>>() // keys
        .Ret<Buffer<F32>>() // keys_out
);