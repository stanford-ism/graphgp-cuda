// bindings.cu
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <cuda_runtime.h>
#include <cstdint>

#include "common.h"
#include "sort.h"
#include "tree.h"
#include "query.h"
#include "depth.h"
#include "refine.h"
#include "refine_inv.h"
#include "refine_logdet.h"

using xla::ffi::Buffer;
using xla::ffi::ResultBuffer;
using xla::ffi::Error;
using xla::ffi::PlatformStream;
using xla::ffi::Ffi;
using xla::ffi::F32;
using xla::ffi::S32;

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

// ==================== REFINE OPERATIONS ====================

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
    size_t n_points = points.dimensions()[0];
    size_t n_dim = points.dimensions()[1];
    size_t n_levels = offsets.dimensions()[0];
    size_t n_cov = cov_bins.dimensions()[0];
    size_t n0 = n_points - neighbors.dimensions()[0];
    size_t k = neighbors.dimensions()[1];

    // handle both unbatched and arbitrarily batched cases
    size_t n_batches = 1;
    size_t n_batch_dims = cov_vals.dimensions().size() - 1;
    for (size_t i = 0; i < n_batch_dims; ++i) {
        n_batches *= cov_vals.dimensions()[i];
    }

    decltype(&refine<1,1,int,float>) dispatch = nullptr;
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

// ---------------------------------------------------------------------------------

Error refine_transpose_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> points, // (N, d) in topological order
    Buffer<S32> neighbors, // (N - n0, k) in original order
    Buffer<S32> offsets, // (L,) marking the ends of each batch
    Buffer<F32> cov_bins, // (R,)
    Buffer<F32> cov_vals, // (B1, B2, ..., R)
    Buffer<F32> values, // (B1, B2, ..., N)
    ResultBuffer<F32> initial_values, // (B1, B2, ..., n0)
    ResultBuffer<F32> xi, // (B1, B2, ..., N - n0)
    ResultBuffer<F32> values_buffer // (B1, B2, ..., N) to use as a temporary buffer
) {
    size_t n_points = points.dimensions()[0];
    size_t n_dim = points.dimensions()[1];
    size_t n_levels = offsets.dimensions()[0];
    size_t n_cov = cov_bins.dimensions()[0];
    size_t n0 = n_points - neighbors.dimensions()[0];
    size_t k = neighbors.dimensions()[1];

    // handle both unbatched and arbitrarily batched cases
    size_t n_batches = 1;
    size_t n_batch_dims = cov_vals.dimensions().size() - 1;
    for (size_t i = 0; i < n_batch_dims; ++i) {
        n_batches *= cov_vals.dimensions()[i];
    }

    decltype(&refine_transpose<1,1,int,float>) dispatch = nullptr;
    DISPATCH_K_DIM(dispatch, refine_transpose);
    dispatch(
        stream,
        points.typed_data(), 
        neighbors.typed_data(),
        offsets.typed_data(),
        cov_bins.typed_data(),
        cov_vals.typed_data(),
        values.typed_data(),
        values_buffer->typed_data(),
        initial_values->typed_data(),
        xi->typed_data(),
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
    refine_transpose_ffi, refine_transpose_ffi_impl,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<F32>>() // points
        .Arg<Buffer<S32>>() // neighbors
        .Arg<Buffer<S32>>() // offsets
        .Arg<Buffer<F32>>() // cov_bins
        .Arg<Buffer<F32>>() // cov_vals
        .Arg<Buffer<F32>>() // values
        .Ret<Buffer<F32>>() // initial_values
        .Ret<Buffer<F32>>() // xi
        .Ret<Buffer<F32>>() // values_buffer
);

// ---------------------------------------------------------------------------------

Error refine_jvp_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> points, // (N, d)
    Buffer<S32> neighbors, // (N - n0, k)
    Buffer<S32> offsets, // (B,) marking the ends of each batch
    Buffer<F32> cov_bins, // (R,)
    Buffer<F32> cov_vals, // (B1, B2, ..., R)
    Buffer<F32> initial_values, // (B1, B2, ..., n0)
    Buffer<F32> xi, // (B1, B2, ..., N - n0)
    Buffer<F32> cov_vals_tangent, // (B1, B2, ..., R)
    Buffer<F32> initial_values_tangent, // (B1, B2, ..., n0)
    Buffer<F32> xi_tangent, // (B1, B2, ..., N - n0)
    ResultBuffer<F32> values, // (B1, B2, ..., N)
    ResultBuffer<F32> values_tangent
) {
    size_t n_points = points.dimensions()[0];
    size_t n_dim = points.dimensions()[1];
    size_t n_levels = offsets.dimensions()[0];
    size_t n_cov = cov_bins.dimensions()[0];
    size_t n0 = n_points - neighbors.dimensions()[0];
    size_t k = neighbors.dimensions()[1];

    // handle both unbatched and arbitrarily batched cases
    size_t n_batches = 1;
    size_t n_batch_dims = cov_vals.dimensions().size() - 1;
    for (size_t i = 0; i < n_batch_dims; ++i) {
        n_batches *= cov_vals.dimensions()[i];
    }

    decltype(&refine_jvp<1,1,int,float>) dispatch = nullptr;
    DISPATCH_K_DIM(dispatch, refine_jvp);
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
    refine_jvp_ffi, refine_jvp_ffi_impl,
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

// ---------------------------------------------------------------------------------

Error refine_vjp_ffi_impl(
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
    ResultBuffer<F32> cov_vals_tangent, // (B1, B2, ..., R)
    ResultBuffer<F32> initial_values_tangent, // (B1, B2, ..., n0)
    ResultBuffer<F32> xi_tangent, // (B1, B2, ..., N - n0)
    ResultBuffer<F32> values_tangent_buffer // (B1, B2, ..., N) to use as a temporary buffer
) {
    size_t n_points = points.dimensions()[0];
    size_t n_dim = points.dimensions()[1];
    size_t n_levels = offsets.dimensions()[0];
    size_t n_cov = cov_bins.dimensions()[0];
    size_t n0 = n_points - neighbors.dimensions()[0];
    size_t k = neighbors.dimensions()[1];

    // handle both unbatched and arbitrarily batched cases
    size_t n_batches = 1;
    size_t n_batch_dims = cov_vals.dimensions().size() - 1;
    for (size_t i = 0; i < n_batch_dims; ++i) {
        n_batches *= cov_vals.dimensions()[i];
    }

    decltype(&refine_vjp<1,1,int,float>) dispatch = nullptr;
    DISPATCH_K_DIM(dispatch, refine_vjp);
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
    refine_vjp_ffi, refine_vjp_ffi_impl,
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
        .Ret<Buffer<F32>>() // cov_vals_tangent
        .Ret<Buffer<F32>>() // initial_values_tangent
        .Ret<Buffer<F32>>() // xi_tangent
        .Ret<Buffer<F32>>() // values_tangent_buffer
);

// ---------------------------------------------------------------------------------

Error refine_inv_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> points, // (N, d)
    Buffer<S32> neighbors, // (N - n0, k)
    Buffer<S32> offsets, // (B,) marking the ends of each batch
    Buffer<F32> cov_bins, // (R,)
    Buffer<F32> cov_vals, // (B1, B2, ..., R)
    Buffer<F32> values, // (B1, B2, ..., N)
    ResultBuffer<F32> initial_values, // (B1, B2, ..., n0)
    ResultBuffer<F32> xi // (B1, B2, ..., N - n0)
) {
    size_t n_points = points.dimensions()[0];
    size_t n_dim = points.dimensions()[1];
    size_t n_levels = offsets.dimensions()[0];
    size_t n_cov = cov_bins.dimensions()[0];
    size_t n0 = n_points - neighbors.dimensions()[0];
    size_t k = neighbors.dimensions()[1];

    // handle both unbatched and arbitrarily batched cases
    size_t n_batches = 1;
    size_t n_batch_dims = cov_vals.dimensions().size() - 1;
    for (size_t i = 0; i < n_batch_dims; ++i) {
        n_batches *= cov_vals.dimensions()[i];
    }

    decltype(&refine_inv<1,1,int,float>) dispatch = nullptr;
    DISPATCH_K_DIM(dispatch, refine_inv);
    dispatch(
        stream,
        points.typed_data(),
        neighbors.typed_data(), 
        offsets.typed_data(),
        cov_bins.typed_data(),
        cov_vals.typed_data(),
        values.typed_data(),
        initial_values->typed_data(),
        xi->typed_data(),
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
    refine_inv_ffi, refine_inv_ffi_impl,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<F32>>() // points
        .Arg<Buffer<S32>>() // neighbors
        .Arg<Buffer<S32>>() // offsets
        .Arg<Buffer<F32>>() // cov_bins
        .Arg<Buffer<F32>>() // cov_vals
        .Arg<Buffer<F32>>() // values
        .Ret<Buffer<F32>>() // initial_values
        .Ret<Buffer<F32>>() // xi
);

// ---------------------------------------------------------------------------------

Error refine_logdet_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> points, // (N, d)
    Buffer<S32> neighbors, // (N - n0, k)
    Buffer<S32> offsets, // (B,) marking the ends of each batch
    Buffer<F32> cov_bins, // (R,)
    Buffer<F32> cov_vals, // (B1, B2, ..., R)
    ResultBuffer<F32> logdet // (B1, B2, ...)
) {
    size_t n_points = points.dimensions()[0];
    size_t n_dim = points.dimensions()[1];
    size_t n_levels = offsets.dimensions()[0];
    size_t n_cov = cov_bins.dimensions()[0];
    size_t n0 = n_points - neighbors.dimensions()[0];
    size_t k = neighbors.dimensions()[1];

    // handle both unbatched and arbitrarily batched cases
    size_t n_batches = 1;
    size_t n_batch_dims = cov_vals.dimensions().size() - 1;
    for (size_t i = 0; i < n_batch_dims; ++i) {
        n_batches *= cov_vals.dimensions()[i];
    }

    decltype(&refine_logdet<1,1,int,float>) dispatch = nullptr;
    DISPATCH_K_DIM(dispatch, refine_logdet);
    dispatch(
        stream,
        points.typed_data(),
        neighbors.typed_data(), 
        offsets.typed_data(),
        cov_bins.typed_data(),
        cov_vals.typed_data(),
        logdet->typed_data(),
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
    refine_logdet_ffi, refine_logdet_ffi_impl,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<F32>>() // points
        .Arg<Buffer<S32>>() // neighbors
        .Arg<Buffer<S32>>() // offsets
        .Arg<Buffer<F32>>() // cov_bins
        .Arg<Buffer<F32>>() // cov_vals
        .Ret<Buffer<F32>>() // logdet
);

// ==================== GRAPH CONSTRUCTION ====================

Error build_tree_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> points_in, // (N, d)
    ResultBuffer<F32> points, // (N, d)
    ResultBuffer<S32> split_dims, // (N,)
    ResultBuffer<S32> indices, // (N,)
    ResultBuffer<S32> tags, // (N,)
    ResultBuffer<F32> ranges // (N,)
) {
    size_t n_points = points_in.dimensions()[0];
    size_t n_dim = points_in.dimensions()[1];
    cudaMemcpyAsync(points->typed_data(), points_in.typed_data(), n_points * n_dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    build_tree(
        stream,
        points_in.typed_data(),
        points->typed_data(),
        split_dims->typed_data(),
        indices->typed_data(),
        tags->typed_data(),
        ranges->typed_data(),
        n_dim,
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

// ---------------------------------------------------------------------------------

Error query_neighbors_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> points, // (N, d)
    Buffer<S32> split_dims, // (N,) split dimensions
    Buffer<S32> query_indices, // (Q,)
    Buffer<S32> max_indices, // (Q,)
    ResultBuffer<S32> neighbors // (Q, k)
) {
    size_t n_points = points.dimensions()[0];
    size_t n_queries = query_indices.dimensions()[0];
    size_t n_dim = points.dimensions()[1];
    size_t k = neighbors->dimensions()[1];

    decltype(&query_neighbors_kernel<1,1,int,float>) dispatch = nullptr;
    DISPATCH_K_DIM(dispatch, query_neighbors_kernel);
    dispatch<<<cld(n_queries, 256), 256, 0, stream>>>(
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

// ---------------------------------------------------------------------------------

Error query_preceding_neighbors_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> points, // (N, d)
    Buffer<S32> split_dims, // (N,) split dimensions
    ResultBuffer<S32> neighbors // (N - n0, k)
) {
    size_t n_points = points.dimensions()[0];
    size_t n_dim = points.dimensions()[1];
    size_t n0 = n_points - neighbors->dimensions()[0];
    size_t k = neighbors->dimensions()[1];

    decltype(&query_preceding_neighbors_kernel<1,1,int,float>) dispatch = nullptr;
    DISPATCH_K_DIM(dispatch, query_preceding_neighbors_kernel);
    dispatch<<<cld(n_points - n0, 256), 256, 0, stream>>>(
        points.typed_data(), 
        split_dims.typed_data(),
        neighbors->typed_data(),
        n0,
        k,
        n_points - n0
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
);

// ---------------------------------------------------------------------------------

Error compute_depths_parallel_ffi_impl(
    cudaStream_t stream,
    Buffer<S32> neighbors, // (N - n0, k)
    ResultBuffer<S32> depths, // (N,)
    ResultBuffer<S32> temp // (N,)
) {
    size_t n_points = depths->dimensions()[0];
    size_t n0 = n_points - neighbors.dimensions()[0];
    size_t k = neighbors.dimensions()[1];
    compute_depths_parallel(stream, neighbors.typed_data(), depths->typed_data(), temp->typed_data(), n0, k, n_points);
    return Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    compute_depths_parallel_ffi, compute_depths_parallel_ffi_impl,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<S32>>() // neighbors
        .Ret<Buffer<S32>>() // depths
        .Ret<Buffer<S32>>() // temp
);

// ---------------------------------------------------------------------------------

Error order_by_depth_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> points_in, // (N, d)
    Buffer<S32> indices_in, // (N,)
    Buffer<S32> neighbors_in, // (N - n0, k)
    Buffer<S32> depths_in, // (N,)
    ResultBuffer<F32> points_out,
    ResultBuffer<S32> indices_out,
    ResultBuffer<S32> neighbors_out,
    ResultBuffer<S32> depths_out,
    ResultBuffer<S32> temp // (2*N,)
) {
    size_t n_points = points_in.dimensions()[0];
    size_t n_dim = points_in.dimensions()[1];
    size_t n0 = n_points - neighbors_in.dimensions()[0];
    size_t k = neighbors_in.dimensions()[1];

    cudaMemcpyAsync(points_out->typed_data(), points_in.typed_data(), n_points * n_dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(indices_out->typed_data(), indices_in.typed_data(), n_points * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(neighbors_out->typed_data(), neighbors_in.typed_data(), (n_points - n0) * k * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(depths_out->typed_data(), depths_in.typed_data(), n_points * sizeof(int), cudaMemcpyDeviceToDevice, stream);

    order_by_depth(
        stream,
        points_out->typed_data(),
        indices_out->typed_data(),
        neighbors_out->typed_data(),
        depths_out->typed_data(),
        temp->typed_data(),
        temp->typed_data() + n_points,
        reinterpret_cast<float*>(temp->typed_data() + n_points),
        n0,
        k,
        n_points,
        n_dim
    );
    return Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    order_by_depth_ffi, order_by_depth_ffi_impl,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<F32>>() // points_in
        .Arg<Buffer<S32>>() // indices_in
        .Arg<Buffer<S32>>() // neighbors_in
        .Arg<Buffer<S32>>() // depths_in
        .Ret<Buffer<F32>>() // points_out
        .Ret<Buffer<S32>>() // indices_out
        .Ret<Buffer<S32>>() // neighbors_out
        .Ret<Buffer<S32>>() // depths_out
        .Ret<Buffer<S32>>() // temp
);

// ---------------------------------------------------------------------------------

Error build_graph_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> points_in,
    ResultBuffer<F32> points_out, // (N, d)
    ResultBuffer<S32> indices_out, // (N,)
    ResultBuffer<S32> neighbors_out, // (N - n0, k)
    ResultBuffer<S32> depths_out, // (N,)
    ResultBuffer<S32> temp_out // (2*N,)
) {
    size_t n_points = points_in.dimensions()[0];
    size_t n_dim = points_in.dimensions()[1];
    size_t n0 = n_points - neighbors_out->dimensions()[0];
    size_t k = neighbors_out->dimensions()[1];

    float* points = points_out->typed_data();
    int* indices = indices_out->typed_data();
    int* neighbors = neighbors_out->typed_data();
    int* depths = depths_out->typed_data();
    int* temp_int = temp_out->typed_data();
    float* temp_float = reinterpret_cast<float*>(temp_out->typed_data());

    // Build tree
    cudaMemcpyAsync(points, points_in.typed_data(), n_points * n_dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    build_tree(
        stream,
        points_in.typed_data(),
        points,
        depths, // use depths as split_dims
        indices,
        temp_int, // use temp for tags
        temp_float, // use temp for ranges
        n_dim,
        n_points
    );

    // Query preceding neighbors
    decltype(&query_preceding_neighbors_kernel<1,1,int,float>) query_dispatch = nullptr;
    DISPATCH_K_DIM(query_dispatch, query_preceding_neighbors_kernel);
    query_dispatch<<<cld(n_points - n0, 256), 256, 0, stream>>>(points, depths, neighbors, n0, k, n_points - n0);

    // Order by depth
    compute_depths_parallel(stream, neighbors, depths, temp_int, n0, k, n_points);
    order_by_depth(stream, points, indices, neighbors, depths, temp_int, temp_int + n_points, temp_float + n_points, n0, k, n_points, n_dim);

    return Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    build_graph_ffi, build_graph_ffi_impl,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<F32>>()  // points_in
        .Ret<Buffer<F32>>()  // points
        .Ret<Buffer<S32>>()  // indices
        .Ret<Buffer<S32>>()  // neighbors
        .Ret<Buffer<S32>>()  // depths
        .Ret<Buffer<S32>>()  // temp
);

// ---------------------------------------------------------------------------------

Error sort_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> keys_in,
    ResultBuffer<F32> keys_out
) {
    size_t n = keys_in.dimensions()[0];
    cudaMemcpyAsync(keys_out->typed_data(), keys_in.typed_data(), n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    sort(keys_out->typed_data(), n, stream);
    return Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sort_ffi, sort_ffi_impl,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<F32>>()  // keys_in
        .Ret<Buffer<F32>>()  // keys_out
);

Error sort_three_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> keys1_in,
    Buffer<F32> keys2_in,
    Buffer<F32> keys3_in,
    ResultBuffer<F32> keys1_out,
    ResultBuffer<F32> keys2_out,
    ResultBuffer<F32> keys3_out
) {
    size_t n = keys1_in.dimensions()[0];
    cudaMemcpyAsync(keys1_out->typed_data(), keys1_in.typed_data(), n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(keys2_out->typed_data(), keys2_in.typed_data(), n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(keys3_out->typed_data(), keys3_in.typed_data(), n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    sort(keys1_out->typed_data(), keys2_out->typed_data(), keys3_out->typed_data(), n, stream);
    return Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sort_three_ffi, sort_three_ffi_impl,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<F32>>()  // keys1_in
        .Arg<Buffer<F32>>()  // keys2_in
        .Arg<Buffer<F32>>()  // keys3_in
        .Ret<Buffer<F32>>()  // keys1_out
        .Ret<Buffer<F32>>()  // keys2_out
        .Ret<Buffer<F32>>()  // keys3_out
);

Error sort_four_ffi_impl(
    cudaStream_t stream,
    Buffer<F32> keys1_in,
    Buffer<F32> keys2_in,
    Buffer<F32> keys3_in,
    Buffer<F32> keys4_in,
    ResultBuffer<F32> keys1_out,
    ResultBuffer<F32> keys2_out,
    ResultBuffer<F32> keys3_out,
    ResultBuffer<F32> keys4_out
) {
    size_t n = keys1_in.dimensions()[0];
    cudaMemcpyAsync(keys1_out->typed_data(), keys1_in.typed_data(), n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(keys2_out->typed_data(), keys2_in.typed_data(), n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(keys3_out->typed_data(), keys3_in.typed_data(), n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(keys4_out->typed_data(), keys4_in.typed_data(), n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    sort(keys1_out->typed_data(), keys2_out->typed_data(), keys3_out->typed_data(), keys4_out->typed_data(), n, stream);
    return Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sort_four_ffi, sort_four_ffi_impl,
    Ffi::Bind()
        .Ctx<PlatformStream<cudaStream_t>>()
        .Arg<Buffer<F32>>()  // keys1_in
        .Arg<Buffer<F32>>()  // keys2_in
        .Arg<Buffer<F32>>()  // keys3_in
        .Arg<Buffer<F32>>()  // keys4_in
        .Ret<Buffer<F32>>()  // keys1_out
        .Ret<Buffer<F32>>()  // keys2_out
        .Ret<Buffer<F32>>()  // keys3_out
        .Ret<Buffer<F32>>()  // keys4_out
);