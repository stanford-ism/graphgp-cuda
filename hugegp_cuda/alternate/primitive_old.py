import ctypes
from pathlib import Path

import jax
import jax.numpy as jnp

from jax import lax
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import mlir, batching
from jax._src.interpreters import ad

from .trace_print import trace


# Define custom primitive, the "refine" function is exposed
refine_p = Primitive("hugegp_cuda_refine")
refine_linear_transpose_p = Primitive("hugegp_cuda_refine_linear_transpose")


def _initialize():
    # Register CUDA bindings as FFI targets
    so_path = next(Path(__file__).parent.glob("libhugegp_cuda*"))
    hugegp_cuda_lib = ctypes.cdll.LoadLibrary(str(so_path))
    jax.ffi.register_ffi_target(
        "hugegp_cuda_refine_bind",
        jax.ffi.pycapsule(hugegp_cuda_lib.refine_bind),
        platform="gpu",
    )
    jax.ffi.register_ffi_target(
        "hugegp_cuda_refine_linear_transpose_bind",
        jax.ffi.pycapsule(hugegp_cuda_lib.refine_linear_transpose_bind),
        platform="gpu",
    )
    jax.ffi.register_ffi_target(
        "hugegp_cuda_query_coarse_neighbors_bind",
        jax.ffi.pycapsule(hugegp_cuda_lib.query_coarse_neighbors_bind),
        platform="gpu",
    )

    # Register required functions for JAX
    refine_p.def_impl(refine_impl)
    refine_p.def_abstract_eval(refine_abstract_eval)
    batching.primitive_batchers[refine_p] = refine_batch
    mlir.register_lowering(refine_p, refine_lowering, platform="gpu")  # type: ignore
    ad.primitive_jvps[refine_p] = refine_value_and_jvp
    ad.primitive_linearizations[refine_p] = refine_linearization
    ad.primitive_transposes[refine_p] = refine_transpose


@trace("refine")
def refine(points, offsets, neighbors, cov_bins, cov_vals, initial_values, xi):
    return refine_p.bind(points, offsets, neighbors, cov_bins, cov_vals, initial_values, xi)


@trace("refine_impl")
def refine_impl(*args):
    values = jax.ffi.ffi_call(
        "hugegp_cuda_refine_bind",
        jax.ShapeDtypeStruct(args[6].shape, jnp.float32),
    )(*args)
    return values


@trace("refine_abstract_eval")
def refine_abstract_eval(*args):
    return ShapedArray(args[6].shape, dtype=jnp.float32)


@trace("refine_lowering")
def refine_lowering(ctx, *args):
    return jax.ffi.ffi_lowering("hugegp_cuda_refine_bind")(ctx, *args)


@trace("refine_value_and_jvp")
def refine_value_and_jvp(primals, tangents):
    nzs = [type(t) is not ad.Zero for t in tangents]
    primals_out, nzs_out, res, jvp_func = refine_linearization(nzs, *primals)
    return primals_out, jvp_func(res, *[t for t in tangents if type(t) is not ad.Zero])




@trace("refine_linearization")
def refine_linearization(nzs, *primals):
    p, o, n, cb, cv, iv, x = primals

    if any(nzs[i] for i in [0, 1, 2, 3]):
        raise NotImplementedError(
            "Differentiation not supported for points, offsets, neighbors, or cov_bins."
        )

    assert not nzs[4]
    assert nzs[5] and nzs[6], "Should differentiate with respect to initial_values and xi."

    return (
        refine(p, o, n, cb, cv, iv, x),
        (False,),  # output is not ad.Zero
        None,  # no residuals to pass on in linear case
        lambda _, div, dx: refine(p, o, n, cb, cv, div, dx),
    )

@trace("refine_transpose")
def refine_transpose(tangents_out, *primals):
    p, o, n, cb, cv, iv, x = primals
    dv = tangents_out

    if any(ad.is_undefined_primal(t) for t in [p, o, n, cb, cv]):
        raise NotImplementedError("Differentiation only supported for xi and initial_values.")

    if not all(ad.is_undefined_primal(t) for t in [x, iv]):
        raise NotImplementedError("Not differentiated with respect to xi and initial_values?")

    dv_buffer, div, dx = jax.ffi.ffi_call(
        "hugegp_cuda_refine_linear_transpose_bind",
        (
            jax.ShapeDtypeStruct(dv.aval.shape, jnp.float32),
            jax.ShapeDtypeStruct(iv.aval.shape, jnp.float32),
            jax.ShapeDtypeStruct(dv.aval.shape, jnp.float32),
        ),
    )(p, o, n, cb, cv, dv)
    return None, None, None, None, None, div, dx


@trace("refine_batch")
def refine_batch(vector_args, batch_axes):
    p, o, n, cb, cv, iv, x = vector_args
    pa, oa, na, cba, cva, iva, xa = batch_axes

    if any(a is not None for a in [pa, oa, na, cba]):
        raise NotImplementedError("Batching only supported for cov_vals, initial_values, and xi.")

    # extract batch size, guaranteed to be the same for all batched arguments
    batch_size = 1
    for i in range(len(vector_args)):
        if batch_axes[i] is not None:
            batch_size = vector_args[i].shape[batch_axes[i]]
            break

    # either move or add batch dimension for each argument
    if cva is None:
        cv = jnp.broadcast_to(cv, (batch_size,) + cv.shape)
    else:
        cv = batching.moveaxis(cv, cva, 0)

    if iva is None:
        iv = jnp.broadcast_to(iv, (batch_size,) + iv.shape)
    else:
        iv = batching.moveaxis(iv, iva, 0)

    if xa is None:
        x = jnp.broadcast_to(x, (batch_size,) + x.shape)
    else:
        x = batching.moveaxis(x, xa, 0)

    # call the primitive with the batched arguments
    return refine(p, o, n, cb, cv, iv, x), 0


# @trace("refine_linear_transpose")
# def refine_linear_transpose(points, offsets, neighbors, cov_bins, cov_vals, values_tangent):
#     return refine_linear_transpose_p.bind(
#         points, offsets, neighbors, cov_bins, cov_vals, values_tangent
#     )

# @trace("refine_linear_transpose_impl")
# def refine_linear_transpose_impl(points, offsets, neighbors, cov_bins, cov_vals, values_tangent, iv_shape=None):
#     dv_buffer, div, dx = jax.ffi.ffi_call(
#         "hugegp_cuda_refine_linear_transpose_bind",
#         (
#             jax.ShapeDtypeStruct(values_tangent.shape, jnp.float32),
#             jax.ShapeDtypeStruct(iv_shape, jnp.float32),
#             jax.ShapeDtypeStruct(values_tangent.shape, jnp.float32),
#         ),
#     )(points, offsets, neighbors, cov_bins, cov_vals, values_tangent)
#     return dv_buffer, div, dx

# @trace("refine_linear_transpose_abstract_eval")
# def refine_linear_transpose_abstract_eval(points, offsets, neighbors, cov_bins, cov_vals, values_tangent, iv_shape=None):
#     return (
#         ShapedArray(values_tangent.shape, dtype=jnp.float32),  # dv
#         ShapedArray(iv_shape, dtype=jnp.float32),  # div
#         ShapedArray(values_tangent.shape, dtype=jnp.float32),  # dx
#     )

# @trace("refine_linear_transpose_lowering")
# def refine_linear_transpose_lowering(ctx, *args):
#     return jax.ffi.ffi_lowering("hugegp_cuda_refine_linear_transpose_bind")(ctx, *args)


# @trace("refine_value_and_jvp")
# def refine_value_and_jvp(primals, tangents):
#     p, o, n, cb, cv, iv, x = primals
#     dp, do, dn, dcb, dcv, div, dx = tangents

#     if not all(type(t) is ad.Zero for t in [dp, do, dn, dcb]):
#         raise NotImplementedError(
#             "Differentiation not supported for points, offsets, neighbors, or cov_bins."
#         )

#     if type(dcv) is ad.Zero:
#         if type(div) is ad.Zero:
#             div = lax.zeros_like_array(iv)
#         if type(dx) is ad.Zero:
#             dx = lax.zeros_like_array(x)

#         # Fused version introduced complexity with abstract eval for VJP, as well as higher order derivatives.
#         # Just use separate calls for now, probably just a bit of a performance hit.
#         # primals_out, tangents_out = refine_jvp_linear(p, o, n, cb, cv, iv, x, div, dx)
#         primals_out = refine(p, o, n, cb, cv, iv, x)
#         tangents_out = refine(p, o, n, cb, cv, div, dx)
#         return primals_out, tangents_out

#     # else if type(div) is ad.Zero and type(dx) is ad.Zero:
#     #     tangents_out =


# @trace("refine_transpose")
# def refine_transpose(tangents_out, *primals):
#     p, o, n, cb, cv, iv, x = primals
#     dv = tangents_out

#     if any(ad.is_undefined_primal(t) for t in [p, o, n, cb, cv]):
#         raise NotImplementedError("Differentiation only supported for xi and initial_values.")

#     if not all(ad.is_undefined_primal(t) for t in [x, iv]):
#         raise NotImplementedError("Not differentiated with respect to xi and initial_values?")

#     div_buffer, dx = jax.ffi.ffi_call(
#         "hugegp_cuda_refine_vjp_linear_bind",
#         (
#             jax.ShapeDtypeStruct(dv.aval.shape, jnp.float32),
#             jax.ShapeDtypeStruct(dv.aval.shape, jnp.float32),
#         ),
#     )(p, o, n, cb, cv, dv)

#     # div used as temporary buffer with n_points, so we need to slice it down to size
#     if iv.aval.ndim == 1:
#         div = div_buffer[: iv.aval.shape[0]]
#     elif iv.aval.ndim == 2:
#         div = div_buffer[:, : iv.aval.shape[1]]
#     return None, None, None, None, None, div, dx


# @trace("refine_vjp_linear_batch")
# def refine_vjp_linear_batch(vector_args, batch_axes):


# @trace("refine_jvp_linear")
# def refine_jvp_linear(*args):
#     return refine_jvp_linear_p.bind(*args)

# @trace("refine_jvp_linear_impl")
# def refine_jvp_linear_impl(*args):
#     values, values_tangent = jax.ffi.ffi_call(
#         "hugegp_cuda_refine_jvp_linear",
#         (
#             jax.ShapeDtypeStruct(args[6].shape, jnp.float32),
#             jax.ShapeDtypeStruct(args[6].shape, jnp.float32),
#         ),
#     )(*args)
#     return values, values_tangent

# @trace("refine_jvp_linear_abstract_eval")
# def refine_jvp_linear_abstract_eval(*args):
#     return (
#         ShapedArray(args[6].shape, dtype=jnp.float32),
#         ShapedArray(args[6].shape, dtype=jnp.float32),
#     )

# @trace("refine_jvp_linear_lowering")
# def refine_jvp_linear_lowering(ctx, *args):
#     return jax.ffi.ffi_lowering("hugegp_cuda_refine_jvp_linear")(ctx, *args)


def query_coarse_neighbors(points, split_dims, k):
    call = jax.ffi.ffi_call(
        "hugegp_cuda_query_coarse_neighbors_bind",
        jax.ShapeDtypeStruct((len(points), k), jnp.uint32),
    )
    neighbors = call(points, split_dims)
    return neighbors
