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
refine_linear_p = Primitive("hugegp_cuda_refine_linear")
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

    # Register refine_linear primitive
    refine_linear_p.def_impl(refine_linear_impl)
    refine_linear_p.def_abstract_eval(refine_linear_abstract_eval)
    batching.primitive_batchers[refine_linear_p] = refine_linear_batch
    mlir.register_lowering(refine_linear_p, refine_linear_lowering, platform="gpu")  # type: ignore
    ad.primitive_jvps[refine_linear_p] = refine_linear_value_and_jvp
    ad.primitive_transposes[refine_linear_p] = refine_linear_transpose_rule

    # Register refine_linear_transpose primitive
    refine_linear_transpose_p.def_impl(refine_linear_transpose_impl)
    refine_linear_transpose_p.def_abstract_eval(refine_linear_transpose_abstract_eval)
    batching.primitive_batchers[refine_linear_transpose_p] = refine_linear_transpose_batch
    mlir.register_lowering(
        refine_linear_transpose_p,
        refine_linear_transpose_lowering,  # type: ignore
        platform="gpu",
    )
    ad.primitive_jvps[refine_linear_transpose_p] = refine_linear_transpose_value_and_jvp
    ad.primitive_transposes[refine_linear_transpose_p] = refine_linear_transpose_transpose_rule
    refine_linear_transpose_p.multiple_results = True


def refine(points, offsets, neighbors, cov_bins, cov_vals, initial_values, xi):
    return refine_linear(points, offsets, neighbors, cov_bins, cov_vals, initial_values, xi)


# ================ refine_linear primitive ================

# @trace("refine_linear")
def refine_linear(points, offsets, neighbors, cov_bins, cov_vals, initial_values, xi):
    return refine_linear_p.bind(points, offsets, neighbors, cov_bins, cov_vals, initial_values, xi)

# @trace("refine_linear_impl")
def refine_linear_impl(*args):
    return jax.ffi.ffi_call(
        "hugegp_cuda_refine_bind", jax.ShapeDtypeStruct(args[6].shape, jnp.float32)
    )(*args)

# @trace("refine_linear_abstract_eval")
def refine_linear_abstract_eval(*args):
    return ShapedArray(args[6].shape, jnp.float32)

# @trace("refine_linear_lowering")
def refine_linear_lowering(ctx, *args):
    return jax.ffi.ffi_lowering("hugegp_cuda_refine_bind")(ctx, *args)

# @trace("refine_linear_value_and_jvp")
def refine_linear_value_and_jvp(primals, tangents):
    if any(type(t) is not ad.Zero for t in tangents[:5]):
        raise NotImplementedError(
            "Differentiation for refine_linear only supported for initial_values and xi."
        )
    if any(type(t) is ad.Zero for t in tangents[5:]):
        raise NotImplementedError("Not differentiated with respect to initial_values and xi?")
    primals_out = refine_linear(*primals)
    tangents_out = refine_linear(*primals[:5], *tangents[5:])
    return primals_out, tangents_out

# @trace("refine_linear_transpose_rule")
def refine_linear_transpose_rule(tangents_out, *primals):
    p, o, n, cb, cv, iv, x = primals
    dv = tangents_out

    if any(ad.is_undefined_primal(t) for t in [p, o, n, cb, cv]):
        raise NotImplementedError("Differentiation only supported for xi and initial_values.")

    if not all(ad.is_undefined_primal(t) for t in [iv, x]):
        raise NotImplementedError("Not differentiated with respect to xi and initial_values?")

    if type(dv) is ad.Zero:
        raise NotImplementedError("Not differentiated with respect to values?")

    dv_buffer, div, dx = refine_linear_transpose(p, o, n, cb, cv, dv, iv_shape=iv.aval.shape)
    return None, None, None, None, None, div, dx

# @trace("refine_linear_batch")
def refine_linear_batch(vector_args, batch_axes):
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
    return refine_linear(p, o, n, cb, cv, iv, x), 0


# ============ refine_linear_transpose primitive =============


def refine_linear_transpose(points, offsets, neighbors, cov_bins, cov_vals, values, iv_shape=None):
    return refine_linear_transpose_p.bind(
        points, offsets, neighbors, cov_bins, cov_vals, values, iv_shape=iv_shape
    )


def refine_linear_transpose_impl(*args, iv_shape=None):
    return jax.ffi.ffi_call(
        "hugegp_cuda_refine_linear_transpose_bind",
        (
            jax.ShapeDtypeStruct(args[5].shape, jnp.float32),
            jax.ShapeDtypeStruct(iv_shape, jnp.float32),
            jax.ShapeDtypeStruct(args[5].shape, jnp.float32),
        ),
    )(*args)


def refine_linear_transpose_abstract_eval(*args, iv_shape=None):
    return (
        ShapedArray(args[5].shape, jnp.float32),
        ShapedArray(iv_shape, jnp.float32),
        ShapedArray(args[5].shape, jnp.float32),
    )


def refine_linear_transpose_lowering(ctx, *args, iv_shape=None):
    return jax.ffi.ffi_lowering("hugegp_cuda_refine_linear_transpose_bind")(ctx, *args)


def refine_linear_transpose_value_and_jvp(primals, tangents, iv_shape=None):
    if any(type(t) is not ad.Zero for t in tangents[:5]):
        raise NotImplementedError(
            "Differentiation for refine_linear_transpose only supported for values."
        )
    if type(tangents[5]) is ad.Zero:
        raise NotImplementedError("Not differentiated with respect to values?")
    primals_out = refine_linear_transpose(*primals, iv_shape=iv_shape)
    tangents_out = refine_linear_transpose(*primals[:5], *tangents[5:], iv_shape=iv_shape)
    return primals_out, tangents_out


def refine_linear_transpose_transpose_rule(tangents_out, *primals, iv_shape=None):
    p, o, n, cb, cv, v = primals
    dv_buffer, div, dx = tangents_out

    if any(ad.is_undefined_primal(t) for t in [p, o, n, cb, cv]):
        raise NotImplementedError("Differentiation only supported for values.")

    if not ad.is_undefined_primal(v):
        raise NotImplementedError("Not differentiated with respect to values?")

    if any(type(t) is ad.Zero for t in [div, dx]):
        raise NotImplementedError("Not differentiated with respect to initial_values and xi?")

    dv = refine_linear(p, o, n, cb, cv, div, dx)
    return None, None, None, None, None, dv


def refine_linear_transpose_batch(vector_args, batch_axes, iv_shape=None):
    p, o, n, cb, cv, v = vector_args
    pa, oa, na, cba, cva, va = batch_axes

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

    if va is None:
        v = jnp.broadcast_to(v, (batch_size,) + v.shape)
    else:
        v = batching.moveaxis(v, va, 0)
    batched_iv_shape = (batch_size,) + iv_shape

    # call the primitive with the batched arguments
    return refine_linear_transpose(p, o, n, cb, cv, v, iv_shape=batched_iv_shape), (0, 0, 0)


# ============ query_coarse_neighbors primitive =============


def query_coarse_neighbors(points, split_dims, k):
    call = jax.ffi.ffi_call(
        "hugegp_cuda_query_coarse_neighbors_bind",
        jax.ShapeDtypeStruct((len(points), k), jnp.uint32),
    )
    neighbors = call(points, split_dims)
    return neighbors
