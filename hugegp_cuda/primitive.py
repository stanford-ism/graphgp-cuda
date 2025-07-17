import ctypes
from pathlib import Path

import jax
import jax.numpy as jnp

import numpy as np

from jax import lax
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import mlir, batching, ad


# Define custom primitive, the "refine" function is exposed
refine_linear_p = Primitive("hugegp_cuda_refine_linear")
refine_linear_transpose_p = Primitive("hugegp_cuda_refine_linear_transpose")


def _initialize():
    # Register CUDA bindings as FFI targets
    so_path = next(Path(__file__).parent.glob("libhugegp_cuda*"))
    hugegp_cuda_lib = ctypes.cdll.LoadLibrary(str(so_path))
    jax.ffi.register_ffi_target(
        "hugegp_cuda_refine_ffi",
        jax.ffi.pycapsule(hugegp_cuda_lib.refine_ffi),
        platform="gpu",
    )
    jax.ffi.register_ffi_target(
        "hugegp_cuda_refine_linear_transpose_ffi",
        jax.ffi.pycapsule(hugegp_cuda_lib.refine_linear_transpose_ffi),
        platform="gpu",
    )
    jax.ffi.register_ffi_target(
        "hugegp_cuda_query_preceding_neighbors_ffi",
        jax.ffi.pycapsule(hugegp_cuda_lib.query_preceding_neighbors_ffi),
        platform="gpu",
    )
    jax.ffi.register_ffi_target(
        "hugegp_cuda_query_preceding_neighbors_alt_ffi",
        jax.ffi.pycapsule(hugegp_cuda_lib.query_preceding_neighbors_alt_ffi),
        platform="gpu",
    )
    jax.ffi.register_ffi_target(
        "hugegp_cuda_compute_levels_ffi",
        jax.ffi.pycapsule(hugegp_cuda_lib.compute_levels_ffi),
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


# ========== exposed functions ==========


def compute_levels(neighbors, *, n_initial):
    call = jax.ffi.ffi_call(
        "hugegp_cuda_compute_levels_ffi",
        jax.ShapeDtypeStruct((len(neighbors),), jnp.uint32),
    )
    levels = call(neighbors, n_initial=np.uint32(n_initial))
    return levels


def query_preceding_neighbors(points, split_dims, k):
    call = jax.ffi.ffi_call(
        "hugegp_cuda_query_preceding_neighbors_ffi",
        jax.ShapeDtypeStruct((len(points), k), jnp.uint32),
    )
    neighbors = call(points, split_dims)
    return neighbors


def query_preceding_neighbors_alt(points, split_dims, k):
    call = jax.ffi.ffi_call(
        "hugegp_cuda_query_preceding_neighbors_alt_ffi",
        jax.ShapeDtypeStruct((len(points), k), jnp.uint32),
    )
    neighbors = call(points, split_dims)
    return neighbors


def refine(points, neighbors, offsets, cov_bins, cov_vals, initial_values, xi):
    return refine_linear(points, neighbors, offsets, cov_bins, cov_vals, initial_values, xi)


# ========== refine_linear primitive ==========


def refine_linear(points, neighbors, offsets, cov_bins, cov_vals, initial_values, xi):
    return refine_linear_p.bind(points, neighbors, offsets, cov_bins, cov_vals, initial_values, xi)


def refine_linear_impl(*args):
    return jax.ffi.ffi_call(
        "hugegp_cuda_refine_ffi",
        jax.ShapeDtypeStruct(
            args[6].shape[:-1] + (args[0].shape[0],), jnp.float32
        ),
    )(*args)


def refine_linear_abstract_eval(*args):
    return ShapedArray(args[6].shape[:-1] + (args[0].shape[0],), jnp.float32)


def refine_linear_lowering(ctx, *args):
    return jax.ffi.ffi_lowering("hugegp_cuda_refine_ffi")(ctx, *args)


def refine_linear_value_and_jvp(primals, tangents):
    if any(type(t) is not ad.Zero for t in tangents[:5]):
        raise NotImplementedError("Differentiation for refine_linear only supported for xi.")
    if any(type(t) is ad.Zero for t in tangents[5:]):
        raise NotImplementedError("Not differentiated with respect to xi?")
    primals_out = refine_linear(*primals)
    tangents_out = refine_linear(*primals[:5], *tangents[5:])
    return primals_out, tangents_out


def refine_linear_transpose_rule(tangents_out, *primals):
    p, n, o, cb, cv, iv, x = primals
    dv = tangents_out

    if any(ad.is_undefined_primal(t) for t in [p, n, o, cb, cv]):
        raise NotImplementedError("Differentiation only supported for initial_values and xi.")

    if not all(ad.is_undefined_primal(t) for t in [iv, x]):
        raise NotImplementedError("Not differentiated with respect to initial_values and xi?")

    if type(dv) is ad.Zero:
        raise NotImplementedError("Not differentiated with respect to values?")

    dv_buffer, div, dx = refine_linear_transpose(p, n, o, cb, cv, dv)
    return None, None, None, None, None, div, dx


def refine_linear_batch(vector_args, batch_axes):
    p, n, o, cb, cv, iv, x = vector_args
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
    return refine_linear(p, n, o, cb, cv, iv, x), 0


# ========== refine_linear_transpose primitive ==========


def refine_linear_transpose(points, neighbors, offsets, cov_bins, cov_vals, values):
    return refine_linear_transpose_p.bind(
        points, neighbors, offsets, cov_bins, cov_vals, values
    )


def refine_linear_transpose_impl(*args):
    return jax.ffi.ffi_call(
        "hugegp_cuda_refine_linear_transpose_ffi",
        (
            jax.ShapeDtypeStruct(args[5].shape, jnp.float32),
            jax.ShapeDtypeStruct(args[5].shape[:-1] + (args[0].shape[0] - args[1].shape[0],), jnp.float32),
            jax.ShapeDtypeStruct(args[5].shape[:-1] + (args[1].shape[0],), jnp.float32),
        ),
    )(*args)


def refine_linear_transpose_abstract_eval(*args):
    return (
        ShapedArray(args[5].shape, jnp.float32),
        ShapedArray(args[5].shape[:-1] + (args[0].shape[0] - args[1].shape[0],), jnp.float32),
        ShapedArray(args[5].shape[:-1] + (args[1].shape[0],), jnp.float32),
    )


def refine_linear_transpose_lowering(ctx, *args):
    return jax.ffi.ffi_lowering("hugegp_cuda_refine_linear_transpose_ffi")(ctx, *args)


def refine_linear_transpose_value_and_jvp(primals, tangents):
    if any(type(t) is not ad.Zero for t in tangents[:6]):
        raise NotImplementedError(
            "Differentiation for refine_linear_transpose only supported for values."
        )
    if type(tangents[6]) is ad.Zero:
        raise NotImplementedError("Not differentiated with respect to values?")
    primals_out = refine_linear_transpose(*primals)
    tangents_out = refine_linear_transpose(*primals[:6], *tangents[6:])
    return primals_out, tangents_out


def refine_linear_transpose_transpose_rule(tangents_out, *primals):
    p, n, o, cb, cv, v = primals
    dv_buffer, div, dx = tangents_out

    if any(ad.is_undefined_primal(t) for t in [p, n, o, cb, cv]):
        raise NotImplementedError("Differentiation only supported for values.")

    if not ad.is_undefined_primal(v):
        raise NotImplementedError("Not differentiated with respect to values?")

    if any(type(t) is ad.Zero for t in [div, dx]):
        raise NotImplementedError("Not differentiated with respect to initial_values and xi?")

    dv = refine_linear(p, n, o, cb, cv, div, dx)
    return None, None, None, None, None, dv


def refine_linear_transpose_batch(vector_args, batch_axes):
    p, n, o, cb, cv, v = vector_args
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

    # call the primitive with the batched arguments
    return refine_linear_transpose(p, n, o, cb, cv, v), (0, 0)
