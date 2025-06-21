import ctypes
from pathlib import Path

import jax
import jax.numpy as jnp

from jax import lax
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import mlir, ad

import numpy as np

# Define custom primitive, the "refine" function is exposed
refine_p = Primitive("hugegp_cuda_refine")

def _initialize():
    # Register CUDA bindings as FFI targets
    so_path = next(Path(__file__).parent.glob("libhugegp_cuda*"))
    hugegp_cuda_lib = ctypes.cdll.LoadLibrary(str(so_path))
    jax.ffi.register_ffi_target(
        "hugegp_cuda_refine_xla",
        jax.ffi.pycapsule(hugegp_cuda_lib.refine_xla),
        platform="gpu",
    )
    jax.ffi.register_ffi_target(
        "hugegp_cuda_refine_transpose_xla",
        jax.ffi.pycapsule(hugegp_cuda_lib.refine_transpose_xla),
        platform="gpu",
    )
    jax.ffi.register_ffi_target(
        "hugegp_cuda_query_coarse_neighbors_xla",
        jax.ffi.pycapsule(hugegp_cuda_lib.query_coarse_neighbors_xla),
        platform="gpu",
    )

    # Register required functions for primitive
    refine_p.def_impl(refine_impl)
    refine_p.def_abstract_eval(refine_abstract_eval)
    mlir.register_lowering(refine_p, refine_lowering, platform="gpu")  # type: ignore
    ad.primitive_jvps[refine_p] = refine_value_and_jvp
    ad.primitive_transposes[refine_p] = refine_transpose

def refine(points, offsets, neighbors, cov_bins, cov_vals, initial_values, xi):
    return refine_p.bind(points, offsets, neighbors, cov_bins, cov_vals, initial_values, xi)

def refine_impl(*args):
    values = jax.ffi.ffi_call(
        "hugegp_cuda_refine_xla",
        jax.ShapeDtypeStruct(args[6].shape, jnp.float32),
    )(*args)
    return values


def refine_abstract_eval(*args):
    return ShapedArray(args[6].shape, dtype=jnp.float32)


def refine_lowering(ctx, *args):
    return jax.ffi.ffi_lowering("hugegp_cuda_refine_xla")(ctx, *args)


def refine_value_and_jvp(primals, tangents):
    p, o, n, cb, cv, iv, x = primals
    dp, do, dn, dcb, dcv, div, dx = tangents

    if not all(type(t) is ad.Zero for t in [dp, dn, do, dcb, dcv]):
        raise NotImplementedError("Differentiation only supported for xi and initial_values.")

    if type(div) is ad.Zero:
        div = lax.zeros_like_array(iv)
    if type(dx) is ad.Zero:
        dx = lax.zeros_like_array(x)

    primals_out = refine(*primals)
    tangents_out = refine(p, o, n, cb, cv, div, dx)
    return primals_out, tangents_out


def refine_transpose(tangents_out, *primals):
    p, o, n, cb, cv, iv, x = primals
    dv = tangents_out

    if any(ad.is_undefined_primal(t) for t in [p, o, n, cb, cv]):
        raise NotImplementedError("Differentiation only supported for xi and initial_values.")

    if not all(ad.is_undefined_primal(t) for t in [x, iv]):
        raise NotImplementedError("Not differentiated with respect to xi and initial_values?")

    div, dx = jax.ffi.ffi_call(
        "hugegp_cuda_refine_transpose_xla",
        (
            jax.ShapeDtypeStruct((len(p),), jnp.float32), # div is used as a temporary buffer, then we slice
            jax.ShapeDtypeStruct((len(p),), jnp.float32),
        ),
    )(p, o, n, cb, cv, dv)
    div = div[:o[0]]
    return None, None, None, None, None, div, dx


def query_coarse_neighbors(points, split_dims, k):
    call = jax.ffi.ffi_call(
        "hugegp_cuda_query_coarse_neighbors_xla",
        jax.ShapeDtypeStruct((len(points), k), jnp.uint32),
    )
    neighbors = call(points, split_dims)
    return neighbors
