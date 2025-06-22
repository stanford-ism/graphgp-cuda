import ctypes
from pathlib import Path

import jax
import jax.numpy as jnp

from jax import lax
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import mlir, ad

# Define custom primitive, the "refine" function is exposed
refine_p = Primitive("hugegp_cuda_refine")
refine_jvp_linear_p = Primitive("hugegp_cuda_refine_jvp_linear")


def _initialize():
    # Register CUDA bindings as FFI targets
    so_path = next(Path(__file__).parent.glob("libhugegp_cuda*"))
    hugegp_cuda_lib = ctypes.cdll.LoadLibrary(str(so_path))
    jax.ffi.register_ffi_target(
        "hugegp_cuda_refine",
        jax.ffi.pycapsule(hugegp_cuda_lib.refine_bind),
        platform="gpu",
    )
    jax.ffi.register_ffi_target(
        "hugegp_cuda_refine_jvp_linear",
        jax.ffi.pycapsule(hugegp_cuda_lib.refine_jvp_linear_bind),
        platform="gpu",
    )
    jax.ffi.register_ffi_target(
        "hugegp_cuda_refine_vjp_linear",
        jax.ffi.pycapsule(hugegp_cuda_lib.refine_vjp_linear_bind),
        platform="gpu",
    )
    jax.ffi.register_ffi_target(
        "hugegp_cuda_query_coarse_neighbors",
        jax.ffi.pycapsule(hugegp_cuda_lib.query_coarse_neighbors_bind),
        platform="gpu",
    )

    # Register required functions for
    refine_p.def_impl(refine_impl)
    refine_p.def_abstract_eval(refine_abstract_eval)
    mlir.register_lowering(refine_p, refine_lowering, platform="gpu")  # type: ignore

    refine_jvp_linear_p.def_impl(refine_jvp_linear_impl)
    refine_jvp_linear_p.def_abstract_eval(refine_jvp_linear_abstract_eval)
    mlir.register_lowering(refine_jvp_linear_p, refine_jvp_linear_lowering, platform="gpu")  # type: ignore

    ad.primitive_jvps[refine_p] = refine_value_and_jvp
    ad.primitive_transposes[refine_p] = refine_transpose


def refine(points, offsets, neighbors, cov_bins, cov_vals, initial_values, xi):
    return refine_p.bind(points, offsets, neighbors, cov_bins, cov_vals, initial_values, xi)


def refine_impl(*args):
    values = jax.ffi.ffi_call(
        "hugegp_cuda_refine",
        jax.ShapeDtypeStruct(args[6].shape, jnp.float32),
    )(*args)
    return values


def refine_abstract_eval(*args):
    return ShapedArray(args[6].shape, dtype=jnp.float32)


def refine_lowering(ctx, *args):
    return jax.ffi.ffi_lowering("hugegp_cuda_refine")(ctx, *args)


def refine_value_and_jvp(primals, tangents):
    p, o, n, cb, cv, iv, x = primals
    dp, do, dn, dcb, dcv, div, dx = tangents

    if not all(type(t) is ad.Zero for t in [dp, do, dn, dcb]):
        raise NotImplementedError(
            "Differentiation not supported for points, offsets, neighbors, or cov_bins."
        )

    if type(dcv) is ad.Zero:
        if type(div) is ad.Zero:
            div = lax.zeros_like_array(iv)
        if type(dx) is ad.Zero:
            dx = lax.zeros_like_array(x)

        # primals_out, tangents_out = refine_jvp_linear(p, o, n, cb, cv, iv, x, div, dx)
        primals_out = refine(p, o, n, cb, cv, iv, x)
        tangents_out = refine(p, o, n, cb, cv, div, dx)
        return primals_out, tangents_out

    # else if type(div) is ad.Zero and type(dx) is ad.Zero:
    #     tangents_out =


def refine_transpose(tangents_out, *primals):
    p, o, n, cb, cv, iv, x = primals
    dv = tangents_out

    if any(ad.is_undefined_primal(t) for t in [p, o, n, cb, cv]):
        raise NotImplementedError("Differentiation only supported for xi and initial_values.")

    if not all(ad.is_undefined_primal(t) for t in [x, iv]):
        raise NotImplementedError("Not differentiated with respect to xi and initial_values?")

    div, dx = jax.ffi.ffi_call(
        "hugegp_cuda_refine_vjp_linear",
        (
            jax.ShapeDtypeStruct(
                (len(p),), jnp.float32
            ),  # div is used as a temporary buffer, then we slice
            jax.ShapeDtypeStruct((len(p),), jnp.float32),
        ),
    )(p, o, n, cb, cv, dv)
    div = div[: o[0]]
    return None, None, None, None, None, div, dx


def refine_jvp_linear(*args):
    return refine_jvp_linear_p.bind(*args)


def refine_jvp_linear_impl(*args):
    values, values_tangent = jax.ffi.ffi_call(
        "hugegp_cuda_refine_jvp_linear",
        (
            jax.ShapeDtypeStruct(args[6].shape, jnp.float32),
            jax.ShapeDtypeStruct(args[6].shape, jnp.float32),
        ),
    )(*args)
    return values, values_tangent


def refine_jvp_linear_abstract_eval(*args):
    return (
        ShapedArray(args[6].shape, dtype=jnp.float32),
        ShapedArray(args[6].shape, dtype=jnp.float32),
    )


def refine_jvp_linear_lowering(ctx, *args):
    return jax.ffi.ffi_lowering("hugegp_cuda_refine_jvp_linear")(ctx, *args)


def query_coarse_neighbors(points, split_dims, k):
    call = jax.ffi.ffi_call(
        "hugegp_cuda_query_coarse_neighbors",
        jax.ShapeDtypeStruct((len(points), k), jnp.uint32),
    )
    neighbors = call(points, split_dims)
    return neighbors
