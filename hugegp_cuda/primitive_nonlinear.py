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
refine_linear_p = Primitive("hugegp_cuda_refine_nonlinear")
refine_linear_transpose_p = Primitive("hugegp_cuda_refine_nonlinear_jvp")


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



# ================ refine_nonlinear primitive ================

def refine_nonlinear(points, offsets, neighbors, cov_bins, cov_vals, initial_values, xi):
    return refine_linear_p.bind(points, offsets, neighbors, cov_bins, cov_vals, initial_values, xi)

def refine_nonlinear_impl(*args):
    return jax.ffi.ffi_call(
        "hugegp_cuda_refine_bind", jax.ShapeDtypeStruct(args[6].shape, jnp.float32)
    )(*args)

def refine_nonlinear_abstract_eval(points, offsets, neighbors, cov_bins, cov_vals, initial_values, xi):
    return ShapedArray(initial_values.shape, jnp.float32)

def refine_nonlinear_lowering(ctx, *args):
    return jax.ffi.ffi_lowering("hugegp_cuda_refine_bind")(ctx, *args)

def refine_nonlinear_linearization(nzs, *primals):
    p, o, n, cb, cv, iv, x = primals

    if any(nsz[i] for i in [0, 1, 2, 3, 5, 6]):
        raise NotImplementedError("Differentiation not supported for points, offsets, neighbors, or cov_bins. Nonlinear expects initial_values and xi to be constant.")

    if not all(nsz[i] for i in [4]):
        raise NotImplementedError("Not differentiated with respect to cov_vals?")
    
    values = refine_nonlinear(p, o, n, cb, cv, iv, x)
    residuals = (p, o, n, cb, cv, x, values)
    
    def jvp_func(residuals, dcv):
        p, o, n, cb, cv, x, values = residuals
        return refine_nonlinear_jvp(p, o, n, cb, cv, x, values, dcv)
    
    return (
        values,
        (False,),
        residuals,
        jvp_func,
    )



def refine_nonlinear_jvp(points, offsets, neighbors, cov_bins, cov_vals, xi, values, cov_vals_tangent):
    call = jax.ffi.ffi_call(
        "hugegp_cuda_refine_nonlinear_jvp_bind",
        jax.ShapeDtypeStruct(values.shape, jnp.float32),
    )
    return call(points, offsets, neighbors, cov_bins, cov_vals, values, xi, cov_vals_tangent)


def refine_nonlinear_jvp_transpose_rule(tangents_out, *primals):
    p, o, n, cb, cv, x, v, dcv = primals
    dv = tangents_out

    if any(ad.is_undefined_primal(t) for t in [p, o, n, cb, cv, x, v]):
        raise NotImplementedError("Differentiation only supported for cov_vals.")
    
    assert ad.is_undefined_primal(dcv), "Not differentiated with respect to cov_vals_tangent?"

    div, dx = jax.ffi.ffi_call(
        "hugegp_cuda_refine_nonlinear_jvp_transpose_bind",
        (
            jax.ShapeDtypeStruct(dv.shape, jnp.float32),
            jax.ShapeDtypeStruct(cv.shape, jnp.float32),
        ),
    )(p, o, n, cb, cv, x, v, dcv)
    return None, None, None, None, None, None, None, None


# ============ query_coarse_neighbors primitive =============


def query_coarse_neighbors(points, split_dims, k):
    call = jax.ffi.ffi_call(
        "hugegp_cuda_query_coarse_neighbors_bind",
        jax.ShapeDtypeStruct((len(points), k), jnp.uint32),
    )
    neighbors = call(points, split_dims)
    return neighbors
