import ctypes
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.tree_util import Partial

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

def refine(points, xi, neighbors, level_offsets, initial_values, cov_r, cov):
    return refine_p.bind(points, xi, neighbors, level_offsets, initial_values, cov_r, cov)

def refine_impl(points, xi, neighbors, level_offsets, initial_values, cov_r, cov):
    values = jax.ffi.ffi_call(
        "hugegp_cuda_refine_xla",
        jax.ShapeDtypeStruct(xi.shape, jnp.float32),  # xi is the second argument
    )(points, xi, neighbors, level_offsets, initial_values, cov_r, cov)
    return values


def refine_abstract_eval(*args):
    return ShapedArray(args[1].shape, dtype=jnp.float32)  # xi is the second argument


def refine_lowering(ctx, *args):
    return jax.ffi.ffi_lowering("hugegp_cuda_refine_xla")(ctx, *args)


def refine_value_and_jvp(primals, tangents):
    p, x, n, lo, iv, cr, c = primals
    dp, dx, dn, dlo, div, dcr, dc = tangents

    if not all(type(t) is ad.Zero for t in [dp, dn, dlo, dcr, dc]):
        raise NotImplementedError("Differentiation only supported for xi and initial_values.")

    if type(dx) is ad.Zero:
        dx = lax.zeros_like_array(x)
    if type(div) is ad.Zero:
        div = lax.zeros_like_array(iv)

    primals_out = refine(*primals)
    tangents_out = refine(p, dx, n, lo, div, cr, c)
    return primals_out, tangents_out


def refine_transpose(tangents_out, *primals):
    p, x, n, lo, iv, cr, c = primals
    dv = tangents_out

    if any(ad.is_undefined_primal(t) for t in [p, n, lo, cr, c]):
        raise NotImplementedError("Differentiation only supported for xi and initial_values.")

    if not all(ad.is_undefined_primal(t) for t in [x, iv]):
        raise NotImplementedError("Not differentiated with respect to xi and initial_values?")

    dx, div = jax.ffi.ffi_call(
        "hugegp_cuda_refine_transpose_xla",
        (
            jax.ShapeDtypeStruct((len(p),), jnp.float32),
            jax.ShapeDtypeStruct((lo[0],), jnp.float32),
        ),
    )(p, n, lo, cr, c, dv)
    return None, dx, None, None, div, None, None


def query_coarse_neighbors(points, split_dims, k):
    call = jax.ffi.ffi_call(
        "hugegp_cuda_query_coarse_neighbors_xla",
        jax.ShapeDtypeStruct((len(points), k), jnp.uint32),
    )
    neighbors = call(points, split_dims, k=np.int32(k))
    return neighbors
