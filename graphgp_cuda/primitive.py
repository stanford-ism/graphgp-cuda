import ctypes
from pathlib import Path

import jax
import jax.numpy as jnp

import numpy as np

from jax import ShapeDtypeStruct
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import mlir, batching, ad
from jax import lax
from jax.tree_util import Partial

# NOTE: This file is verbose and repetitive. Effort was made to reduce boilerplate with automatic primitive
# registration, although this results in confusing code. Waiting for jax to make this process easier.

refine_p = Primitive("graphgp_cuda_refine")
refine_transpose_p = Primitive("graphgp_cuda_refine_transpose")
refine_jvp_p = Primitive("graphgp_cuda_refine_jvp")
refine_vjp_p = Primitive("graphgp_cuda_refine_vjp")
refine_inv_p = Primitive("graphgp_cuda_refine_inv")
refine_logdet_p = Primitive("graphgp_cuda_refine_logdet")

refine_p_64 = Primitive("graphgp_cuda_refine_64")
refine_transpose_p_64 = Primitive("graphgp_cuda_refine_transpose_64")
refine_jvp_p_64 = Primitive("graphgp_cuda_refine_jvp_64")
refine_vjp_p_64 = Primitive("graphgp_cuda_refine_vjp_64")
refine_inv_p_64 = Primitive("graphgp_cuda_refine_inv_64")
refine_logdet_p_64 = Primitive("graphgp_cuda_refine_logdet_64")


def refine(points, neighbors, offsets, cov_bins, cov_vals, initial_values, xi):
    casted_args = _cast_all(points, neighbors, offsets, cov_bins, cov_vals, initial_values, xi)
    if casted_args[0].dtype == jnp.float64:
        return refine_p_64.bind(*casted_args)
    else:
        return refine_p.bind(*casted_args)


def refine_inv(points, neighbors, offsets, cov_bins, cov_vals, values):
    casted_args = _cast_all(points, neighbors, offsets, cov_bins, cov_vals, values)
    if casted_args[0].dtype == jnp.float64:
        return refine_inv_p_64.bind(*casted_args)
    else:
        return refine_inv_p.bind(*casted_args)


def refine_logdet(points, neighbors, offsets, cov_bins, cov_vals):
    casted_args = _cast_all(points, neighbors, offsets, cov_bins, cov_vals)
    if casted_args[0].dtype == jnp.float64:
        return refine_logdet_p_64.bind(*casted_args)
    else:
        return refine_logdet_p.bind(*casted_args)


def _determine_dtype(*args):
    for x in args:
        if x.dtype in (jnp.int64, jnp.float64):
            return jnp.int64, jnp.float64, "_64"
    return jnp.int32, jnp.float32, ""


def _cast_all(*args):
    int_dtype, float_dtype, _ = _determine_dtype(*args)
    casted_args = []
    for x in args:
        if jnp.issubdtype(x.dtype, jnp.integer):
            casted_args.append(x.astype(int_dtype))
        else:
            casted_args.append(x.astype(float_dtype))
    return tuple(casted_args)


def initialize():
    try:
        so_path = next(Path(__file__).parent.glob("libgraphgp_cuda*"))
        lib = ctypes.cdll.LoadLibrary(str(so_path))
    except (StopIteration, OSError) as e:
        raise RuntimeError(f"Failed to load graphgp_cuda library: {e}")

    # Register all FFI names
    for name in [
        "refine",
        "refine_transpose",
        "refine_jvp",
        "refine_vjp",
        "refine_inv",
        "refine_logdet",
        "build_tree",
        "query_preceding_neighbors",
        "query_neighbors",
        "compute_depths_parallel",
        "order_by_depth",
        "build_graph",
        "sort",
        "sort_three",
    ]:
        jax.ffi.register_ffi_target(
            f"graphgp_cuda_{name}_ffi",
            jax.ffi.pycapsule(getattr(lib, f"{name}_ffi")),
            platform="CUDA",
        )
        jax.ffi.register_ffi_target(
            f"graphgp_cuda_{name}_ffi_64",
            jax.ffi.pycapsule(getattr(lib, f"{name}_ffi_64")),
            platform="CUDA",
        )

    # Define abstract evaluations, these essentially serve as the definitions of the FFI functions in Python

    def refine_abstract_eval(
        points, neighbors, offsets, cov_bins, cov_vals, initial_values, xi, dtype=jnp.float32
    ):
        n = points.shape[0]
        batch_shape = initial_values.shape[:-1]
        return ShapedArray(batch_shape + (n,), dtype)

    def refine_transpose_abstract_eval(
        points, neighbors, offsets, cov_bins, cov_vals, values, dtype=jnp.float32
    ):
        n = points.shape[0]
        n0 = n - neighbors.shape[0]
        batch_shape = values.shape[:-1]
        return (
            ShapedArray(batch_shape + (n0,), dtype),
            ShapedArray(batch_shape + (n - n0,), dtype),
            ShapedArray(batch_shape + (n,), dtype),
        )

    def refine_jvp_abstract_eval(
        points,
        neighbors,
        offsets,
        cov_bins,
        cov_vals,
        initial_values,
        xi,
        cov_vals_tangent,
        initial_values_tangent,
        xi_tangent,
        dtype=jnp.float32,
    ):
        n = points.shape[0]
        batch_shape = initial_values.shape[:-1]
        return (
            ShapedArray(batch_shape + (n,), dtype),
            ShapedArray(batch_shape + (n,), dtype),
        )

    def refine_vjp_abstract_eval(
        points,
        neighbors,
        offsets,
        cov_bins,
        cov_vals,
        initial_values,
        xi,
        values,
        values_tangent,
        dtype=jnp.float32,
    ):
        return (
            ShapedArray(cov_vals.shape, dtype),
            ShapedArray(initial_values.shape, dtype),
            ShapedArray(xi.shape, dtype),
            ShapedArray(values.shape, dtype),
        )

    def refine_inv_abstract_eval(
        points, neighbors, offsets, cov_bins, cov_vals, values, dtype=jnp.float32
    ):
        n = points.shape[0]
        n0 = n - neighbors.shape[0]
        batch_shape = values.shape[:-1]
        return (
            ShapedArray(batch_shape + (n0,), dtype),  # initial_values
            ShapedArray(batch_shape + (n - n0,), dtype),  # xi
        )

    def refine_logdet_abstract_eval(
        points, neighbors, offsets, cov_bins, cov_vals, dtype=jnp.float32
    ):
        batch_shape = cov_vals.shape[:-1]
        return ShapedArray(batch_shape, dtype)  # logdet

    # Automatically set up all primitives
    setup_ffi_primitive(
        refine_p,
        "graphgp_cuda_refine_ffi",
        refine_abstract_eval,
        batch_args=(4, 5, 6),
        transpose_prim=refine_transpose_p,
        jvp_prim=refine_jvp_p,
        n_nonlinear=5,
        n_linearized=1,
        n_transpose_buffer=1,
        platform="gpu",
    )

    setup_ffi_primitive(
        refine_transpose_p,
        "graphgp_cuda_refine_transpose_ffi",
        refine_transpose_abstract_eval,
        batch_args=(4, 5),
        transpose_prim=refine_p,
        n_nonlinear=5,
        n_buffer=1,
        platform="gpu",
    )
    refine_transpose_p.multiple_results = True

    setup_ffi_primitive(
        refine_jvp_p,
        "graphgp_cuda_refine_jvp_ffi",
        refine_jvp_abstract_eval,
        batch_args=(4, 5, 6, 7, 8, 9),
        transpose_prim=None, # TODO: refine_vjp_p, but need to handle values correctly!
        n_nonlinear=7,
        n_transpose_buffer=1,
        platform="gpu",
    )
    refine_jvp_p.multiple_results = True

    setup_ffi_primitive(
        refine_vjp_p,
        "graphgp_cuda_refine_vjp_ffi",
        refine_vjp_abstract_eval,
        batch_args=(4, 5, 6, 7, 8),
        transpose_prim=refine_jvp_p,
        n_nonlinear=7,
        n_buffer=1,
        platform="gpu",
    )
    refine_vjp_p.multiple_results = True

    setup_ffi_primitive(
        refine_inv_p,
        "graphgp_cuda_refine_inv_ffi",
        refine_inv_abstract_eval,
        batch_args=(4, 5),
        n_nonlinear=5,
        platform="gpu",
    )
    refine_inv_p.multiple_results = True

    setup_ffi_primitive(
        refine_logdet_p,
        "graphgp_cuda_refine_logdet_ffi",
        refine_logdet_abstract_eval,
        batch_args=(4,),
        n_nonlinear=5,
        platform="gpu",
    )

    # ==================== Set up 64 bit versions ====================
    setup_ffi_primitive(
        refine_p_64,
        "graphgp_cuda_refine_ffi_64",
        Partial(refine_abstract_eval, dtype=jnp.float64),
        batch_args=(4, 5, 6),
        transpose_prim=refine_transpose_p_64,
        jvp_prim=refine_jvp_p_64,
        n_nonlinear=5,
        n_linearized=1,
        n_transpose_buffer=1,
        platform="gpu",
    )

    setup_ffi_primitive(
        refine_transpose_p_64,
        "graphgp_cuda_refine_transpose_ffi_64",
        Partial(refine_transpose_abstract_eval, dtype=jnp.float64),
        batch_args=(4, 5),
        transpose_prim=refine_p_64,
        n_nonlinear=5,
        n_buffer=1,
        platform="gpu",
    )
    refine_transpose_p_64.multiple_results = True

    setup_ffi_primitive(
        refine_jvp_p_64,
        "graphgp_cuda_refine_jvp_ffi_64",
        Partial(refine_jvp_abstract_eval, dtype=jnp.float64),
        batch_args=(4, 5, 6, 7, 8, 9),
        transpose_prim=None, # TODO: refine_vjp_p_64, but need to handle values correctly!
        n_nonlinear=7,
        n_transpose_buffer=1,
        platform="gpu",
    )
    refine_jvp_p_64.multiple_results = True

    setup_ffi_primitive(
        refine_vjp_p_64,
        "graphgp_cuda_refine_vjp_ffi_64",
        Partial(refine_vjp_abstract_eval, dtype=jnp.float64),
        batch_args=(4, 5, 6, 7, 8),
        transpose_prim=refine_jvp_p_64,
        n_nonlinear=7,
        n_buffer=1,
        platform="gpu",
    )
    refine_vjp_p_64.multiple_results = True

    setup_ffi_primitive(
        refine_inv_p_64,
        "graphgp_cuda_refine_inv_ffi_64",
        Partial(refine_inv_abstract_eval, dtype=jnp.float64),
        batch_args=(4, 5),
        n_nonlinear=5,
        platform="gpu",
    )
    refine_inv_p_64.multiple_results = True

    setup_ffi_primitive(
        refine_logdet_p_64,
        "graphgp_cuda_refine_logdet_ffi_64",
        Partial(refine_logdet_abstract_eval, dtype=jnp.float64),
        batch_args=(4,),
        n_nonlinear=5,
        platform="gpu",
    )


# ---------------- Graph construction (not differentiable) ----------------


@jax.jit
def build_tree(points):
    int_dtype, float_dtype, type_suffix = _determine_dtype(points)
    call = jax.ffi.ffi_call(
        "graphgp_cuda_build_tree_ffi" + type_suffix,
        (
            jax.ShapeDtypeStruct(points.shape, float_dtype),
            jax.ShapeDtypeStruct((points.shape[0],), int_dtype),
            jax.ShapeDtypeStruct((points.shape[0],), int_dtype),
            jax.ShapeDtypeStruct((points.shape[0],), int_dtype),
            jax.ShapeDtypeStruct((points.shape[0],), float_dtype),
        ),
    )
    points, split_dims, indices, tags, ranges = call(*_cast_all(points))
    return points, split_dims, indices


@Partial(jax.jit, static_argnames=("n0", "k"))
def query_preceding_neighbors(points, split_dims, *, n0, k):
    int_dtype, float_dtype, type_suffix = _determine_dtype(points)
    call = jax.ffi.ffi_call(
        "graphgp_cuda_query_preceding_neighbors_ffi" + type_suffix,
        jax.ShapeDtypeStruct((points.shape[0] - n0, k), int_dtype),
    )
    neighbors = call(*_cast_all(points, split_dims))
    return neighbors


@Partial(jax.jit, static_argnames=("k"))
def query_neighbors(points, split_dims, query_indices, max_indices, *, k):
    int_dtype, float_dtype, type_suffix = _determine_dtype(
        points, split_dims, query_indices, max_indices
    )
    call = jax.ffi.ffi_call(
        "graphgp_cuda_query_neighbors_ffi" + type_suffix,
        jax.ShapeDtypeStruct((query_indices.shape[0], k), int_dtype),
    )
    neighbors = call(*_cast_all(points, split_dims, query_indices, max_indices))
    return neighbors


@Partial(jax.jit, static_argnames="n0")
def compute_depths_parallel(neighbors, *, n0):
    int_dtype, _, type_suffix = _determine_dtype(neighbors)
    call = jax.ffi.ffi_call(
        "graphgp_cuda_compute_depths_parallel_ffi" + type_suffix,
        (
            jax.ShapeDtypeStruct((neighbors.shape[0] + n0,), int_dtype),
            jax.ShapeDtypeStruct((neighbors.shape[0] + n0,), int_dtype),
        ),
    )
    depths, temp = call(*_cast_all(neighbors))
    return depths


@jax.jit
def order_by_depth(points, indices, neighbors, depths):
    int_dtype, float_dtype, type_suffix = _determine_dtype(points, indices, neighbors, depths)
    call = jax.ffi.ffi_call(
        "graphgp_cuda_order_by_depth_ffi" + type_suffix,
        (
            jax.ShapeDtypeStruct(points.shape, float_dtype),
            jax.ShapeDtypeStruct(indices.shape, int_dtype),
            jax.ShapeDtypeStruct(neighbors.shape, int_dtype),
            jax.ShapeDtypeStruct(depths.shape, int_dtype),
            jax.ShapeDtypeStruct((2 * points.shape[0],), int_dtype),
        ),
    )
    points, indices, neighbors, depths, _ = call(*_cast_all(points, indices, neighbors, depths))
    return points, indices, neighbors, depths


@Partial(jax.jit, static_argnames=("n0", "k"))
def build_graph(points, *, n0, k):
    int_dtype, float_dtype, type_suffix = _determine_dtype(points)
    call = jax.ffi.ffi_call(
        "graphgp_cuda_build_graph_ffi" + type_suffix,
        (
            jax.ShapeDtypeStruct(points.shape, float_dtype),
            jax.ShapeDtypeStruct((points.shape[0],), int_dtype),
            jax.ShapeDtypeStruct((points.shape[0] - n0, k), int_dtype),
            jax.ShapeDtypeStruct((points.shape[0],), int_dtype),
            jax.ShapeDtypeStruct((2 * points.shape[0],), int_dtype),
        ),
    )
    points, indices, neighbors, depths, _ = call(*_cast_all(points))
    return points, indices, neighbors, depths


# ================= Helper functions for custom primitives ====================


# Call FFI assuming same outputs as abstract_eval
def make_ffi_impl(ffi_name, abstract_eval):
    def impl(*args):
        abstract_out = abstract_eval(*args)
        if isinstance(abstract_out, ShapedArray):
            abstract_out = ShapeDtypeStruct(abstract_out.shape, abstract_out.dtype)
        elif isinstance(abstract_out, tuple):
            abstract_out = tuple(ShapeDtypeStruct(s.shape, s.dtype) for s in abstract_out)
        return jax.ffi.ffi_call(
            ffi_name,
            abstract_out,
        )(*args)

    return impl


# Put batch dimensions along leading axis, if argument is batchable
def make_batching_rule(prim, batch_args):
    def batching_rule(vector_args, batch_axes):
        bad_batch_args = tuple(
            j
            for j, bd in enumerate(batch_axes)
            if j not in batch_args and bd is not batching.not_mapped
        )
        if len(bad_batch_args) > 0:
            raise NotImplementedError(
                f"Batching {prim} not supported for arguments {bad_batch_args}."
            )
        (size,) = {
            x.shape[bd] for x, bd in zip(vector_args, batch_axes) if bd is not batching.not_mapped
        }
        args = []
        for j, (x, bd) in enumerate(zip(vector_args, batch_axes)):
            if bd is batching.not_mapped:
                if j in batch_args:
                    args.append(jnp.broadcast_to(x[None], (size,) + x.shape))
                else:
                    args.append(x)
            else:
                args.append(batching.bdim_at_front(x, bd, size))
        out = prim.bind(*args)
        if prim.multiple_results:
            return out, (0,) * len(out)
        else:
            return out, 0

    return batching_rule


# Assumes all nonlinear args first, and buffers are at the end of outputs
def make_transpose_rule(transpose_prim, *, n_nonlinear=0, n_buffer=0, n_transpose_buffer=0):
    def transpose_rule(tangents_out, *primals):
        if transpose_prim is None:
            raise NotImplementedError("Transpose rule not provided for this set of arguments.")
        if any(ad.is_undefined_primal(t) for t in primals[:n_nonlinear]):
            raise ValueError("Transposition only valid for linear arguments.")
        # if not all(ad.is_undefined_primal(t) for t in primals[n_nonlinear:]):
        #     raise ValueError("Not all linear arguments were undefined primals?")
        if type(tangents_out) is list:  # TODO: why is this a list?
            tangents_out = tuple(tangents_out)
        if type(tangents_out) is not tuple:
            tangents_out = (tangents_out,)
        if all(type(t) is ad.Zero for t in tangents_out):
            return (ad.Zero,) * len(primals) if len(primals) > 1 else ad.Zero
        if any(type(t) is ad.Zero for t in tangents_out[: len(tangents_out) - n_buffer]):
            tangents_out = tuple(
                tan if type(tan) is not ad.Zero else jnp.zeros_like(primal)
                for primal, tan in zip(primals, tangents_out)
            )
            # print(tangents_out)
            # raise NotImplementedError("Output tangents were mix of Zero and non-Zero.")
            # TODO: this can arise in jacrev, easy to handle
        tangents_in = transpose_prim.bind(
            *primals[:n_nonlinear], *tangents_out[: len(tangents_out) - n_buffer]
        )
        if type(tangents_in) is list:  # TODO: why is this a list?
            tangents_in = tuple(tangents_in)
        if type(tangents_in) is not tuple:
            tangents_in = (tangents_in,)
        tangents_in = (None,) * n_nonlinear + tangents_in[: len(tangents_in) - n_transpose_buffer]
        tangents_in = tuple(
            tangents_in[i] if ad.is_undefined_primal(primals[i]) else None
            for i in range(len(primals))
        )
        # TODO: convert back to scalar if single output?
        return tangents_in

    return transpose_rule


# Assumes all nonlinear args first, with linearized args at the end of this set
# JVP function must take all primals and tangents corresponding to linearized and linear args
# Only allows linear + 1 linearization JVP, in general could have multiple linearizations
def make_jvp_rule(prim, jvp_prim, *, n_nonlinear=0, n_linearized=0):
    def jvp_rule(primals, tangents):
        primals_out = prim.bind(*primals)

        if all(type(t) is ad.Zero for t in tangents):
            return primals_out, (ad.Zero,) * len(primals)

        # linear case
        if all(type(t) is ad.Zero for t in tangents[:n_nonlinear]):
            tangents = tuple(
                tan if type(tan) is not ad.Zero else jnp.zeros_like(primal)
                for primal, tan in zip(primals[n_nonlinear:], tangents[n_nonlinear:])
            )
            tangents_out = prim.bind(*primals[:n_nonlinear], *tangents)

        # nonlinear case
        else:
            n_fixed = n_nonlinear - n_linearized
            if not all(type(t) is ad.Zero for t in tangents[:n_fixed]):
                bad_args = tuple(
                    j for j, t in enumerate(tangents) if j < n_fixed and type(t) is not ad.Zero
                )
                raise NotImplementedError(
                    f"Differentiation not supported for arguments {bad_args}."
                )
            if jvp_prim is None:
                raise NotImplementedError("JVP rule not provided for this set of arguments.")
            tangents = tuple(
                tan if type(tan) is not ad.Zero else jnp.zeros_like(primal)
                for primal, tan in zip(primals[n_fixed:], tangents[n_fixed:])
            )
            _, tangents_out = jvp_prim.bind(
                *primals, *tangents
            )  # TODO: do primals on same forward pass if all concrete

        return primals_out, tangents_out

    return jvp_rule


# Convenience wrapper to set up an FFI primitive with configurable rules
# See individual functions for meaning of optional arguments
def setup_ffi_primitive(
    prim,
    ffi_name,
    abstract_eval,
    *,
    batch_args=None,
    transpose_prim=None,
    jvp_prim=None,
    n_nonlinear=0,
    n_linearized=0,
    n_buffer=0,
    n_transpose_buffer=0,
    platform="gpu",
):
    prim.def_impl(make_ffi_impl(ffi_name, abstract_eval))
    prim.def_abstract_eval(abstract_eval)
    if batch_args is not None:
        batching.primitive_batchers[prim] = make_batching_rule(prim, batch_args)
    if transpose_prim is not None:
        ad.primitive_transposes[prim] = make_transpose_rule(
            transpose_prim,
            n_nonlinear=n_nonlinear,
            n_buffer=n_buffer,
            n_transpose_buffer=n_transpose_buffer,
        )
    ad.primitive_jvps[prim] = make_jvp_rule(
        prim, jvp_prim, n_nonlinear=n_nonlinear, n_linearized=n_linearized
    )
    mlir.register_lowering(prim, jax.ffi.ffi_lowering(ffi_name), platform=platform)
    return prim, prim.bind
