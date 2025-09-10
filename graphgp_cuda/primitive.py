import ctypes
from pathlib import Path

import jax
import jax.numpy as jnp

import numpy as np

from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import mlir, batching, ad
from jax import lax


# Define custom primitive, the "refine" function is exposed
refine_p = Primitive("graphgp_cuda_refine")
refine_linear_transpose_p = Primitive("graphgp_cuda_refine_linear_transpose")
refine_nonlinear_jvp_p = Primitive("graphgp_cuda_refine_nonlinear_jvp")
refine_nonlinear_vjp_p = Primitive("graphgp_cuda_refine_nonlinear_vjp")


def initialize():
    # Register CUDA bindings as FFI targets
    try:
        so_path = next(Path(__file__).parent.glob("libgraphgp_cuda*"))
        graphgp_cuda_lib = ctypes.cdll.LoadLibrary(str(so_path))
    except (StopIteration, OSError) as e:
        raise RuntimeError(f"Failed to load graphgp_cuda library: {e}")

    jax.ffi.register_ffi_target(
        "graphgp_cuda_refine_ffi",
        jax.ffi.pycapsule(graphgp_cuda_lib.refine_ffi),
        platform="gpu",
    )
    jax.ffi.register_ffi_target(
        "graphgp_cuda_refine_linear_transpose_ffi",
        jax.ffi.pycapsule(graphgp_cuda_lib.refine_linear_transpose_ffi),
        platform="gpu",
    )
    jax.ffi.register_ffi_target(
        "graphgp_cuda_refine_nonlinear_jvp_ffi",
        jax.ffi.pycapsule(graphgp_cuda_lib.refine_nonlinear_jvp_ffi),
        platform="gpu",
    )
    jax.ffi.register_ffi_target(
        "graphgp_cuda_refine_nonlinear_vjp_ffi",
        jax.ffi.pycapsule(graphgp_cuda_lib.refine_nonlinear_vjp_ffi),
        platform="gpu",
    )
    jax.ffi.register_ffi_target(
        "graphgp_cuda_build_tree_ffi",
        jax.ffi.pycapsule(graphgp_cuda_lib.build_tree_ffi),
        platform="gpu",
    )
    jax.ffi.register_ffi_target(
        "graphgp_cuda_query_neighbors_ffi",
        jax.ffi.pycapsule(graphgp_cuda_lib.query_neighbors_ffi),
        platform="gpu",
    )
    jax.ffi.register_ffi_target(
        "graphgp_cuda_query_preceding_neighbors_ffi",
        jax.ffi.pycapsule(graphgp_cuda_lib.query_preceding_neighbors_ffi),
        platform="gpu",
    )
    jax.ffi.register_ffi_target(
        "graphgp_cuda_compute_depths_parallel_ffi",
        jax.ffi.pycapsule(graphgp_cuda_lib.compute_depths_parallel_ffi),
        platform="gpu",
    )
    jax.ffi.register_ffi_target(
        "graphgp_cuda_compute_depths_serial_ffi",
        jax.ffi.pycapsule(graphgp_cuda_lib.compute_depths_serial_ffi),
        platform="gpu",
    )
    jax.ffi.register_ffi_target(
        "graphgp_cuda_order_by_depth_ffi",
        jax.ffi.pycapsule(graphgp_cuda_lib.order_by_depth_ffi),
        platform="gpu",
    )
    jax.ffi.register_ffi_target(
        "graphgp_cuda_build_graph_ffi",
        jax.ffi.pycapsule(graphgp_cuda_lib.build_graph_ffi),
        platform="gpu",
    )
    jax.ffi.register_ffi_target(
        "graphgp_cuda_sort_ffi",
        jax.ffi.pycapsule(graphgp_cuda_lib.sort_ffi),
        platform="gpu",
    )

    # Register refine primitive
    refine_p.def_impl(refine_impl)
    refine_p.def_abstract_eval(refine_abstract_eval)
    batching.primitive_batchers[refine_p] = refine_batch
    mlir.register_lowering(refine_p, refine_lowering, platform="gpu")  # type: ignore
    ad.primitive_jvps[refine_p] = refine_value_and_jvp
    ad.primitive_transposes[refine_p] = refine_transpose_rule

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

    # Register refine_nonlinear_jvp primitive
    refine_nonlinear_jvp_p.def_impl(refine_nonlinear_jvp_impl)
    refine_nonlinear_jvp_p.def_abstract_eval(refine_nonlinear_jvp_abstract_eval)
    # batching.primitive_batchers[refine_nonlinear_jvp_p] = refine_nonlinear_jvp_batch
    mlir.register_lowering(
        refine_nonlinear_jvp_p,
        refine_nonlinear_jvp_lowering,  # type: ignore
        platform="gpu",
    )
    ad.primitive_transposes[refine_nonlinear_jvp_p] = refine_nonlinear_jvp_transpose_rule
    refine_nonlinear_jvp_p.multiple_results = True

    # Register refine_nonlinear_vjp primitive
    refine_nonlinear_vjp_p.def_impl(refine_nonlinear_vjp_impl)
    refine_nonlinear_vjp_p.def_abstract_eval(refine_nonlinear_vjp_abstract_eval)
    mlir.register_lowering(
        refine_nonlinear_vjp_p,
        refine_nonlinear_vjp_lowering,  # type: ignore
        platform="gpu",
    )
    refine_nonlinear_vjp_p.multiple_results = True


# ================== refine primitive ====================


def refine(points, neighbors, offsets, cov_bins, cov_vals, initial_values, xi):
    return refine_p.bind(points, neighbors, offsets, cov_bins, cov_vals, initial_values, xi)


def refine_impl(*args):
    return jax.ffi.ffi_call(
        "graphgp_cuda_refine_ffi",
        jax.ShapeDtypeStruct(args[6].shape[:-1] + (args[0].shape[0],), jnp.float32),
    )(*args)


def refine_abstract_eval(*args):
    return ShapedArray(args[6].shape[:-1] + (args[0].shape[0],), jnp.float32)


def refine_lowering(ctx, *args):
    return jax.ffi.ffi_lowering("graphgp_cuda_refine_ffi")(ctx, *args)


def refine_value_and_jvp(primals, tangents):
    if any(type(t) is not ad.Zero for t in tangents[:4]):
        raise NotImplementedError(
            "Differentiation for refine only supported for cov_vals, initial_values, and xi."
        )

    # handle ad.Zero for initial_values and xi
    tangents = (
        *tangents[:5],
        lax.zeros_like_array(primals[5]) if type(tangents[5]) is ad.Zero else tangents[5],
        lax.zeros_like_array(primals[6]) if type(tangents[6]) is ad.Zero else tangents[6],
    )

    if type(tangents[4]) is ad.Zero:  # linear case
        primals_out = refine(*primals)
        tangents_out = refine(*primals[:5], *tangents[5:])
    else:  # nonlinear
        primals_out = refine(*primals)
        _, tangents_out = refine_nonlinear_jvp(*primals, *tangents[4:])
        # even though can do x, dx -> y, dy in one pass, this way enables VJP via transpose
    return primals_out, tangents_out


def refine_transpose_rule(tangents_out, *primals):
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


def refine_batch(vector_args, batch_axes):
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
    return refine(p, n, o, cb, cv, iv, x), 0


# ========== refine_nonlinear_jvp primitive ===========


def refine_nonlinear_jvp(
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
):
    return refine_nonlinear_jvp_p.bind(
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
    )


def refine_nonlinear_jvp_impl(*args):
    return jax.ffi.ffi_call(
        "graphgp_cuda_refine_nonlinear_jvp_ffi",
        (
            jax.ShapeDtypeStruct(args[6].shape[:-1] + (args[0].shape[0],), jnp.float32),
            jax.ShapeDtypeStruct(args[6].shape[:-1] + (args[0].shape[0],), jnp.float32),
        ),
    )(*args)


def refine_nonlinear_jvp_abstract_eval(*args):
    return (
        ShapedArray(args[6].shape[:-1] + (args[0].shape[0],), jnp.float32),
        ShapedArray(args[6].shape[:-1] + (args[0].shape[0],), jnp.float32),
    )


def refine_nonlinear_jvp_lowering(ctx, *args):
    return jax.ffi.ffi_lowering("graphgp_cuda_refine_nonlinear_jvp_ffi")(ctx, *args)


def refine_nonlinear_jvp_transpose_rule(tangents_out, *primals):
    p, n, o, cb, cv, iv, x, dcv_u, div_u, dx_u = primals
    _, dv = tangents_out

    # assert all(ad.is_undefined_primal(t) for t in [dcv, div, dx])
    assert not any(ad.is_undefined_primal(t) for t in [p, n, o, cb, cv, iv, x])

    values = refine(p, n, o, cb, cv, iv, x)
    dv_buffer, dcv, div, dx = refine_nonlinear_vjp(p, n, o, cb, cv, iv, x, values, dv)
    dcv = dcv if ad.is_undefined_primal(dcv_u) else None
    div = div if ad.is_undefined_primal(div_u) else None
    dx = dx if ad.is_undefined_primal(dx_u) else None
    return None, None, None, None, None, None, None, dcv, div, dx


# ========== refine_nonlinear_vjp primitive ==========


def refine_nonlinear_vjp(
    points, neighbors, offsets, cov_bins, cov_vals, initial_values, xi, values, values_tangent
):
    return refine_nonlinear_vjp_p.bind(
        points,
        neighbors,
        offsets,
        cov_bins,
        cov_vals,
        initial_values,
        xi,
        values,
        values_tangent,
    )


def refine_nonlinear_vjp_impl(*args):
    return jax.ffi.ffi_call(
        "graphgp_cuda_refine_nonlinear_vjp_ffi",
        (
            jax.ShapeDtypeStruct(args[7].shape, jnp.float32),
            jax.ShapeDtypeStruct(args[4].shape, jnp.float32),
            jax.ShapeDtypeStruct(args[5].shape, jnp.float32),
            jax.ShapeDtypeStruct(args[6].shape, jnp.float32),
        ),
    )(*args)


def refine_nonlinear_vjp_abstract_eval(*args):
    return (
        ShapedArray(args[7].shape, jnp.float32),
        ShapedArray(args[4].shape, jnp.float32),
        ShapedArray(args[5].shape, jnp.float32),
        ShapedArray(args[6].shape, jnp.float32),
    )


def refine_nonlinear_vjp_lowering(ctx, *args):
    return jax.ffi.ffi_lowering("graphgp_cuda_refine_nonlinear_vjp_ffi")(ctx, *args)


# ========== refine_linear_transpose primitive ==========


def refine_linear_transpose(points, neighbors, offsets, cov_bins, cov_vals, values):
    return refine_linear_transpose_p.bind(points, neighbors, offsets, cov_bins, cov_vals, values)


def refine_linear_transpose_impl(*args):
    return jax.ffi.ffi_call(
        "graphgp_cuda_refine_linear_transpose_ffi",
        (
            jax.ShapeDtypeStruct(args[5].shape, jnp.float32),
            jax.ShapeDtypeStruct(
                args[5].shape[:-1] + (args[0].shape[0] - args[1].shape[0],), jnp.float32
            ),
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
    return jax.ffi.ffi_lowering("graphgp_cuda_refine_linear_transpose_ffi")(ctx, *args)


def refine_linear_transpose_value_and_jvp(primals, tangents):
    if any(type(t) is not ad.Zero for t in tangents[:5]):
        raise NotImplementedError(
            "Differentiation for refine_linear_transpose only supported for values."
        )
    if type(tangents[5]) is ad.Zero:
        raise NotImplementedError("Not differentiated with respect to values?")
    primals_out = refine_linear_transpose(*primals)
    tangents_out = refine_linear_transpose(*primals[:5], *tangents[5:])
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

    dv = refine(p, n, o, cb, cv, div, dx)
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
    return refine_linear_transpose(p, n, o, cb, cv, v), (0, 0, 0)


# ========== Graph construction (not differentiable) ==========


def build_tree(points):
    call = jax.ffi.ffi_call(
        "graphgp_cuda_build_tree_ffi",
        (
            jax.ShapeDtypeStruct(points.shape, jnp.float32),
            jax.ShapeDtypeStruct((points.shape[0],), jnp.int32),
            jax.ShapeDtypeStruct((points.shape[0],), jnp.int32),
            jax.ShapeDtypeStruct((points.shape[0],), jnp.int32),
            jax.ShapeDtypeStruct((points.shape[0],), jnp.float32),
        ),
    )
    points, split_dims, indices, tags, ranges = call(points)
    return points, split_dims, indices


def query_neighbors(points, split_dims, query_indices, max_indices, *, k):
    call = jax.ffi.ffi_call(
        "graphgp_cuda_query_neighbors_ffi",
        jax.ShapeDtypeStruct((query_indices.shape[0], k), jnp.int32),
    )
    neighbors = call(points, split_dims, query_indices, max_indices)
    return neighbors


def query_preceding_neighbors(points, split_dims, *, n0, k):
    call = jax.ffi.ffi_call(
        "graphgp_cuda_query_preceding_neighbors_ffi",
        jax.ShapeDtypeStruct((points.shape[0] - n0, k), jnp.int32),
    )
    neighbors = call(points, split_dims)
    return neighbors


def compute_depths_parallel(neighbors, *, n0):
    call = jax.ffi.ffi_call(
        "graphgp_cuda_compute_depths_parallel_ffi",
        (
            jax.ShapeDtypeStruct((neighbors.shape[0] + n0,), jnp.int32),
            jax.ShapeDtypeStruct((neighbors.shape[0] + n0,), jnp.int32),
        ),
    )
    depths, temp = call(neighbors)
    return depths


def compute_depths_serial(neighbors, *, n0):
    call = jax.ffi.ffi_call(
        "graphgp_cuda_compute_depths_serial_ffi",
        jax.ShapeDtypeStruct((neighbors.shape[0] + n0,), jnp.int32)
    )
    depths = call(neighbors)
    return depths


def order_by_depth(points, indices, neighbors, depths):
    call = jax.ffi.ffi_call(
        "graphgp_cuda_order_by_depth_ffi",
        (
            jax.ShapeDtypeStruct(points.shape, jnp.float32),
            jax.ShapeDtypeStruct(indices.shape, jnp.int32),
            jax.ShapeDtypeStruct(neighbors.shape, jnp.int32),
            jax.ShapeDtypeStruct(depths.shape, jnp.int32),
            jax.ShapeDtypeStruct((2 * depths.shape[0],), jnp.int32),
        ),
    )
    points, indices, neighbors, depths, temp = call(points, indices, neighbors, depths)
    return points, indices, neighbors, depths


def build_graph(points, *, n0, k):
    call = jax.ffi.ffi_call(
        "graphgp_cuda_build_graph_ffi",
        (
            jax.ShapeDtypeStruct(points.shape, jnp.float32),
            jax.ShapeDtypeStruct((points.shape[0],), jnp.int32),
            jax.ShapeDtypeStruct((points.shape[0] - n0, k), jnp.int32),
            jax.ShapeDtypeStruct((points.shape[0],), jnp.int32),
            jax.ShapeDtypeStruct((2 * points.shape[0],), jnp.int32),
        ),
    )
    points, indices, neighbors, depths, temp = call(points)
    return points, indices, neighbors, depths

def sort(keys):
    call = jax.ffi.ffi_call(
        "graphgp_cuda_sort_ffi",
        jax.ShapeDtypeStruct(keys.shape, jnp.float32),
    )
    keys_sorted = call(keys)
    return keys_sorted