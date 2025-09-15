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

def cast_all(*args):
    dtypes = [arg.dtype for arg in args]
    target_dtypes = [
        jnp.float32
        if dt in [jnp.float32, jnp.float64]
        else jnp.int32
        if dt in [jnp.int32, jnp.int64]
        else dt
        for dt in dtypes
    ]
    return tuple(jnp.asarray(arg, dtype=dt) for arg, dt in zip(args, target_dtypes))

def register_ffi_primitive(cdll, function_name, abstract_eval, leading_is_batch, platform="gpu"):
    """
    Register a JAX primitive backed by an XLA FFI call. Mainly to reduce boilerplate, but
    also handles batching rule if FFI function takes a leading batch dimension for some args.
    """

    name = f"graphgp_cuda_{function_name}"
    jax.ffi.register_ffi_target(
        name,
        jax.ffi.pycapsule(getattr(cdll, function_name)),
        platform=platform,
    )
    prim = Primitive(name)

    def impl(*args):
        abstract_out = abstract_eval(*args)
        if isinstance(abstract_out, ShapedArray):
            abstract_out = ShapeDtypeStruct(abstract_out.shape, abstract_out.dtype)
        elif isinstance(abstract_out, tuple):
            abstract_out = tuple(ShapeDtypeStruct(s.shape, s.dtype) for s in abstract_out)
        return jax.ffi.ffi_call(
            name,
            abstract_out,
        )(*args)

    def lowering(ctx, *args):
        return jax.ffi.ffi_lowering(name)(ctx, *args)

    def batching_rule(vector_args, batch_axes):
        bad_batch_args = tuple(
            j for j, a in enumerate(batch_axes) if leading_is_batch[j] is False and a is not None
        )
        if len(bad_batch_args) > 0:
            raise NotImplementedError(
                f"Batching {name} not supported for arguments {bad_batch_args}."
            )
        (size,) = {
            x.shape[bd] for x, bd in zip(vector_args, batch_axes) if bd is not batching.not_mapped
        }
        args = [batching.bdim_at_front(x, bd, size) for x, bd in zip(vector_args, batch_axes)]
        out = prim.bind(*args)
        return (out, (0,) * len(out))

    prim.def_impl(impl)
    prim.def_abstract_eval(abstract_eval)
    batching.primitive_batchers[prim] = batching_rule
    mlir.register_lowering(prim, lowering, platform=platform)
    return prim, prim.bind

def register_transposes(f_p, f_transpose_p, n_nonlinear=0, forward_buffers=0, backward_buffers=0):
    """
    Idea is we want to register two functions as transposes of each other.
    We will require the nonlinear arguments to be the first n_nonlinear arguments, in the same order for both.
    Sometimes the function will have buffer outputs which should be discarded, these should be the first outputs.
    Otherwise we just pass the arguments through.
    We should verify the nonlinear args are not undefined primals, and the linear args are.
    If the output tangent is ad.Zero, return ad.Zero. If it's a tuple and not all ad.Zero, fill array in that case.
    """
    pass

try:
    so_path = next(Path(__file__).parent.glob("libgraphgp_cuda*"))
    lib = ctypes.cdll.LoadLibrary(str(so_path))
except (StopIteration, OSError) as e:
    raise RuntimeError(f"Failed to load graphgp_cuda library: {e}")

sort_p, sort = register_ffi_primitive(
    lib, "sort_ffi", lambda keys: ShapedArray(keys.shape, jnp.float32), leading_is_batch=(False,)
)

# ----------------- refine primitives -------------------

refine_p, refine = register_ffi_primitive(
    lib,
    "refine_ffi",
    lambda p, n, o, cb, cv, iv, x: ShapedArray(iv.shape[:-1] + (p.shape[0],), jnp.float32),
    leading_is_batch=(False, False, False, False, True, True, True),
)

refine_transpose_p, refine_transpose = register_ffi_primitive(
    lib,
    "refine_transpose_ffi",
    lambda p, n, o, cb, cv, v: (
        ShapedArray(v.shape, jnp.float32),
        ShapedArray(v.shape[:-1] + (p.shape[0] - n.shape[0],), jnp.float32),
        ShapedArray(v.shape[:-1] + (n.shape[0],), jnp.float32),
    ),
    leading_is_batch=(False, False, False, False, True, True),
)
refine_transpose_p.multiple_results = True

refine_jvp_p, refine_jvp = register_ffi_primitive(
    lib,
    "refine_jvp_ffi",
    lambda p, n, o, cb, cv, iv, x, cv_t, iv_t, x_t: (
        ShapedArray(iv.shape[:-1] + (p.shape[0],), jnp.float32),
        ShapedArray(iv.shape[:-1] + (p.shape[0],), jnp.float32),
    ),
    leading_is_batch=(False, False, False, False, True, True, True, True, True, True),
)
refine_jvp_p.multiple_results = True

refine_vjp_p, refine_vjp = register_ffi_primitive(
    lib,
    "refine_vjp_ffi",
    lambda p, n, o, cb, cv, iv, x, v, v_t: (
        ShapedArray(v.shape, jnp.float32),
        ShapedArray(cv.shape, jnp.float32),
        ShapedArray(iv.shape, jnp.float32),
        ShapedArray(x.shape, jnp.float32),
    ),
    leading_is_batch=(False, False, False, False, True, True, True, False, False),
)
refine_vjp_p.multiple_results = True

register_transposes(refine_jvp_p, refine_vjp_p, n_nonlinear=7, forward_buffers=0, backward_buffers=1)
register_transposes(refine_p, refine_transpose_p, n_nonlinear=3, forward_buffers=0, backward_buffers=1)


def refine_value_and_jvp(primals, tangents):
    p, n, o, cb, cv, iv, x = primals
    p_t, n_t, o_t, cb_t, cv_t, iv_t, x_t = tangents

    if any(type(t) is not ad.Zero for t in [p_t, n_t, o_t, cb_t]):
        raise NotImplementedError(
            "Differentiation for refine only supported for cov_vals, initial_values, and xi."
        )

    # handle ad.Zero for initial_values and xi
    tangents = (
        *tangents[:5],
        lax.zeros_like_array(primals[5]) if type(tangents[5]) is ad.Zero else tangents[5],
        lax.zeros_like_array(primals[6]) if type(tangents[6]) is ad.Zero else tangents[6],
    )

    if type(cv_t) is ad.Zero:  
        v = refine(p, n, o, cb, cv, iv, x)
        dv = refine(p, n, o, cb, cv, iv_t, x_t)
    else:
        v = refine(p, n, o, cb, cv, iv, x)
        _, dv = refine_jvp(p, n, o, cb, cv, iv, x, cv_t, iv_t, x_t)
    return v, dv

# # Register refine primitive
# refine_p.def_impl(refine_impl)
# refine_p.def_abstract_eval(refine_abstract_eval)
# batching.primitive_batchers[refine_p] = refine_batch
# mlir.register_lowering(refine_p, refine_lowering, platform="gpu")  # type: ignore
# ad.primitive_jvps[refine_p] = refine_value_and_jvp
# ad.primitive_transposes[refine_p] = refine_transpose_rule


# ---------------- graph construction (not differentiable) ----------------

build_tree_p, build_tree = register_ffi_primitive(
    lib,
    "build_tree_ffi",
    lambda points: (
        ShapedArray(points.shape, jnp.float32),
        ShapedArray((points.shape[0],), jnp.int32),
        ShapedArray((points.shape[0],), jnp.int32),
        ShapedArray((points.shape[0],), jnp.int32),
        ShapedArray((points.shape[0],), jnp.float32),
    ),
    leading_is_batch=(False,),
)
build_tree = jax.jit(build_tree)



@Partial(jax.jit, static_argnames=("n0", "k"))
def query_preceding_neighbors(points, split_dims, *, n0, k):
    call = jax.ffi.ffi_call(
        "graphgp_cuda_query_preceding_neighbors_ffi",
        jax.ShapeDtypeStruct((points.shape[0] - n0, k), jnp.int32),
    )
    neighbors = call(*cast_all(points, split_dims))
    return neighbors


@Partial(jax.jit, static_argnames="n0")
def compute_depths_parallel(neighbors, *, n0):
    call = jax.ffi.ffi_call(
        "graphgp_cuda_compute_depths_parallel_ffi",
        (
            jax.ShapeDtypeStruct((neighbors.shape[0] + n0,), jnp.int32),
            jax.ShapeDtypeStruct((neighbors.shape[0] + n0,), jnp.int32),
        ),
    )
    depths, temp = call(*cast_all(neighbors))
    return depths


@Partial(jax.jit, static_argnames="n0")
def compute_depths_serial(neighbors, *, n0):
    call = jax.ffi.ffi_call(
        "graphgp_cuda_compute_depths_serial_ffi",
        jax.ShapeDtypeStruct((neighbors.shape[0] + n0,), jnp.int32),
    )
    depths = call(*cast_all(neighbors))
    return depths


@jax.jit
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
    points, indices, neighbors, depths, temp = call(*cast_all(points, indices, neighbors, depths))
    return points, indices, neighbors, depths


@Partial(jax.jit, static_argnames=("n0", "k"))
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
    points, indices, neighbors, depths, temp = call(*cast_all(points))
    return points, indices, neighbors, depths



# ================== refine primitive ====================


def refine(points, neighbors, offsets, cov_bins, cov_vals, initial_values, xi):
    return refine_p.bind(points, neighbors, offsets, cov_bins, cov_vals, initial_values, xi)


def refine_impl(*args):
    return jax.ffi.ffi_call(
        "graphgp_cuda_refine_ffi",
        jax.ShapeDtypeStruct(args[6].shape[:-1] + (args[0].shape[0],), jnp.float32),
    )(*cast_all(*args))


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
    )(*cast_all(*args))


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
    )(*cast_all(*args))


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
    )(*cast_all(*args))


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





def sort_three(keys1, keys2, keys3):
    call = jax.ffi.ffi_call(
        "graphgp_cuda_sort_three_ffi",
        (
            jax.ShapeDtypeStruct(keys1.shape, jnp.float32),
            jax.ShapeDtypeStruct(keys2.shape, jnp.float32),
            jax.ShapeDtypeStruct(keys3.shape, jnp.float32),
        ),
    )
    keys_sorted = call(*cast_all(keys1, keys2, keys3))
    return keys_sorted


def sort_four(keys1, keys2, keys3, keys4):
    call = jax.ffi.ffi_call(
        "graphgp_cuda_sort_four_ffi",
        (
            jax.ShapeDtypeStruct(keys1.shape, jnp.float32),
            jax.ShapeDtypeStruct(keys2.shape, jnp.float32),
            jax.ShapeDtypeStruct(keys3.shape, jnp.float32),
            jax.ShapeDtypeStruct(keys4.shape, jnp.float32),
        ),
    )
    keys_sorted = call(*cast_all(keys1, keys2, keys3, keys4))
    return keys_sorted
