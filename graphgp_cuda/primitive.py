import ctypes
from pathlib import Path
import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import mlir, batching, ad
from jax.tree_util import Partial

refine_p = {"32": Primitive("graphgp_cuda_refine_32"), "64": Primitive("graphgp_cuda_refine_64")}
refine_transpose_p = {
    "32": Primitive("graphgp_cuda_refine_transpose_32"),
    "64": Primitive("graphgp_cuda_refine_transpose_64"),
}
refine_jvp_p = {
    "32": Primitive("graphgp_cuda_refine_jvp_32"),
    "64": Primitive("graphgp_cuda_refine_jvp_64"),
}
refine_vjp_p = {
    "32": Primitive("graphgp_cuda_refine_vjp_32"),
    "64": Primitive("graphgp_cuda_refine_vjp_64"),
}
refine_inv_p = {
    "32": Primitive("graphgp_cuda_refine_inv_32"),
    "64": Primitive("graphgp_cuda_refine_inv_64"),
}
refine_logdet_p = {
    "32": Primitive("graphgp_cuda_refine_logdet_32"),
    "64": Primitive("graphgp_cuda_refine_logdet_64"),
}


def refine(points, neighbors, offsets, cov_bins, cov_vals, initial_values, xi):
    casted_args = _cast_all(points, neighbors, offsets, cov_bins, cov_vals, initial_values, xi)
    if casted_args[0].dtype == jnp.float64:
        return refine_p["64"].bind(*casted_args)[0]
    else:
        return refine_p["32"].bind(*casted_args)[0]


def refine_inv(points, neighbors, offsets, cov_bins, cov_vals, values):
    casted_args = _cast_all(points, neighbors, offsets, cov_bins, cov_vals, values)
    if casted_args[0].dtype == jnp.float64:
        return refine_inv_p["64"].bind(*casted_args)
    else:
        return refine_inv_p["32"].bind(*casted_args)


def refine_logdet(points, neighbors, offsets, cov_bins, cov_vals):
    casted_args = _cast_all(points, neighbors, offsets, cov_bins, cov_vals)
    if casted_args[0].dtype == jnp.float64:
        return refine_logdet_p["64"].bind(*casted_args)[0]
    else:
        return refine_logdet_p["32"].bind(*casted_args)[0]


def _determine_dtype(*args):
    for x in args:
        if x.dtype in (jnp.int64, jnp.float64):
            return jnp.int64, jnp.float64, "_64"
    return jnp.int32, jnp.float32, "_32"


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
            f"graphgp_cuda_{name}_ffi_32",
            jax.ffi.pycapsule(getattr(lib, f"{name}_ffi_32")),
            platform="CUDA",
        )
        jax.ffi.register_ffi_target(
            f"graphgp_cuda_{name}_ffi_64",
            jax.ffi.pycapsule(getattr(lib, f"{name}_ffi_64")),
            platform="CUDA",
        )

    for mode in ["32", "64"]:
        dtype = jnp.float32 if mode == "32" else jnp.float64

        # Refine forward pass
        def refine_abstract(points, neighbors, offsets, cov_bins, cov_vals, initial_values, xi):
            n = points.shape[0]
            batch_shape = initial_values.shape[:-1]
            return (ShapedArray(batch_shape + (n,), dtype),)

        # Refine JVP rule
        def refine_value_and_jvp(primals, tangents):
            primals_out = refine_p[mode].bind(*primals)

            if all(isinstance(t, ad.Zero) for t in tangents[:5]):  # linear case
                filled_tangents = tuple(ad.instantiate_zeros(t) for t in tangents[5:])
                tangents_out = refine_p[mode].bind(*primals[:5], *filled_tangents)

            elif all(isinstance(t, ad.Zero) for t in tangents[:4]):  # covariance derivative
                filled_tangents = tuple(ad.instantiate_zeros(t) for t in tangents[4:])
                tangents_out = refine_jvp_p[mode].bind(*primals, primals_out[0], *filled_tangents)
            else:
                nonzero_tangents = tuple(i for i, t in enumerate(tangents) if not isinstance(t, ad.Zero))
                raise NotImplementedError(f"JVP not supported for non-Zero tangents {nonzero_tangents}.")

            return primals_out, tangents_out

        register_primitive(
            refine_p[mode],
            ffi_name=f"graphgp_cuda_refine_ffi_{mode}",
            abstract=refine_abstract,
            batch_args=(4, 5, 6),
            transposes=[(refine_transpose_p[mode], (5, 6))],
            jvp_rule=refine_value_and_jvp,
            platform="gpu",
        )

        # Refine linear transpose
        def refine_transpose_abstract(points, neighbors, offsets, cov_bins, cov_vals, values):
            n = points.shape[0]
            n0 = n - neighbors.shape[0]
            batch_shape = values.shape[:-1]
            return (
                ShapedArray(batch_shape + (n0,), dtype),
                ShapedArray(batch_shape + (n - n0,), dtype),
                ShapedArray(batch_shape + (n,), dtype),  # buffer
            )

        register_primitive(
            refine_transpose_p[mode],
            ffi_name=f"graphgp_cuda_refine_transpose_ffi_{mode}",
            abstract=refine_transpose_abstract,
            batch_args=(4, 5),
            transposes=[(refine_p[mode], (5,))],
            platform="gpu",
        )

        # Refine covariance linearization (JVP)
        def refine_jvp_abstract(
            points,
            neighbors,
            offsets,
            cov_bins,
            cov_vals,
            initial_values,
            xi,
            values,
            cov_vals_tangent,
            initial_values_tangent,
            xi_tangent,
        ):
            n = points.shape[0]
            batch_shape = values.shape[:-1]
            return (ShapedArray(batch_shape + (n,), dtype),)

        register_primitive(
            refine_jvp_p[mode],
            ffi_name=f"graphgp_cuda_refine_jvp_ffi_{mode}",
            abstract=refine_jvp_abstract,
            batch_args=(4, 5, 6, 7, 8, 9, 10),
            transposes=[(refine_vjp_p[mode], (8, 9, 10))],
            platform="gpu",
        )

        # Refine covariance linearization (VJP)
        def refine_vjp_abstract(
            points,
            neighbors,
            offsets,
            cov_bins,
            cov_vals,
            initial_values,
            xi,
            values,
            values_tangent,
        ):
            return (
                ShapedArray(cov_vals.shape, dtype),
                ShapedArray(initial_values.shape, dtype),
                ShapedArray(xi.shape, dtype),
                ShapedArray(values.shape, dtype),  # buffer
            )

        register_primitive(
            refine_vjp_p[mode],
            ffi_name=f"graphgp_cuda_refine_vjp_ffi_{mode}",
            abstract=refine_vjp_abstract,
            batch_args=(4, 5, 6, 7, 8),
            transposes=[(refine_jvp_p[mode], (8,))],
            platform="gpu",
        )

        # Refine inverse
        def refine_inv_abstract(
            points,
            neighbors,
            offsets,
            cov_bins,
            cov_vals,
            values,
        ):
            n = points.shape[0]
            n0 = n - neighbors.shape[0]
            batch_shape = values.shape[:-1]
            return (
                ShapedArray(batch_shape + (n0,), dtype),  # initial_values
                ShapedArray(batch_shape + (n - n0,), dtype),  # xi
            )

        register_primitive(
            refine_inv_p[mode],
            ffi_name=f"graphgp_cuda_refine_inv_ffi_{mode}",
            abstract=refine_inv_abstract,
            batch_args=(4, 5),
            platform="gpu",
        )

        # Refine log determinant
        def refine_logdet_abstract(
            points,
            neighbors,
            offsets,
            cov_bins,
            cov_vals,
        ):
            batch_shape = cov_vals.shape[:-1]
            return (ShapedArray(batch_shape, dtype),)  # logdet

        register_primitive(
            refine_logdet_p[mode],
            ffi_name=f"graphgp_cuda_refine_logdet_ffi_{mode}",
            abstract=refine_logdet_abstract,
            batch_args=(4,),
            platform="gpu",
        )


# ===================== Automatic custom primitive registration =====================


def make_impl_rule(ffi_name, abstract):
    """
    Use abstract function to generate abstract_eval and impl for primitive.
    The abstract function should return a tuple of ShapedArrays describing the output(s) of the primitive.
    """

    def impl(*args):
        abstract_out = abstract(*args)
        if not isinstance(abstract_out, tuple):
            raise ValueError("Abstract function must return a tuple of ShapedArrays.")
        ffi_out = tuple(ShapeDtypeStruct(s.shape, s.dtype) for s in abstract_out)
        out = jax.ffi.ffi_call(
            ffi_name,
            ffi_out,
        )(*args)
        return out

    return impl


def make_transpose_rule(transposes):
    """
    Make a transpose rule with a list of (transpose_prim, linear_args) pairs. The transpose primitive should take all nonlinear
    primal inputs, as well as tangents for all outputs, and return tangents for the linear arguments.

    For example, if (x, y, z) -> (a, b) is linear in x and y for both outputs, linear_args should be (0, 1) and
    the transpose primitive should have the signature (z, at, bt) -> (xt, yt). The function could also be linear
    in just z, in which case linear_args should be (2,) with transpose signature (x, y, at, bt) -> (zt,).

    There is no support for different combinations of outputs being linear in different arguments, so we cannot
    write code which takes advantage when at or bt is Zero. This could be added in the future.
    """

    def transpose_rule(tangents_out, *primals_in):
        # find first transpose primitive that covers all undefined primals
        undefined_primals = tuple(i for i, p in enumerate(primals_in) if ad.is_undefined_primal(p))
        for transpose_prim, linear_args in transposes:
            nonlinear_args = tuple(i for i in range(len(primals_in)) if i not in linear_args)
            if not any(i in undefined_primals for i in nonlinear_args) and transpose_prim is not None:
                # process inputs and pass to transpose primitive
                trimmed_primals = tuple(p for i, p in enumerate(primals_in) if i in nonlinear_args)
                filled_tangents_out = (ad.instantiate_zeros(t) for t in tangents_out)
                tangents_in = transpose_prim.bind(*trimmed_primals, *filled_tangents_out)

                # make tuple with tangents where undefined primal and None elsewhere, as expected by JAX
                mapping = dict(zip(linear_args, tangents_in))
                tangents_in_full = tuple(
                    mapping.get(i, None) if i in undefined_primals else None for i in range(len(primals_in))
                )
                # NOTE: This ensures that extra returned buffers are truncated.
                return tangents_in_full

        raise ValueError(f"No transpose rule for arguments {undefined_primals}.")

    return transpose_rule


def make_batching_rule(prim, batch_args):
    """
    Make a rule that puts batch dimensions along leading axis for all batch_args, broadcasting as needed.
    If any non-batch_args have a batch dimension, raise an error.
    """

    def batching_rule(vector_args, batch_axes):
        bad_batch_args = tuple(
            j for j, bd in enumerate(batch_axes) if j not in batch_args and bd is not batching.not_mapped
        )
        if len(bad_batch_args) > 0:
            raise NotImplementedError(f"Batching {prim} not supported for arguments {bad_batch_args}.")
        (size,) = {x.shape[bd] for x, bd in zip(vector_args, batch_axes) if bd is not batching.not_mapped}
        args = []
        for j, (x, bd) in enumerate(zip(vector_args, batch_axes)):
            if bd is batching.not_mapped:
                if j in batch_args:
                    args.append(jnp.broadcast_to(x, (size,) + x.shape))
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


def register_primitive(prim, *, ffi_name, abstract, platform, batch_args=(), transposes=[], jvp_rule=None):
    """
    Call several helper functions to register a custom JAX primitive with a given abstract eval, FFI implementation, batching rule, and transposes.
    """
    prim.def_abstract_eval(abstract)
    prim.def_impl(make_impl_rule(ffi_name, abstract))
    mlir.register_lowering(prim, jax.ffi.ffi_lowering(ffi_name), platform=platform)

    prim.multiple_results = True
    batching.primitive_batchers[prim] = make_batching_rule(prim, batch_args)

    ad.primitive_transposes[prim] = make_transpose_rule(transposes)
    if jvp_rule is not None:
        ad.primitive_jvps[prim] = jvp_rule


# ===================== Graph construction (not differentiable) =====================


@jax.jit
def build_tree(points):
    int_dtype, float_dtype, mode = _determine_dtype(points)
    call = jax.ffi.ffi_call(
        "graphgp_cuda_build_tree_ffi" + mode,
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
    int_dtype, float_dtype, mode = _determine_dtype(points)
    call = jax.ffi.ffi_call(
        "graphgp_cuda_query_preceding_neighbors_ffi" + mode,
        jax.ShapeDtypeStruct((points.shape[0] - n0, k), int_dtype),
    )
    neighbors = call(*_cast_all(points, split_dims))
    return neighbors


@Partial(jax.jit, static_argnames=("k"))
def query_neighbors(points, split_dims, query_indices, max_indices, *, k):
    int_dtype, float_dtype, mode = _determine_dtype(points, split_dims, query_indices, max_indices)
    call = jax.ffi.ffi_call(
        "graphgp_cuda_query_neighbors_ffi" + mode,
        jax.ShapeDtypeStruct((query_indices.shape[0], k), int_dtype),
    )
    neighbors = call(*_cast_all(points, split_dims, query_indices, max_indices))
    return neighbors


@Partial(jax.jit, static_argnames="n0")
def compute_depths_parallel(neighbors, *, n0):
    int_dtype, _, mode = _determine_dtype(neighbors)
    call = jax.ffi.ffi_call(
        "graphgp_cuda_compute_depths_parallel_ffi" + mode,
        (
            jax.ShapeDtypeStruct((neighbors.shape[0] + n0,), int_dtype),
            jax.ShapeDtypeStruct((neighbors.shape[0] + n0,), int_dtype),
        ),
    )
    depths, temp = call(*_cast_all(neighbors))
    return depths


@jax.jit
def order_by_depth(points, indices, neighbors, depths):
    int_dtype, float_dtype, mode = _determine_dtype(points, indices, neighbors, depths)
    call = jax.ffi.ffi_call(
        "graphgp_cuda_order_by_depth_ffi" + mode,
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
    int_dtype, float_dtype, mode = _determine_dtype(points)
    call = jax.ffi.ffi_call(
        "graphgp_cuda_build_graph_ffi" + mode,
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


# Approach below is complicated and I'm not sure it's worth the loss of transparency!

# def make_value_and_jvp_rule(prim, transposes, linearizations):
#     """
#     Make a JVP rule for a primitive with a list of transposes and linearizations.

#     Transposes should be a list of (transpose_prim, linear_args) pairs as expected by make_transpose_rule. If transpose_prim is None forward-mode autodiff will work but reverse-mode will not.
#     Linearizations should be a list of (linearization_prim, linear_args) pairs. The linearization primitive should take all primal inputs and outputs, as well as tangents for linear argument, and return tangents for all outputs.

#     For example, if (x, y, z) -> (a, b) is a nonlinear function, and we specify linear_args = (0, 1), then the linearization primitive should be (x, y, z, a, b, xt, yt) -> (at, bt).
#     If instead we specify linear_args = (2,), then the linearization primitive should be (x, y, z, a, b, zt) -> (at, bt).

#     When the JVP rule is called, tangents will be a mix of concrete values and Zeros. We need to identify a combination of linear configurations and linearizations whose linear_args cover the non-Zero tangents.
#     We do this by exhaustive search over all combinations in order of increasing number, stopping at the first valid one. This is fine since we expect the number of implemented functions to be small.
#     Then we need to correctly apply the product rule to calculate the desired tangents using the chosen functions.
#     """

#     def value_and_jvp_rule(primals, tangents):
#         primals_out = prim.bind(*primals)
#         required_linear = set(i for i, t in enumerate(tangents) if type(t) is not ad.Zero)

#         if len(transposes) == 0 and len(linearizations) == 0:
#             raise NotImplementedError("No sets of linear arguments or linearizations exist for primitive, so no autodiff support.")

#         combined = [(prim, linear_args, "linear") for _, linear_args in transposes] + [(lin_prim, linear_args, "linearization") for lin_prim, linear_args in linearizations]

#         # iterate over all combinations of linear and linearizations, stopping when we find one that covers all required linear arguments
#         for n in range(1, len(combined) + 1):
#             for combo in itertools.combinations(combined, n):
#                 covered_args = set(linear_args for _, linear_args, _ in combo)
#                 if required_linear.issubset(covered_args):
#                     pass
#                     # do product rule stuff


#         raise NotImplementedError(f"No combination of linear arguments and linearizations could cover the requested tangents {required_linear}.")
