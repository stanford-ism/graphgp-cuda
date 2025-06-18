import ctypes
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np


def _register():
    so_path = next(Path(__file__).parent.glob("libhugegp_cuda*"))
    hugegp_cuda_lib = ctypes.cdll.LoadLibrary(str(so_path))
    jax.ffi.register_ffi_target(
        "hugegp_cuda_refine_static_xla",
        jax.ffi.pycapsule(hugegp_cuda_lib.refine_static_xla),
        platform="CUDA",
    )
    jax.ffi.register_ffi_target(
        "hugegp_cuda_query_coarse_neighbors_xla",
        jax.ffi.pycapsule(hugegp_cuda_lib.query_coarse_neighbors_xla),
        platform="CUDA",
    )


def refine_static(
    points,
    xi,
    indices,
    neighbors,
    level_offsets,
    cov_distances,
    cov_values,
    initial_values,
):
    call = jax.ffi.ffi_call(
        "hugegp_cuda_refine_static_xla",
        jax.ShapeDtypeStruct((len(points),), jnp.float32),
        vmap_method="sequential",
    )
    values = call(
        points, xi, indices, neighbors, level_offsets, cov_distances, cov_values, initial_values
    )
    return values


def query_coarse_neighbors(tree, k):
    points, indices, split_dims = tree
    call = jax.ffi.ffi_call(
        "hugegp_cuda_query_coarse_neighbors_xla",
        jax.ShapeDtypeStruct((len(points), k), jnp.uint32),
        vmap_method="sequential",
    )
    neighbors = call(points, indices, split_dims, k=np.int32(k))
    return neighbors
