from .ffi import _register, refine_static, query_coarse_neighbors

_register()

__all__ = ["refine_static", "query_coarse_neighbors"]
