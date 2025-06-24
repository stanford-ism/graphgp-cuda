import numpy as np
import jaxlib.xla_extension
import functools

_indentation = 0

def _trace(msg=None):
    """Print a message at current indentation."""
    if msg is not None:
        print("  " * _indentation + msg)


def _trace_indent(msg=None):
    """Print a message and then indent the rest."""
    global _indentation
    _trace(msg)
    _indentation = 1 + _indentation


def _trace_unindent(msg=None):
    """Unindent then print a message."""
    global _indentation
    _indentation = _indentation - 1
    _trace(msg)


def trace(name):
    """A decorator for functions to trace arguments and results."""

    def trace_func(func):  # pylint: disable=missing-docstring
        def pp(v):
            """Print certain values more succinctly"""
            vtype = str(type(v))
            if "jax._src.xla_bridge._JaxComputationBuilder" in vtype:
                return "<JaxComputationBuilder>"
            elif "jaxlib._jax_.XlaOp" in vtype:
                return "<XlaOp at 0x{:x}>".format(id(v))
            elif (
                "partial_eval.JaxprTracer" in vtype
                or "batching.BatchTracer" in vtype
                or "ad.JVPTracer" in vtype
            ):
                return "Traced<{}>".format(v.aval)
            elif isinstance(v, tuple):
                return "({})".format(pp_values(v))
            elif isinstance(v, jaxlib.xla_extension.ArrayImpl):
                return "ArrayImpl<{}>".format(v.shape)
            elif isinstance(v, np.ndarray):
                return "NdArray<{}>".format(v.shape)
            else:
                return str(v)

        def pp_values(args):
            return ", ".join([pp(arg) for arg in args])

        @functools.wraps(func)
        def func_wrapper(*args):
            _trace_indent("call {}({})".format(name, pp_values(args)))
            res = func(*args)
            _trace_unindent("|<- {} = {}".format(name, pp(res)))
            return res

        return func_wrapper

    return trace_func