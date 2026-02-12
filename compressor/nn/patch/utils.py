import copy
import types
import functools

from typing import *

__all__ = ["copy_func"]


def copy_func(f: types.FunctionType, globals: dict[str, Any] | None = None):
    """
    Copy a function, optionally replacing its globals
    """
    if globals is None:
        globals = f.__globals__

    g = types.FunctionType(f.__code__, globals, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__module__ = f.__module__
    g.__kwdefaults__ = copy.copy(f.__kwdefaults__)
    
    return g