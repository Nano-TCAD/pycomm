from pycomm import backend_flags

import numpy as np

if backend_flags["cupy_avail"]:
    import cupy as cp

def _get_module_from_array(arr: ArrayLike):
    """Return the array module of the input array.

    Parameters
    ----------
    arr : ArrayLike
        Input array.

    Returns
    -------
    module : module
        The array module of the input array. (numpy or cupy)
    la : module
        The linear algebra module of the array module. (scipy.linalg or cupyx.scipy.linalg)
    """
    if backend_flags["cupy_avail"]:
        xp = cp.get_array_module(arr)

        if xp == cp:
            return cp

    return np