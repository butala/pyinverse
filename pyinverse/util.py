import math

import numpy as np


"""
MDB: Floating point number equality is tested with numpy.isclose below
instead of math.isclose. The functions are similar, but are not
equivalent. I found the math module implementation to be too stringent
(note that abs_tol=0.0 by default in the math implementation but not
numpy).
"""

def robust_scalar_arcsin(x):
    """Return the arcsin of *x* (see :func:`math.asin`). The arcsin
    function is defined on the domain :math:`|x| \leq 1`. Due to
    finite precision issues, *x* can be epsilon larger than 1 or less
    than -1. Catch for these cases.

    """
    try:
        return math.asin(x)
    except ValueError:
        if x > 1:
            assert np.isclose(x, 1)
            return math.pi / 2
        if x < -1:
            assert np.isclose(x, -1)
            return -math.pi / 2
        raise

def robust_scalar_sqrt(x):
    """Return the sqrt of *x* (see :func:`math.sqrt`). The sqrt
    function is defined on the domain :math:`x \geq 0`. Due to
    finite precision issues, *x* can be epsilon less
    than 0. Catch for this case.

    """
    try:
        return math.sqrt(x)
    except ValueError:
        if x < 0:
            assert np.isclose(x, 0)
            return 0.0
        raise

# generate vectorized functions that accept numpy.array input
robust_arcsin = np.vectorize(robust_scalar_arcsin)
robust_sqrt = np.vectorize(robust_scalar_sqrt)
