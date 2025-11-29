import numpy as np
from numba import njit
from ..core.math import mix, clamp

@njit(fastmath=True)
def opUnion(d1, d2):
    return min(d1, d2)

@njit(fastmath=True)
def opSubtraction(d1, d2):
    return max(-d1, d2)

@njit(fastmath=True)
def opIntersection(d1, d2):
    return max(d1, d2)

@njit(fastmath=True)
def opSmoothUnion(d1, d2, k):
    h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return mix(d2, d1, h) - k * h * (1.0 - h)

@njit(fastmath=True)
def opSmoothSubtraction(d1, d2, k):
    h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0)
    return mix(d2, -d1, h) + k * h * (1.0 - h)

@njit(fastmath=True)
def opSmoothIntersection(d1, d2, k):
    h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return mix(d2, d1, h) + k * h * (1.0 - h)
