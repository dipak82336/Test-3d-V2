import numpy as np
from numba import njit

@njit(fastmath=True)
def vec3(x, y, z):
    return np.array([x, y, z], dtype=np.float32)

@njit(fastmath=True)
def normalize(v):
    l = np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if l == 0:
        return v
    return v / l

@njit(fastmath=True)
def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@njit(fastmath=True)
def cross(a, b):
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ], dtype=np.float32)

@njit(fastmath=True)
def length(v):
    return np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

@njit(fastmath=True)
def mix(a, b, t):
    return a * (1.0 - t) + b * t

@njit(fastmath=True)
def clamp(x, min_val, max_val):
    return min(max(x, min_val), max_val)

@njit(fastmath=True)
def reflect(i, n):
    return i - 2.0 * dot(n, i) * n
