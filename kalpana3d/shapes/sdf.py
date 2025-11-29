import numpy as np
from numba import njit
from ..core.math import length, vec3, normalize, clamp, dot

@njit(fastmath=True)
def sdSphere(p, r):
    return length(p) - r

@njit(fastmath=True)
def sdBox(p, b):
    q = np.abs(p) - b
    return length(np.maximum(q, 0.0)) + min(max(q[0], max(q[1], q[2])), 0.0)

@njit(fastmath=True)
def sdRoundBox(p, b, r):
    return sdBox(p, b) - r

@njit(fastmath=True)
def sdCylinder(p, h, r):
    d = vec3(length(vec3(p[0], 0, p[2])) - r, np.abs(p[1]) - h, 0)
    return min(max(d[0], d[1]), 0.0) + length(np.maximum(d, 0.0))

@njit(fastmath=True)
def sdPlane(p, n, h):
    return np.dot(p, n) + h

@njit(fastmath=True)
def sdTorus(p, t):
    # t[0] = major radius, t[1] = minor radius
    q = vec3(length(vec3(p[0], 0, p[2])) - t[0], p[1], 0)
    return length(q) - t[1]

@njit(fastmath=True)
def sdCapsule(p, a, b, r):
    pa = p - a
    ba = b - a
    h = clamp(np.dot(pa, ba) / np.dot(ba, ba), 0.0, 1.0)
    return length(pa - ba * h) - r

@njit(fastmath=True)
def sdCone(p, c, h):
    # c is the sin/cos of the angle, h is height
    # c must be normalized
    q = length(vec3(p[0], 0, p[2]))
    return max(np.dot(vec3(c[0], c[1], 0), vec3(q, p[1], 0)), -h - p[1])

@njit(fastmath=True)
def sdHexPrism(p, h):
    # h[0] = radius, h[1] = height
    k = vec3(-0.8660254, 0.5, 0.57735027)
    p = np.abs(p)
    p = p - 2.0 * min(np.dot(vec3(k[0], k[1], 0), vec3(p[0], p[1], 0)), 0.0) * vec3(k[0], k[1], 0)
    d = vec3(
       length(vec3(p[0], p[1], 0) - vec3(clamp(p[0], -k[2]*h[0], k[2]*h[0]), h[0], 0)) * np.sign(p[1] - h[0]),
       p[2] - h[1],
       0
    )
    return min(max(d[0], d[1]), 0.0) + length(np.maximum(d, 0.0))

@njit(fastmath=True)
def sdRoundCone(p, r1, r2, h):
    # r1: radius at bottom (y=0)
    # r2: radius at top (y=h)
    # h: height

    # Sampling p relative to cylinder axis (y-axis)
    # q is radial distance from y-axis
    q = vec3(length(vec3(p[0], 0, p[2])), p[1], 0)

    b = (r1 - r2) / h
    a = np.sqrt(1.0 - b*b)

    k = dot(q, vec3(-b, a, 0))

    if k < 0.0: return length(q) - r1
    if k > a*h: return length(q - vec3(0, h, 0)) - r2

    return dot(q, vec3(a, b, 0)) - r1
