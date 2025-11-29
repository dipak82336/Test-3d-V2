import numpy as np
from numba import njit
from ..core.math import vec3, length, dot, mix, normalize

@njit(fastmath=True)
def sdStar5(p, r, rf):
    k1 = np.array([0.809016994375, -0.587785252292], dtype=np.float32)
    k2 = np.array([-k1[0], k1[1]], dtype=np.float32)
    p_xy = np.array([p[0], p[1]], dtype=np.float32)
    p_xy[0] = abs(p_xy[0])
    p_xy -= 2.0 * max(dot(k1, p_xy), np.float32(0.0)) * k1
    p_xy -= 2.0 * max(dot(k2, p_xy), np.float32(0.0)) * k2
    p_xy[0] = abs(p_xy[0])
    p_xy[1] -= r
    ba = rf * np.array([-k1[1], k1[0]], dtype=np.float32) - np.array([0.0, 1.0], dtype=np.float32)
    h = np.float32(dot(p_xy, ba)) / np.float32(dot(ba, ba))
    h = min(max(h, np.float32(0.0)), r)
    return length(p_xy - ba * h) * np.sign(p_xy[1] * ba[0] - p_xy[0] * ba[1])

@njit(fastmath=True)
def sdExtrudedStar(p, r, rf, h):
    d = sdStar5(p, r, rf)
    w = np.array([d, abs(p[2]) - h], dtype=np.float32)
    return min(max(w[0], w[1]), 0.0) + length(np.maximum(w, 0.0))

@njit(fastmath=True)
def sdMandelbulb(p):
    w = p
    m = dot(w, w)
    dz = 1.0
    
    for i in range(8):
        # dz = 8 * r^7 * dz + 1
        dz = 8.0 * (m ** 3.5) * dz + 1.0
        
        # z = z^8 + c
        r = length(w)
        if r == 0: break
        b = 8.0 * np.arccos(w[1] / r)
        a = 8.0 * np.arctan2(w[0], w[2])
        w = p + (r ** 8.0) * vec3(np.sin(b) * np.sin(a), np.cos(b), np.sin(b) * np.cos(a))
        
        m = dot(w, w)
        if m > 256.0:
            break
            
    return 0.25 * np.log(m) * np.sqrt(m) / dz

@njit(fastmath=True)
def sdSierpinski(p):
    z = p
    n = 0
    while n < 10:
        if z[0] + z[1] < 0.0: z = vec3(-z[1], -z[0], z[2]) # Fold 1
        if z[0] + z[2] < 0.0: z = vec3(-z[2], z[1], -z[0]) # Fold 2
        if z[1] + z[2] < 0.0: z = vec3(z[0], -z[2], -z[1]) # Fold 3
        
        z = z * 2.0 - vec3(1.0, 1.0, 1.0)
        n += 1
        
    return (length(z) - 2.0) * pow(2.0, -float(n))
