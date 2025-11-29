import numpy as np
from numba import njit
from ..core.math import vec3, length, dot, mix, normalize, clamp

@njit(fastmath=True)
def sdCapsule(p, a, b, r):
    pa = p - a
    ba = b - a
    h = clamp(np.dot(pa, ba) / np.dot(ba, ba), 0.0, 1.0)
    return length(pa - ba * h) - r

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

@njit(fastmath=True)
def rotateY(p, a):
    c = np.cos(a)
    s = np.sin(a)
    return vec3(c * p[0] - s * p[2], p[1], s * p[0] + c * p[2])

@njit(fastmath=True)
def rotateZ(p, a):
    c = np.cos(a)
    s = np.sin(a)
    return vec3(c * p[0] - s * p[1], s * p[0] + c * p[1], p[2])

@njit(fastmath=True)
def sdTree(p):
    # A procedural tree based on KIFS (Kaleidoscopic Iterated Function System)
    # Using 'fold' and 'rotate' to create branching.

    scale = 1.0
    d = 1e9

    # Parameters
    iters = 6
    angle_z = 0.6  # Branching angle ~35 degrees
    angle_y = 1.6  # Rotation around trunk ~90 degrees
    reduction = 0.70 # Scale down factor
    len_y = 1.2
    radius = 0.08

    curr_p = p

    for i in range(iters):
        # Current segment (Trunk/Branch)
        # Tapering radius: radius * (reduction^i)
        r = radius * pow(0.65, float(i))
        l = len_y * pow(0.72, float(i))

        # Distance to this segment (Capsule from 0 to 0,l,0)
        # We assume local space is at the base of the branch
        seg_d = sdCapsule(curr_p, vec3(0,0,0), vec3(0,l,0), r)

        d = min(d, seg_d)

        # Transform for next iteration (Children)

        # 1. Move to end of current branch
        curr_p = curr_p - vec3(0, l, 0)

        # 2. Branching Logic
        # Fold space to create multiple branches from one path

        # Rotate around Y to distribute branches in 3D
        # Using a golden angle or similar helps avoid alignment
        # But for a tree, we often want opposite or alternate branching.
        # Let's try rotating Y by 90 degrees + some offset
        curr_p = rotateY(curr_p, angle_y)

        # Fold X (Symmetry) -> Creates 2 branches
        curr_p = vec3(abs(curr_p[0]), curr_p[1], abs(curr_p[2]))

        # Rotate Z to angle out
        curr_p = rotateZ(curr_p, angle_z)

        # Fold again? Or just let it be.
        # With just Fold X and Rotate Z, we get a planar bifurcation.
        # Rotating Y before folding distributes them.

        # Additional twist to make it look organic?
        # curr_p = rotateY(curr_p, 0.5)

        # Scale is handled by reducing l and r in the next loop,
        # so we don't scale the space coordinate (which preserves distance field metric)

    return d
