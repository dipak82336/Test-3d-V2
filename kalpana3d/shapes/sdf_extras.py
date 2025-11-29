import numpy as np
from numba import njit
from ..core.math import vec3, length, dot, mix, normalize, clamp

@njit(fastmath=True)
def fract(x):
    return x - np.floor(x)

@njit(fastmath=True)
def hash33(p):
    p = vec3( dot(p,vec3(127.1,311.7, 74.7)),
              dot(p,vec3(269.5,183.3,246.1)),
              dot(p,vec3(113.5,271.9,124.6)))
    return -1.0 + 2.0*fract(np.sin(p)*43758.5453123)

@njit(fastmath=True)
def noise(p):
    i = np.floor(p)
    f = fract(p)

    # Quintic interpolation
    u = f*f*f*(f*(f*6.0-15.0)+10.0)

    # Gradients
    # We unroll the 8 corners for explicit dot products

    # Bottom Face (z=0)
    n000 = dot(hash33(i + vec3(0.0,0.0,0.0)), f - vec3(0.0,0.0,0.0))
    n100 = dot(hash33(i + vec3(1.0,0.0,0.0)), f - vec3(1.0,0.0,0.0))
    n010 = dot(hash33(i + vec3(0.0,1.0,0.0)), f - vec3(0.0,1.0,0.0))
    n110 = dot(hash33(i + vec3(1.0,1.0,0.0)), f - vec3(1.0,1.0,0.0))

    # Top Face (z=1)
    n001 = dot(hash33(i + vec3(0.0,0.0,1.0)), f - vec3(0.0,0.0,1.0))
    n101 = dot(hash33(i + vec3(1.0,0.0,1.0)), f - vec3(1.0,0.0,1.0))
    n011 = dot(hash33(i + vec3(0.0,1.0,1.0)), f - vec3(0.0,1.0,1.0))
    n111 = dot(hash33(i + vec3(1.0,1.0,1.0)), f - vec3(1.0,1.0,1.0))

    # Interpolate
    nx00 = mix(n000, n100, u[0])
    nx10 = mix(n010, n110, u[0])
    nx01 = mix(n001, n101, u[0])
    nx11 = mix(n011, n111, u[0])

    nxy0 = mix(nx00, nx10, u[1])
    nxy1 = mix(nx01, nx11, u[1])

    return mix(nxy0, nxy1, u[2])

@njit(fastmath=True)
def fbm(p):
    v = 0.0
    a = 0.5
    shift = vec3(100.0, 100.0, 100.0)
    pp = p
    for i in range(5):
        v += a * noise(pp)
        pp = pp * 2.0 + shift
        a *= 0.5
    return v

@njit(fastmath=True)
def sdCapsule(p, a, b, r):
    pa = p - a
    ba = b - a
    h = clamp(np.dot(pa, ba) / np.dot(ba, ba), 0.0, 1.0)
    return length(pa - ba * h) - r

@njit(fastmath=True)
def sdRoundCone(p, a, b, r1, r2):
    ba = b - a
    l2 = dot(ba,ba)
    rr = r1 - r2
    a2 = l2 - rr*rr
    il2 = 1.0/l2

    pa = p - a
    y = dot(pa,ba)
    z = y - l2

    # pa*l2 - ba*y
    term1 = vec3(
        pa[0]*l2 - ba[0]*y,
        pa[1]*l2 - ba[1]*y,
        pa[2]*l2 - ba[2]*y
    )
    x2 = dot(term1, term1)

    y2 = y*y*l2
    z2 = z*z*l2

    k = np.sign(rr)*rr*rr*x2

    if np.sign(z)*a2*z2 > k: return np.sqrt(x2 + z2)*il2 - r2
    if np.sign(y)*a2*y2 < k: return np.sqrt(x2 + y2)*il2 - r1

    return (np.sqrt(x2*a2 + y2*rr*rr) - y*rr)*il2 - r1

@njit(fastmath=True)
def smin_local(a, b, k):
    h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return mix(b, a, h) - k * h * (1.0 - h)

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
    # Organic Tree with Detail (Bark)

    # 1. Bark Texture (Displacement)
    # Scale p for noise
    noise_p = p * 4.0
    # Reduce fbm complexity implicitly by relying on simplified fbm function in this file if needed
    # But since fbm is already implemented, we use it.
    # To avoid timeout, we use fewer noise octaves or just one noise call?
    # fbm has a loop of 5. Let's use a simpler noise.

    displacement = noise(noise_p) * 0.02  # Single octave noise for speed

    # 2. Structure
    d = 1e9

    # Parameters
    iters = 6 # Increase slightly for better shape

    # Start at origin
    curr_p = p

    # Root flare (simple smin with a sphere or cone at base?)
    # Just let the first segment be thick.

    # Dimensions
    h = 1.2
    r = 0.15

    # Decay
    h_decay = 0.75
    r_decay = 0.65

    for i in range(iters):
        # Segment
        # Tapered cylinder from (0,0,0) to (0,h,0)
        # We start the segment slightly below 0 to ensure connection if folded

        p_seg = curr_p

        # Round Cone
        # Use a slightly curved segment? No, straight is fine with noise.
        d_seg = sdRoundCone(p_seg, vec3(0,0,0), vec3(0,h,0), r, r*r_decay)

        if i == 0:
            d = d_seg
        else:
            # Smooth Union
            k = 0.1 * pow(0.8, float(i))
            d = smin_local(d, d_seg, k)

        # Transform for next generation

        # Move up the branch
        # We move almost to the end.
        curr_p = curr_p - vec3(0, h * 0.85, 0)

        # Rotate (Spiral / Twist)
        # To avoid perfect symmetry, we can add a phase shift
        twist = 1.8 + 0.1 * np.sin(float(i))
        curr_p = rotateY(curr_p, twist)

        # Branching (Fold)
        # Fold X to create bifurcation
        curr_p = vec3(abs(curr_p[0]), curr_p[1], abs(curr_p[2]))

        # Fold Z? If we fold Z too, we get 4 branches.
        # Tree.png has many branches. 4 might be okay if spread out.
        # But it creates a "bush".
        # Let's try folding ONLY X, but rotating Y more?
        # If we only fold X, we get 2 branches per node.
        # Let's try just X fold.

        # Angle out
        spread = 0.6 + 0.1*np.cos(float(i)*2.0)
        curr_p = rotateZ(curr_p, spread)

        # Decay dimensions
        h *= h_decay
        r *= r_decay

    # Apply bark
    # Reduce displacement effect on distance to avoid artifacts
    return d - displacement * 0.5
