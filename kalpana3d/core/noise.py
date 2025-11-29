import numpy as np
from numba import njit
from kalpana3d.math_core import vec3, dot, mix

@njit(fastmath=True)
def hash13(p):
    # p is vec3
    # Ensure p is float32
    h = dot(p, vec3(12.9898, 78.233, 45.164))
    # fract(sin(h)*43758.5453123)
    val = np.sin(h) * 43758.5453123
    return val - np.floor(val)

@njit(fastmath=True)
def noise(p):
    i = np.floor(p)
    f = p - i
    
    # Cubic smoothing
    # f * f * (3.0 - 2.0 * f)
    u = f * f * (3.0 - 2.0 * f)
    
    # 8 corners
    # We need to hash the corners
    # This is slow if we call hash13 8 times.
    
    res = mix(mix(mix( hash13(i + vec3(0.0,0.0,0.0)), 
                        hash13(i + vec3(1.0,0.0,0.0)), u[0]),
                   mix( hash13(i + vec3(0.0,1.0,0.0)), 
                        hash13(i + vec3(1.0,1.0,0.0)), u[0]), u[1]),
               mix(mix( hash13(i + vec3(0.0,0.0,1.0)), 
                        hash13(i + vec3(1.0,0.0,1.0)), u[0]),
                   mix( hash13(i + vec3(0.0,1.0,1.0)), 
                        hash13(i + vec3(1.0,1.0,1.0)), u[0]), u[1]), u[2])
    return res

@njit(fastmath=True)
def fbm(p, octaves):
    v = 0.0
    a = 0.5
    shift = vec3(100.0, 100.0, 100.0)
    # Numba loop
    for i in range(octaves):
        v += a * noise(p)
        p = p * 2.0 + shift
        a *= 0.5
    return v
