import numpy as np
from numba import njit

# Material Structure Layout:
# [R, G, B, Metallic, Roughness, 0, 0, 0] (8 floats per material)
MATERIAL_SIZE = 8

@njit(fastmath=True)
def create_material(r, g, b, metallic, roughness):
    mat = np.zeros(MATERIAL_SIZE, dtype=np.float32)
    mat[0] = r
    mat[1] = g
    mat[2] = b
    mat[3] = metallic
    mat[4] = roughness
    return mat

@njit(fastmath=True)
def get_material_color(materials, mat_id):
    idx = int(mat_id)
    return materials[idx, 0:3]

@njit(fastmath=True)
def get_material_props(materials, mat_id):
    # Returns (Metallic, Roughness)
    idx = int(mat_id)
    return materials[idx, 3], materials[idx, 4]
