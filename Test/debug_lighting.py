import sys
import os
import numpy as np
from numba import njit

# Add Library to Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kalpana3d.core.engine import get_lighting

# Dummy data
p = np.array([0, 0, 0], dtype=np.float32)
n = np.array([0, 1, 0], dtype=np.float32)
rd = np.array([0, -1, 0], dtype=np.float32)
mat_id = 0.0
materials = np.zeros((2, 15), dtype=np.float32)
lights = np.zeros((1, 7), dtype=np.float32)
num_objects = 1
object_types = np.array([0], dtype=np.int32)
inv_matrices = np.zeros((1, 4, 4), dtype=np.float32)
scales = np.zeros((1, 3), dtype=np.float32)
material_ids = np.array([0], dtype=np.float32)
operations = np.array([0], dtype=np.float32)

print("Calling get_lighting...")
try:
    col = get_lighting(p, n, rd, mat_id, materials, lights, num_objects, object_types, inv_matrices, scales, material_ids, operations)
    print("Success:", col)
except Exception as e:
    print("Failed:", e)
