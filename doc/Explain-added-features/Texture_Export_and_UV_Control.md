# Texture Export and UV Control Implementation Guide

**Date:** November 2025
**Project:** Kalpana3D Engine

This document explains the technical implementation of the **Texture Export** and **UV Control** features added to the Kalpana3D engine. It serves as a reference for understanding how we solved the challenges of exporting procedural geometry with textures to standard OBJ/MTL formats.

---

## 1. The Challenge

The Kalpana3D engine is a **Signed Distance Field (SDF)** based procedural engine. Unlike traditional mesh-based engines, it does not store geometry as vertices and triangles with pre-defined UV coordinates.

**Key Problems:**
1.  **No UVs:** SDFs are mathematical functions. When we convert them to a mesh (Marching Cubes), the resulting triangles have no texture coordinates (UVs). Without UVs, textures cannot be mapped onto the object in external software (like Blender).
2.  **Texture Paths:** The engine loaded textures into memory (arrays) but didn't track where they came from. The exporter needs the original file paths to copy them.
3.  **Control:** The user needed a way to control the size/tiling of textures directly from the YAML scene file.

---

## 2. Solution Overview

We implemented a pipeline that flows from the YAML Parser -> Engine Core -> Mesher/Exporter.

### Architecture Changes

| Component | File | Change |
| :--- | :--- | :--- |
| **Parser** | `utils/parser.py` | Added tracking of `texture_paths` and reading of `UVScale`. |
| **Engine** | `core/engine.py` | Updated rendering logic to use `UVScale` for consistent previews. |
| **Mesher** | `core/mesher.py` | Implemented "Box Projection" UV generation during export. |

---

## 3. Technical Implementation Details

### A. Texture Path Tracking (`parser.py`)

We modified `parse_scene` to store the original file paths of loaded textures.

```python
# kalpana3d/utils/parser.py

# New list to track paths
texture_paths = []

def load_texture(path):
    # ... loading logic ...
    texture_paths.append(path) # Store path
    return float(len(texture_paths) - 1)

# Return texture_paths in the scene dictionary
return {
    # ...
    'texture_paths': texture_paths,
}
```

**Why?** The exporter (`export_test_scenes.py`) uses this list to find and copy the texture files (`.png`, `.jpg`) to the output directory alongside the OBJ file.

### B. UV Generation: Box Projection (`mesher.py`)

Since we don't have explicit UVs, we generate them mathematically using **Box Projection** (also known as Triplanar Mapping logic). This maps the texture based on the direction the surface is facing.

We added a helper function `calc_uv` in `mesher.py`:

```python
# kalpana3d/core/mesher.py

def calc_uv(p, n, scale):
    # p = Vertex Position, n = Vertex Normal
    
    # 1. Find dominant axis (Is the surface facing X, Y, or Z?)
    a = np.abs(n)
    axis = 0
    if a[1] > a[0] and a[1] > a[2]: axis = 1 # Y-axis (Top/Bottom)
    if a[2] > a[0] and a[2] > a[1]: axis = 2 # Z-axis (Front/Back)
    
    uv = np.zeros(2, dtype=np.float32)
    
    # 2. Project position onto the plane perpendicular to the axis
    if axis == 0: # Facing X -> Map YZ plane
        uv[0] = p[2] * scale
        uv[1] = p[1] * scale
    elif axis == 1: # Facing Y -> Map XZ plane
        uv[0] = p[0] * scale
        uv[1] = p[2] * scale
    else: # Facing Z -> Map XY plane
        uv[0] = p[0] * scale
        uv[1] = p[1] * scale
        
    return uv
```

**Why?** This ensures that textures look correct on box-like shapes without needing manual unwrapping. The `scale` parameter controls how many times the texture repeats.

### C. UV Control (`UVScale`)

We added a new parameter `UVScale` to the material definition in YAML.

**1. YAML Input:**
```yaml
Materials:
  - Color: [1, 1, 1]
    AlbedoMap: textures/brick.png
    UVScale: 2.0 # User controls this!
```

**2. Parser (`parser.py`):**
Reads `UVScale` (default 0.5) and appends it to the material data array (index 21).

**3. Engine (`engine.py`):**
Updated `triplanar_sample` to use this scale so the real-time preview matches the export.

**4. Exporter (`mesher.py`):**
Reads the scale from the material array and passes it to `calc_uv`.

```python
# kalpana3d/core/mesher.py inside export_obj_mesh

# Get UV Scale from material data (index 21)
uv_scale = materials_db[mat_id, 21]

# Generate UV
uv = calc_uv(v, n, uv_scale)

# Write to OBJ
f.write(f"vt {uv[0]:.4f} {uv[1]:.4f}\n")
```

---

## 4. How to Use

1.  **Define Material:** In your YAML scene, add `UVScale` to your material.
2.  **Export:** Run the export script.
3.  **Result:** The OBJ file will have `vt` coordinates scaled according to your setting.

## 5. Future Improvements

-   **Spherical/Cylindrical Mapping:** Currently, we only support Box Projection. For spheres or cylinders, a different projection logic in `calc_uv` would be better.
-   **Seam Correction:** Box projection can create visible seams at 45-degree angles. This is acceptable for now but could be improved with blending logic (though difficult to export to standard OBJ UVs).

---

## 6. Debugging Techniques

During the implementation, we encountered issues with **Numba compilation errors** and **UTF-16 encoded output** from PowerShell. Here are the command-line techniques we used to solve them.

### A. Reading UTF-16 Encoded Files
PowerShell often saves output files (like `output.txt`) in UTF-16LE encoding. Standard tools might display this as "garbage" or with extra spaces. We used Python to read and decode it correctly:

```powershell
python -c "print(open('output.txt', 'rb').read().decode('utf-16'))"
```
*   **Why:** `rb` reads binary mode, avoiding default encoding assumptions. `.decode('utf-16')` converts it to readable text.

### B. Filtering Large Log Files
When looking for specific debug prints (e.g., checking array shapes) inside a massive error log, we used a Python one-liner to filter lines:

```powershell
python -c "print([l for l in open('output.txt', 'rb').read().decode('utf-16').splitlines() if 'Shape' in l])"
```
*   **Result:** Prints only lines containing the word "Shape", helping us instantly verify data dimensions:
    ```
    ['Materials Shape: (1, 22)', 'AABB Mins Shape: (1, 3)', 'AABB Maxs Shape: (1, 3)']
    ```

### C. Capturing Full Output
To ensure we captured both standard output (prints) and errors (tracebacks) without truncation, we used:

```powershell
python -u Test/export_test_scenes.py > output.txt 2>&1
```
*   **`-u`**: Unbuffered binary stdout/stderr (prevents output from getting stuck in a buffer).
*   **`> output.txt`**: Redirects standard output to file.
*   **`2>&1`**: Redirects stderr (errors) to the same place as stdout.
