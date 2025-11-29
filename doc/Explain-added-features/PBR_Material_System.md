# PBR Material System Implementation Guide

**Date:** November 2025
**Project:** Kalpana3D Engine

This document details the implementation of the **Physically Based Rendering (PBR)** material system in the Kalpana3D engine. It explains how we achieved realistic visuals (metals, glass, neon) within a pure Python/Numba ray-marching engine.

---

## 1. The Challenge

Implementing a PBR system in a custom ray-marcher comes with specific challenges:
1.  **Performance:** Calculating complex lighting equations (Fresnel, Distribution, Geometry) for every pixel in Python is too slow. We had to use **Numba** to compile everything to machine code.
2.  **Data Structure:** Numba requires strict typing. We couldn't use Python classes or dictionaries for materials. We had to flatten everything into large NumPy arrays.
3.  **Complex Light Interactions:** Features like **Transmission** (Glass) and **Subsurface Scattering** (Wax/Marble) are difficult to approximate in a simple ray-marcher without full path tracing.

---

## 2. Data Architecture (`parser.py`)

To satisfy Numba's requirements, we flattened the material data into a single 2D float array.

**Material Array Structure (Nx22):**
Each row represents a material, and columns represent properties:

| Index | Property | Description |
| :--- | :--- | :--- |
| 0-2 | **Color** | RGB Albedo (Base Color) |
| 3 | **Metallic** | 0.0 (Dielectric) to 1.0 (Metal) |
| 4 | **Roughness** | 0.0 (Smooth) to 1.0 (Rough) |
| 5-7 | **Emission** | RGB Emissive Color (Light source) |
| 8 | **Transmission** | 0.0 (Opaque) to 1.0 (Transparent/Glass) |
| 9 | **IOR** | Index of Refraction (e.g., 1.45 for Glass) |
| 10 | **SSS** | Subsurface Scattering Strength |
| 11 | **Bump** | Bump Map Strength |
| 12 | **ClearCoat** | Secondary specular layer (Car Paint) |
| 13 | **Anisotropic** | Stretched highlights (Brushed Metal) |
| 14 | **AnisoRot** | Rotation of anisotropy |
| 15 | **Sheen** | Cloth/Velvet edge lighting |
| 16 | **SheenTint** | Tint of the sheen |
| 17 | **Displacement** | Geometric displacement strength |
| 18-20 | **Texture IDs** | Indices for Albedo, Roughness, Normal maps |
| 21 | **UVScale** | Texture tiling scale |

**Why?** This allows `engine.py` to access any property in O(1) time without object overhead, which is critical for the GPU-like performance of Numba.

---

## 3. Rendering Logic (`engine.py`)

The core lighting logic resides in the `get_lighting` function. We implemented a **Cook-Torrance Specular BRDF**.

### A. The "Holy Trinity" of PBR
We combined three main components to calculate how light reflects:

1.  **Distribution (NDF):** *How rough is the surface?*
    *   We used the **GGX (Trowbridge-Reitz)** function.
    *   *Effect:* Controls the size and sharpness of the specular highlight.

2.  **Geometry (G):** *Do micro-facets shadow each other?*
    *   We used the **Schlick-GGX** approximation.
    *   *Effect:* Darkens the reflection at grazing angles on rough surfaces.

3.  **Fresnel (F):** *How much light reflects vs. refracts?*
    *   We used the **Fresnel-Schlick** approximation.
    *   *Effect:* Makes surfaces more reflective at grazing angles (edges).

### B. Special Effects Implementation

#### 1. Neon "Hot Core" Emission
To make neon look like "energy" and not just flat color, we implemented a view-dependent gradient:
```python
# engine.py
if em_intensity > 4.0:
    # Calculate how much we are looking directly at the surface
    ndotv = max(dot(n, view_dir), 0.0)
    
    # Core Mask: High power of NdotV creates a small center spot
    core_mask = float(pow(ndotv, 8.0))
    
    # Mix Vibrant Color (Edges) with White Hot (Center)
    final_col = mix(vibrant_col, white_hot, core_mask)
```
*Result:* A bright white core that fades to a colored glow, simulating plasma.

#### 2. Glass (Transmission)
Since we don't do recursive ray-tracing for performance, we approximated glass:
*   **Reflection:** Standard Fresnel reflection.
*   **Refraction:** We simply mix the background color (or environment map) based on the transmission factor.
*   *Note:* This is a "thin-walled" approximation, perfect for windows or simple gems.

#### 3. Clear Coat (Car Paint)
We added a **second specular layer** on top of the base layer.
*   The base layer handles the colored paint (metallic/roughness).
*   The Clear Coat layer adds a sharp, white reflection on top, simulating the varnish.

---

## 4. Debugging & Optimization

**The Numba Challenge:**
We faced significant "Typing Errors" because Numba infers types at compile time.
*   *Issue:* Passing `None` or mixed types (int/float) caused crashes.
*   *Solution:* We enforced strict `float32` arrays for everything. Even boolean flags (like "has texture") are stored as `-1.0` (False) or `ID` (True) floats.

**Empty Space Skipping:**
For the exporter, we optimized the meshing process by checking the distance field *before* processing a voxel. If the distance is large, we skip the entire voxel, reducing export time from minutes to seconds.

---

## 5. Conclusion

The Kalpana3D material system is a "Uber Shader" implementation. Instead of having different shaders for Plastic, Metal, and Glass, we have **one mathematical model** that can represent all of them simply by changing the parameters (Metallic, Roughness, Transmission). This is the industry standard workflow used in engines like Unreal Engine 5 and Blender Cycles.
