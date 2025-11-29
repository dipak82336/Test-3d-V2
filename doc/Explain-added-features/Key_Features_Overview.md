# Kalpana3D: Key Features Overview

**Date:** November 2025
**Project:** Kalpana3D Procedural Engine

This document provides a concise summary of the core features implemented in the engine, with key code snippets to illustrate how they work.

---

## 1. Core Engine (SDF & Raymarching)

### Signed Distance Fields (SDF)
We define geometry using mathematical functions that return the distance to the surface.

**Key Code (`core/engine.py`):**
```python
# Sphere SDF
def sdSphere(p, s):
    return length(p) - s

# Box SDF
def sdBox(p, b):
    q = abs(p) - b
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0)
```

### Raymarching Loop
Instead of rasterizing triangles, we "march" rays step-by-step until they hit a surface (`d < SURF_DIST`).

**Key Code (`core/engine.py`):**
```python
for i in range(MAX_STEPS):
    p = ro + rd * d0
    dS = map_scene(p, ...)[0] # Get distance to nearest object
    d0 += dS
    if d0 > MAX_DIST or dS < SURF_DIST: break
```

---

## 2. PBR Material System

### Physically Based Rendering
We use the **Cook-Torrance** model with **GGX** distribution for realistic lighting.

**Key Code (`core/engine.py`):**
```python
# Fresnel-Schlick (Reflection vs Refraction)
F = F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0)

# GGX Distribution (Roughness)
D = NDF_GGX(N, H, roughness)

# Geometry Shadowing
G = G_SchlickGGX(N, V, roughness)

# Specular Term
Specular = (D * G * F) / (4 * NdotV * NdotL)
```

### Advanced Material Features
*   **Transmission (Glass):** Approximated by mixing background color based on `transmission` factor.
*   **Clear Coat:** A second specular layer on top of the base layer (for car paint).
*   **Anisotropy:** Stretches highlights for brushed metal effects.
*   **Neon Emission:** View-dependent "hot core" gradient.

---

## 3. Texturing & UVs

### Triplanar Mapping
Since SDFs don't have UVs, we project textures from 3 directions (X, Y, Z) and blend them based on the normal.

**Key Code (`core/engine.py`):**
```python
# Sample from 3 axes
cx = sample(texture, p.yz * scale)
cy = sample(texture, p.xz * scale)
cz = sample(texture, p.xy * scale)

# Blend based on normal
w = abs(n)
w = w / (w.x + w.y + w.z)
color = cx*w.x + cy*w.y + cz*w.z
```

### UV Export
For OBJ export, we generate explicit UV coordinates using "Box Projection".

**Key Code (`core/mesher.py`):**
```python
# Determine dominant axis
axis = argmax(abs(n))

# Project UVs
if axis == 0: uv = p.yz * scale
elif axis == 1: uv = p.xz * scale
else: uv = p.xy * scale
```

---

## 4. Lighting & Environment

### HDRI Environment Maps
We use high-dynamic-range images for realistic background lighting and reflections.

**Key Code (`core/engine.py`):**
```python
# Equirectangular mapping
phi = atan2(d.z, d.x)
theta = asin(d.y)
uv = vec2(0.5 + phi/(2*PI), 0.5 + theta/PI)
color = sample_hdri(uv)
```

### Soft Shadows
We trace a ray towards the light and check how close it passes to objects (`k*h/t`).

**Key Code (`core/engine.py`):**
```python
res = 1.0
for i in range(16):
    h = map_scene(p + lightDir*t, ...)
    res = min(res, k * h / t) # Penumbra factor
```

---

## 5. Optimization

### AABB Culling
We calculate Axis-Aligned Bounding Boxes for objects and skip expensive SDF evaluations if the ray is far away.

**Key Code (`core/engine.py`):**
```python
# Check distance to AABB
dx = max(max(min.x - p.x, p.x - max.x), 0.0)
...
d_aabb = length(vec3(dx, dy, dz))

if d_aabb > d_current_best: continue # Skip object
```

### Empty Space Skipping (Mesher)
When generating the mesh, if a voxel is far from any surface, we skip it entirely.

**Key Code (`core/mesher.py`):**
```python
d = map_scene(center, ...)
if d > voxel_diagonal: continue # Skip voxel
```
