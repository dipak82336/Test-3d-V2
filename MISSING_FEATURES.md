# Kalpana3D Feature Roadmap & Status

**Date:** November 2025
**Status:** Phase 2 Complete

---

## ✅ Phase 1: Material System (Completed)

We successfully implemented a high-end PBR material system with the following capabilities:

1.  **Subsurface Scattering (SSS)**: For wax, marble, skin.
2.  **Procedural Bump/Normal Mapping**: For surface detail without textures.
3.  **Anisotropy**: For brushed metals (pots, pans).
4.  **Clear Coat**: For car paint and lacquered wood.
5.  **Sheen**: For velvet and fabrics.
6.  **Displacement**: True geometric deformation.

---

## ✅ Phase 2: Realism & Export (Completed)

We addressed the critical weaknesses identified in Phase 1:

1.  **HDRI Environment Lighting**: 
    *   **Status**: Implemented.
    *   **Result**: Realistic reflections and ambient lighting. Metals now look correct.
2.  **Texture Mapping & UVs**:
    *   **Status**: Implemented.
    *   **Result**: Support for Albedo, Roughness, and Normal maps using Triplanar/Box projection.
3.  **OBJ Export**:
    *   **Status**: Implemented.
    *   **Result**: Scenes can be exported to standard 3D formats with textures and UVs.

---

## 🚧 Phase 3: Missing Features (The "Wishlist")

These are the advanced features that are currently missing and would take the engine to the next level (v2.0).

### 1. Optimization (BVH / Spatial Partitioning) [CRITICAL]
*   **Current State**: The engine checks every object for every ray step (Linear complexity).
*   **Problem**: Performance drops drastically with >20 objects.
*   **Solution**: Implement a Bounding Volume Hierarchy (BVH) or Octree to skip objects that are far away.

### 2. Global Illumination (GI)
*   **Current State**: Direct lighting + Single bounce reflection. Shadows are pitch black (unless filled by ambient light).
*   **Problem**: No "color bleeding" or bounced light. Interiors look fake.
*   **Solution**: Path Tracing or Irradiance Caching to calculate multi-bounce lighting.

### 3. Advanced UV Mapping
*   **Current State**: Box Projection (Triplanar) only.
*   **Problem**: Textures look wrong on Spheres (pinching) or Cylinders (stretching).
*   **Solution**: Implement Spherical and Cylindrical projection modes in `calc_uv`.

### 4. Animation System
*   **Current State**: Static images.
*   **Problem**: Cannot render moving objects or camera fly-throughs.
*   **Solution**: Add a `Time` parameter to the YAML and engine, and a frame-rendering loop.

### 5. Post-Processing
*   **Current State**: Raw pixel output.
*   **Problem**: Bright lights clip to white (no Tone Mapping). No Bloom for neon lights.
*   **Solution**: Add a post-processing pass for Bloom, ACES Tone Mapping, and Color Grading.

### 6. Advanced Shapes (Fractals)
*   **Current State**: Basic primitives (Sphere, Box, Torus).
*   **Problem**: Limited to simple geometry.
*   **Solution**: Add Mandelbulb, Menger Sponge, and other fractal SDFs for infinite detail.

### 7. Physics
*   **Current State**: Static objects.
*   **Problem**: Objects float or intersect; no gravity or collisions.
*   **Solution**: Implement a basic rigid body simulation step before rendering.
