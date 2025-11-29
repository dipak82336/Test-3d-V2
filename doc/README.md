# Kalpana3D Engine Documentation

**Version:** 1.0 (November 2025)
**Type:** Procedural SDF Ray-Marching Engine

## 1. Introduction

Kalpana3D is a high-performance, pure Python 3D engine that uses **Signed Distance Fields (SDFs)** and **Ray Marching** to generate geometry. Unlike traditional engines that use triangles, Kalpana3D defines shapes using mathematical functions, allowing for infinite resolution, smooth blending, and complex boolean operations.

It features a fully **Physically Based Rendering (PBR)** material system, supporting realistic metals, glass, and neon lights, and can export generated models to standard **OBJ/MTL** formats with textures.

---

## 2. Getting Started

### Requirements
*   Python 3.8+
*   Numba (for high-performance compilation)
*   NumPy
*   Pillow (PIL) (for texture loading)
*   PyYAML

### Installation
```bash
pip install numba numpy pyyaml pillow
```

### Running a Scene
To render a scene defined in a YAML file:
```bash
python main.py
```
*(Note: You may need to edit `main.py` to point to your specific YAML file)*

### Exporting to OBJ
To convert your procedural scene to a 3D model (OBJ) for use in Blender/Unity:
```bash
python Test/export_test_scenes.py
```
This will generate `.obj`, `.mtl`, and texture files in `gallery/models/`.

---

## 3. The YAML Scene Format

The engine is data-driven. You define your scene in a `.yaml` file. Here is the complete format structure.

### A. Environment
Controls the background lighting.
```yaml
EnvironmentMap: textures/studio_small.jpg  # Path to HDRI image
```

### B. Materials
Define the look of your objects using PBR parameters.

```yaml
Materials:
  - Color: [1.0, 0.0, 0.0]        # RGB Base Color (0-1)
    Metallic: 0.0                 # 0 = Plastic/Dielectric, 1 = Metal
    Roughness: 0.5                # 0 = Smooth/Mirror, 1 = Matte/Rough
    
    # Advanced PBR
    Transmission: 0.0             # 1.0 = Glass (Transparent)
    IOR: 1.45                     # Index of Refraction (1.45 = Glass, 2.4 = Diamond)
    Emission: [0, 0, 0]           # RGB Light Emission (Neon)
    
    # Surface Detail
    NormalMap: textures/brick_normal.png
    AlbedoMap: textures/brick_albedo.png
    UVScale: 2.0                  # Texture tiling scale (Higher = smaller tiles)
    Displacement: 0.0             # Geometric displacement strength
    
    # Special Effects
    ClearCoat: 0.0                # Secondary specular layer (Car Paint)
    Sheen: 0.0                    # Velvet/Cloth edge lighting
    Anisotropic: 0.0              # Brushed metal effect
```

### C. Objects
Define the geometry using SDF primitives.

```yaml
Objects:
  - Type: 1             # Shape Type (See list below)
    Pos: [0, 0, 0]      # Position [X, Y, Z]
    Rot: [0, 0, 0]      # Rotation [X, Y, Z] in degrees
    Scale: [1, 1, 1]    # Size [X, Y, Z]
    Mat: 0              # Material Index (0 refers to the first material above)
    Op: 0               # Boolean Operation (See list below)
```

#### Shape Types (`Type`)
| ID | Shape | Scale Parameter Usage |
| :--- | :--- | :--- |
| `0` | **Sphere** | `Scale[0]` = Radius |
| `1` | **Box** | `Scale` = Half-extents (x, y, z) |
| `2` | **Cylinder** | `Scale[0]` = Radius, `Scale[1]` = Height |
| `3` | **RoundBox** | `Scale` = Extents (plus internal rounding radius) |
| `4` | **Plane** | Infinite floor plane |
| `5` | **Torus** | `Scale[0]` = Main Radius, `Scale[1]` = Ring Radius |
| `6` | **Capsule** | `Scale[0]` = Radius, `Scale[1]` = Height |

#### Boolean Operations (`Op`)
| ID | Operation | Description |
| :--- | :--- | :--- |
| `0` | **Union** | Combines shapes (A + B) |
| `1` | **Subtract** | Cuts this shape out of previous ones (A - B) |
| `2` | **Intersect** | Keeps only the overlapping part (A & B) |
| `3` | **Smooth Union** | Blends shapes together like liquid |
| `4` | **Smooth Sub** | Smoothly cuts away |
| `5` | **Smooth Int** | Smoothly intersects |
| `6` | **Repeat** | Infinitely repeats the shape in space |

---

## 4. Key Features

For detailed technical explanations of specific features, please refer to the following documents in `doc/Explain-added-features/`:

*   **[Key Features Overview](Explain-added-features/Key_Features_Overview.md)**: A quick cheat-sheet of the engine's core capabilities.
*   **[PBR Material System](Explain-added-features/PBR_Material_System.md)**: Deep dive into the physics of lighting, metals, glass, and neon.
*   **[Texture Export & UV Control](Explain-added-features/Texture_Export_and_UV_Control.md)**: How we generate UVs for procedural shapes and export them to OBJ.

---

## 5. Troubleshooting

*   **"Numba Compilation Error"**: Ensure all your material parameters are floats (e.g., `1.0` not `1`).
*   **"Texture Not Found"**: Check the paths in your YAML file. They should be relative to the YAML file or the project root.
*   **"Garbage Output in Terminal"**: If you see weird characters, use the Python one-liners described in the [Debugging Techniques](Explain-added-features/Texture_Export_and_UV_Control.md#6-debugging-techniques) section.
