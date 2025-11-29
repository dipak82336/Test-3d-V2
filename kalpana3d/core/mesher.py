import numpy as np
from numba import njit, prange
from kalpana3d.core.engine import map_scene_new, calc_normal, get_lighting
from kalpana3d.core.math import vec3, mix, normalize
from kalpana3d.core.marching_cubes_tables import edge_table, tri_table

@njit(fastmath=True)
def vertex_interp(iso_level, p1, p2, val1, val2):
    iso = np.float32(iso_level)
    eps = np.float32(0.00001)
    if abs(iso - val1) < eps: return vec3(p1[0], p1[1], p1[2])
    if abs(iso - val2) < eps: return vec3(p2[0], p2[1], p2[2])
    if abs(val1 - val2) < eps: return vec3(p1[0], p1[1], p1[2])
    mu = (iso - val1) / (val2 - val1)
    return vec3(
        p1[0] + mu * (p2[0] - p1[0]),
        p1[1] + mu * (p2[1] - p1[1]),
        p1[2] + mu * (p2[2] - p1[2])
    )

@njit(parallel=True, fastmath=True)
def generate_mesh(min_bound, max_bound, resolution, 
                  num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, lights, aabb_mins, aabb_maxs):
    
    # Setup grid
    step = resolution
    nx = int((max_bound[0] - min_bound[0]) / step)
    ny = int((max_bound[1] - min_bound[1]) / step)
    nz = int((max_bound[2] - min_bound[2]) / step)
    
    # Pre-allocate (estimate)
    max_tris = nx * ny * nz * 2 # Conservative estimate
    vertices = np.zeros((max_tris * 3, 3), dtype=np.float32)
    normals = np.zeros((max_tris * 3, 3), dtype=np.float32)
    tri_materials = np.zeros(max_tris, dtype=np.int32)
    
    tri_idx = 0
    
    # Offsets for Bourke's indexing
    offsets = np.array([
        [0,0,0], [1,0,0], [1,0,1], [0,0,1],
        [0,1,0], [1,1,0], [1,1,1], [0,1,1]
    ], dtype=np.float32) * step

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if tri_idx >= max_tris: break
                
                pos = vec3(min_bound[0] + x*step, min_bound[1] + y*step, min_bound[2] + z*step)
                
                vals = np.zeros(8, dtype=np.float32)
                p = np.zeros((8, 3), dtype=np.float32)
                
                # Optimization: Empty Space Skipping
                cube_index = 0
                
                # Evaluate corner 0 first
                p0 = pos + offsets[0]
                d0, _ = map_scene_new(p0, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, aabb_mins, aabb_maxs)
                
                # If distance is greater than diagonal of voxel, surface cannot intersect
                # Diagonal = sqrt(3) * step ~= 1.732 * step
                # Use slightly larger safety margin
                if d0 > step * 1.75:
                    continue
                    
                vals[0] = d0
                if d0 < 0.0: cube_index |= 1
                
                p[0] = p0
                
                # Evaluate remaining 7 corners
                for i in range(1, 8):
                    p[i] = pos + offsets[i]
                    d, _ = map_scene_new(p[i], num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, aabb_mins, aabb_maxs)
                    vals[i] = d
                    if d < 0.0: # Iso-level 0
                        cube_index |= (1 << i)
                
                if edge_table[cube_index] == 0:
                    continue
                
                # Compute vertices on edges
                vert_list = np.zeros((12, 3), dtype=np.float32)
                
                if edge_table[cube_index] & 1: vert_list[0] = vertex_interp(0.0, p[0], p[1], vals[0], vals[1])
                if edge_table[cube_index] & 2: vert_list[1] = vertex_interp(0.0, p[1], p[2], vals[1], vals[2])
                if edge_table[cube_index] & 4: vert_list[2] = vertex_interp(0.0, p[2], p[3], vals[2], vals[3])
                if edge_table[cube_index] & 8: vert_list[3] = vertex_interp(0.0, p[3], p[0], vals[3], vals[0])
                if edge_table[cube_index] & 16: vert_list[4] = vertex_interp(0.0, p[4], p[5], vals[4], vals[5])
                if edge_table[cube_index] & 32: vert_list[5] = vertex_interp(0.0, p[5], p[6], vals[5], vals[6])
                if edge_table[cube_index] & 64: vert_list[6] = vertex_interp(0.0, p[6], p[7], vals[6], vals[7])
                if edge_table[cube_index] & 128: vert_list[7] = vertex_interp(0.0, p[7], p[4], vals[7], vals[4])
                if edge_table[cube_index] & 256: vert_list[8] = vertex_interp(0.0, p[0], p[4], vals[0], vals[4])
                if edge_table[cube_index] & 512: vert_list[9] = vertex_interp(0.0, p[1], p[5], vals[1], vals[5])
                if edge_table[cube_index] & 1024: vert_list[10] = vertex_interp(0.0, p[2], p[6], vals[2], vals[6])
                if edge_table[cube_index] & 2048: vert_list[11] = vertex_interp(0.0, p[3], p[7], vals[3], vals[7])
                
                # Create triangles
                for i in range(0, 16, 3):
                    if tri_table[cube_index, i] == -1: break
                    
                    v1 = vert_list[tri_table[cube_index, i]]
                    v2 = vert_list[tri_table[cube_index, i+1]]
                    v3 = vert_list[tri_table[cube_index, i+2]]
                    
                    # Calculate Normal
                    n1 = calc_normal(v1, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, aabb_mins, aabb_maxs)
                    n2 = calc_normal(v2, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, aabb_mins, aabb_maxs)
                    n3 = calc_normal(v3, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, aabb_mins, aabb_maxs)
                    
                    # Get Material ID
                    _, m1 = map_scene_new(v1, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, aabb_mins, aabb_maxs)
                    
                    tri_materials[tri_idx] = int(m1)
                    
                    vertices[tri_idx*3 + 0] = v1
                    vertices[tri_idx*3 + 1] = v2
                    vertices[tri_idx*3 + 2] = v3
                    
                    normals[tri_idx*3 + 0] = n1
                    normals[tri_idx*3 + 1] = n2
                    normals[tri_idx*3 + 2] = n3
                    
                    tri_idx += 1
                    
    return vertices[:tri_idx*3], normals[:tri_idx*3], tri_materials[:tri_idx]

def calc_uv(p, n, scale):
    # Box Mapping / Triplanar UVs
    # 0: X-axis, 1: Y-axis, 2: Z-axis
    a = np.abs(n)
    axis = 0
    if a[1] > a[0] and a[1] > a[2]: axis = 1
    if a[2] > a[0] and a[2] > a[1]: axis = 2
    
    uv = np.zeros(2, dtype=np.float32)
    
    if axis == 0:
        uv[0] = p[2] * scale
        uv[1] = p[1] * scale
    elif axis == 1:
        uv[0] = p[0] * scale
        uv[1] = p[2] * scale
    else:
        uv[0] = p[0] * scale
        uv[1] = p[1] * scale
        
    return uv

def export_obj_mesh(filename, vertices, normals, tri_materials, materials_db, texture_paths=None):
    import os
    
    # Generate MTL filename
    mtl_filename = filename.replace('.obj', '.mtl')
    mtl_name = os.path.basename(mtl_filename)
    
    # Write MTL File
    with open(mtl_filename, 'w') as f:
        f.write(f"# Material Library for {os.path.basename(filename)}\n")
        
        for i in range(len(materials_db)):
            # Extract PBR properties
            albedo = materials_db[i, 0:3]
            metallic = materials_db[i, 3]
            roughness = materials_db[i, 4]
            emission = materials_db[i, 5:8]
            
            # Convert PBR to Phong (Approximate)
            
            # Diffuse (Kd): 
            # "Artistic" Realism for simple viewers.
            
            # Diffuse (Kd) & Specular (Ks) - COMPATIBILITY MODE (ARTISTIC)
            # Problem: Strict physics (Black Diffuse for Metal) looks like a black void in simple viewers.
            # Solution: We "hack" the Diffuse to be the Albedo color so it looks like metal even without reflections.
            
            # Diffuse (Kd) & Specular (Ks) - ARTISTIC BRIGHTNESS (CARTOON/VIVID)
            # Problem: "Realism" looks dark/fake in simple viewers.
            # Solution: Abandon physics. Make it look like a bright, vivid illustration.
            # High Diffuse = Bright Color. High Specular = Shiny.
            
            if metallic > 0.5:
                # CONDUCTOR (Metal - Gold)
                # Diffuse: 0.8 (Very Bright Yellow) - Forces the color to be visible.
                kd = albedo * 0.8
                
                # Specular: 1.0 (Max) - Strong colored reflection.
                ks = albedo * 1.0
                
                # Ambient: 0.5 (High) - Fills shadows with color.
                ka = albedo * 0.5
                
                # Shininess: 60.0 (Distinct Highlight)
                ns = 60.0 
            else:
                # DIELECTRIC (Plastic, Rubber)
                
                if roughness > 0.5:
                    # Matte / Rubber (Red)
                    kd = albedo * 0.9 # Vivid Red
                    ks = np.array([0.0, 0.0, 0.0]) # No shine (Pure Matte)
                    ns = 0.0 
                    ka = kd * 0.3 # Bright shadows
                else:
                    # Shiny Plastic (Blue)
                    kd = albedo * 0.8 # Vivid Blue
                    f0 = 0.3 # White highlight
                    ks = np.array([f0, f0, f0])
                    ns = 300.0 # Sharp gloss
                    ka = kd * 0.3 # Bright shadows
            
            f.write(f"\nnewmtl mat_{i}\n")
            f.write(f"Kd {kd[0]:.4f} {kd[1]:.4f} {kd[2]:.4f}\n")
            f.write(f"Ks {ks[0]:.4f} {ks[1]:.4f} {ks[2]:.4f}\n")
            f.write(f"Ns {ns:.4f}\n")
            f.write(f"Ka {ka[0]:.4f} {ka[1]:.4f} {ka[2]:.4f}\n")
            f.write(f"d 1.0\n")
            f.write(f"illum 2\n") # Highlight on
            
            # Texture Maps
            # Indices: Albedo=18, Roughness=19, Normal=20
            # Bump Strength=11
            
            tex_albedo_id = int(materials_db[i, 18])
            tex_rough_id = int(materials_db[i, 19])
            tex_normal_id = int(materials_db[i, 20])
            bump_strength = materials_db[i, 11]
            
            if texture_paths:
                if tex_albedo_id >= 0 and tex_albedo_id < len(texture_paths):
                    f.write(f"map_Kd {os.path.basename(texture_paths[tex_albedo_id])}\n")
                    
                if tex_normal_id >= 0 and tex_normal_id < len(texture_paths):
                    f.write(f"map_Bump {os.path.basename(texture_paths[tex_normal_id])}\n")
                elif tex_rough_id >= 0 and tex_rough_id < len(texture_paths) and bump_strength > 0.0:
                    # Use roughness map as bump if bump strength is active
                    f.write(f"map_Bump {os.path.basename(texture_paths[tex_rough_id])}\n")

    # Write OBJ File
    with open(filename, 'w') as f:
        f.write(f"# Exported from Kalpana3D\n")
        f.write(f"mtllib {mtl_name}\n")
        f.write(f"o Mesh\n")
        
        # Vertices
        for v in vertices:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            
        # UVs (Texture Coordinates)
        # We need to calculate UVs for each vertex based on its material's UVScale
        # Since vertices are shared, we might have seams issues if we just use vertex normal.
        # But for now, let's just calculate per vertex.
        
        # We need to look up the material for each vertex. 
        # But vertices are shared across triangles which might have different materials?
        # Actually generate_mesh returns vertices for triangles, but they are indexed?
        # No, generate_mesh returns flat list of vertices (tri_idx * 3).
        # So vertices are NOT shared in the returned array! They are unique per triangle vertex.
        # This is perfect for UVs (no sharing issues).
        
        for i in range(len(vertices)):
            v = vertices[i]
            n = normals[i]
            
            # Find which triangle this vertex belongs to
            tri_idx = i // 3
            mat_id = tri_materials[tri_idx]
            
            # Get UV Scale
            uv_scale = 0.5
            if mat_id < len(materials_db):
                uv_scale = materials_db[mat_id, 21]
                
            uv = calc_uv(v, n, uv_scale)
            f.write(f"vt {uv[0]:.4f} {uv[1]:.4f}\n")
            
        # Normals
        for n in normals:
            f.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
            
        # Faces (Grouped by Material)
        num_tris = len(vertices) // 3
        
        # Create a list of (material_id, tri_index)
        tri_indices = np.arange(num_tris)
        # Sort by material
        sorted_indices = np.argsort(tri_materials)
        
        current_mat = -1
        
        for i in sorted_indices:
            mat_id = tri_materials[i]
            
            if mat_id != current_mat:
                f.write(f"usemtl mat_{mat_id}\n")
                current_mat = mat_id
                
            idx = i * 3 + 1
            # Front face: v/vt/vn
            f.write(f"f {idx}/{idx}/{idx} {idx+1}/{idx+1}/{idx+1} {idx+2}/{idx+2}/{idx+2}\n")
            # Back face (Double sided) - Optional, but good for open surfaces
            # f.write(f"f {idx+2}//{idx+2} {idx+1}//{idx+1} {idx}//{idx}\n")
