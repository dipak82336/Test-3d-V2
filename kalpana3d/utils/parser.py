import yaml
import numpy as np
import os
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

def make_translation(x, y, z):
    return np.array([[1,0,0,x], [0,1,0,y], [0,0,1,z], [0,0,0,1]], dtype=np.float32)

def make_rotation(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    Rx = np.array([[1,0,0,0], [0,cx,-sx,0], [0,sx,cx,0], [0,0,0,1]], dtype=np.float32)
    Ry = np.array([[cy,0,sy,0], [0,1,0,0], [-sy,0,cy,0], [0,0,0,1]], dtype=np.float32)
    Rz = np.array([[cz,-sz,0,0], [sz,cz,0,0], [0,0,1,0], [0,0,0,1]], dtype=np.float32)
    
    return Rz @ Ry @ Rx

def parse_scene(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    # Arrays for Numba
    object_types = []
    inv_matrices = []
    scales = []
    material_ids = []
    operations = []
    
    materials = []
    lights = []
    
    # ... (Materials/Lights parsing same) ...
    if 'Materials' not in data:
        # Default Grey with 0 SSS/Bump/ClearCoat/Aniso/Sheen/Displacement
        materials.append([0.8, 0.8, 0.8, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0])
    else:
        for m in data['Materials']:
            # [R, G, B, Metallic, Roughness, EmR, EmG, EmB, Transmission, IOR, SSS, Bump, ClearCoat, Aniso, AnisoRot, Sheen, SheenTint, Displacement]
            base = m['Color'] + [m.get('Metallic', 0.0), m.get('Roughness', 0.5)]
            emission = m.get('Emission', [0.0, 0.0, 0.0])
            transmission = [m.get('Transmission', 0.0)]
            ior = [m.get('IOR', 1.45)]
            sss = [m.get('SSS', 0.0)]
            bump = [m.get('Bump', 0.0)]
            clear_coat = [m.get('ClearCoat', 0.0)]
            aniso = [m.get('Anisotropic', 0.0)]
            aniso_rot = [m.get('AnisoRot', 0.0)]
            sheen = [m.get('Sheen', 0.0)]
            sheen_tint = [m.get('SheenTint', 0.5)]  # 0.5 = balanced white/albedo mix
            displacement = [m.get('Displacement', 0.0)]
            materials.append(base + emission + transmission + ior + sss + bump + clear_coat + aniso + aniso_rot + sheen + sheen_tint + displacement)
            
    if 'Lights' not in data:
        lights.append([5, 8, 5, 1, 1, 1, 1.0])
    else:
        for l in data['Lights']:
            lights.append(l['Pos'] + l['Color'] + [l['Intensity']])

    # Parse Objects
    for obj in data['Objects']:
        typ = obj['Type']
        pos = obj.get('Pos', [0,0,0])
        rot = obj.get('Rot', [0,0,0])
        scale = obj.get('Scale', [1,1,1])
        mat_id = obj.get('Mat', 0)
        op = obj.get('Op', 0) # 0=Union, 1=Sub, 2=Int
        
        T = make_translation(*pos)
        R = make_rotation(*rot)
import yaml
import numpy as np
import os
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

def make_translation(x, y, z):
    return np.array([[1,0,0,x], [0,1,0,y], [0,0,1,z], [0,0,0,1]], dtype=np.float32)

def make_rotation(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    Rx = np.array([[1,0,0,0], [0,cx,-sx,0], [0,sx,cx,0], [0,0,0,1]], dtype=np.float32)
    Ry = np.array([[cy,0,sy,0], [0,1,0,0], [-sy,0,cy,0], [0,0,0,1]], dtype=np.float32)
    Rz = np.array([[cz,-sz,0,0], [sz,cz,0,0], [0,0,1,0], [0,0,0,1]], dtype=np.float32)
    
    return Rz @ Ry @ Rx

def calculate_aabb(typ, scale, world_mat):
    # Plane (Type 4) is infinite
    if typ == 4:
        inf = 1e9
        return np.array([-inf, -inf, -inf], dtype=np.float32), np.array([inf, inf, inf], dtype=np.float32)
        
    # Conservative local bounds
    # For simplicity, use a cube that covers the max dimension
    # This is safe for Sphere, Box, etc.
    # For Cylinder (h, r), max(h, r) covers it.
    # For Torus (r1, r2), r1+r2 covers it.
    
    # Refine for Box (Type 1) - scale is exact extents
    if typ == 1:
        lx, ly, lz = scale[0], scale[1], scale[2]
    else:
        # Conservative radius for others
        r = np.max(np.array(scale))
        # Extra padding for Torus/Capsule/etc
        if typ == 5: r = scale[0] + scale[1] # Torus
        lx, ly, lz = r, r, r
        
    corners = np.array([
        [-lx, -ly, -lz, 1],
        [ lx, -ly, -lz, 1],
        [-lx,  ly, -lz, 1],
        [ lx,  ly, -lz, 1],
        [-lx, -ly,  lz, 1],
        [ lx, -ly,  lz, 1],
        [-lx,  ly,  lz, 1],
        [ lx,  ly,  lz, 1]
    ], dtype=np.float32)
    
    # Transform to world
    world_corners = (world_mat @ corners.T).T
    
    # Find min/max
    aabb_min = np.min(world_corners[:, :3], axis=0)
    aabb_max = np.max(world_corners[:, :3], axis=0)
    
    # Add padding for displacement/smooth blending
    padding = 0.5
    return aabb_min - padding, aabb_max + padding

def parse_scene(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    # Arrays for Numba
    object_types = []
    inv_matrices = []
    scales = []
    material_ids = []
    operations = []
    aabb_mins = []
    aabb_maxs = []
    
    materials = []
    lights = []
    
    # Texture Management
    texture_paths = []
    texture_atlas_list = []
    
    def load_texture(path):
        if path in texture_paths:
            return float(texture_paths.index(path))
            
        if not HAS_PIL:
            print(f"Warning: PIL not found, cannot load texture {path}")
            return -1.0
            
        try:
            # Resolve path (similar to env map)
            yaml_dir = os.path.dirname(yaml_path)
            full_path = os.path.join(yaml_dir, path)
            if not os.path.exists(full_path):
                if os.path.exists(path):
                    full_path = path
                else:
                    print(f"Warning: Texture not found {path}")
                    return -1.0
            
            img = Image.open(full_path).convert('RGB')
            img = img.resize((512, 512))
            tex_data = np.array(img, dtype=np.float32) / 255.0
            
            texture_paths.append(path)
            texture_atlas_list.append(tex_data)
            return float(len(texture_paths) - 1)
            
        except Exception as e:
            print(f"Error loading texture {path}: {e}")
            return -1.0

    if 'Materials' not in data:
        # Default Grey with 0 SSS/Bump/ClearCoat/Aniso/Sheen/Displacement/Textures/UVScale
        materials.append([0.8, 0.8, 0.8, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, -1.0, -1.0, -1.0, 0.5])
    else:
        for m in data['Materials']:
            # [R, G, B, Metallic, Roughness, EmR, EmG, EmB, Transmission, IOR, SSS, Bump, ClearCoat, Aniso, AnisoRot, Sheen, SheenTint, Displacement, TexAlbedo, TexRough, TexNormal, UVScale]
            base = m['Color'] + [m.get('Metallic', 0.0), m.get('Roughness', 0.5)]
            emission = m.get('Emission', [0.0, 0.0, 0.0])
            transmission = [m.get('Transmission', 0.0)]
            ior = [m.get('IOR', 1.45)]
            sss = [m.get('SSS', 0.0)]
            bump = [m.get('Bump', 0.0)]
            clear_coat = [m.get('ClearCoat', 0.0)]
            aniso = [m.get('Anisotropic', 0.0)]
            aniso_rot = [m.get('AnisoRot', 0.0)]
            sheen = [m.get('Sheen', 0.0)]
            sheen_tint = [m.get('SheenTint', 0.5)]  # 0.5 = balanced white/albedo mix
            displacement = [m.get('Displacement', 0.0)]
            
            # Textures
            tex_albedo = load_texture(m.get('AlbedoMap', ''))
            tex_rough = load_texture(m.get('RoughnessMap', ''))
            tex_normal = load_texture(m.get('NormalMap', ''))
            
            uv_scale = m.get('UVScale', 0.5)
            
            materials.append(base + emission + transmission + ior + sss + bump + clear_coat + aniso + aniso_rot + sheen + sheen_tint + displacement + [tex_albedo, tex_rough, tex_normal, uv_scale])

    # Parse Objects
    for obj in data['Objects']:
        typ = obj['Type']
        pos = obj.get('Pos', [0,0,0])
        rot = obj.get('Rot', [0,0,0])
        scale = obj.get('Scale', [1,1,1])
        mat_id = obj.get('Mat', 0)
        op = obj.get('Op', 0) # 0=Union, 1=Sub, 2=Int
        
        T = make_translation(*pos)
        R = make_rotation(*rot)
        world_mat = T @ R
        inv_mat = np.linalg.inv(world_mat).astype(np.float32)
        
        # Calculate AABB
        b_min, b_max = calculate_aabb(typ, scale, world_mat)
        
        object_types.append(typ)
        inv_matrices.append(inv_mat)
        scales.append(scale)
        material_ids.append(mat_id)
        operations.append(op)
        aabb_mins.append(b_min)
        aabb_maxs.append(b_max)
        
    # Environment Map (HDRI / IBL)
    env_map_path = data.get('EnvironmentMap', None)
    env_map = np.zeros((1, 1, 3), dtype=np.float32) # Default empty
    
    if env_map_path and HAS_PIL:
        try:
            # Resolve path relative to YAML file
            yaml_dir = os.path.dirname(yaml_path)
            full_path = os.path.join(yaml_dir, env_map_path)
            
            # If not found, try project root
            if not os.path.exists(full_path):
                # Assuming project root is 2 levels up from utils/parser.py? 
                # No, let's just try relative to CWD as fallback
                if os.path.exists(env_map_path):
                    full_path = env_map_path
            
            if os.path.exists(full_path):
                print(f"Loading Environment Map: {full_path}")
                img = Image.open(full_path).convert('RGB')
                # Resize to reasonable dimension for performance
                if img.width > 1024:
                    img.thumbnail((1024, 512))
                env_map = np.array(img, dtype=np.float32) / 255.0
            else:
                print(f"Environment Map not found: {full_path}")
        except Exception as e:
            print(f"Failed to load Environment Map: {e}")

    # Stack Texture Atlas
    if len(texture_atlas_list) > 0:
        texture_atlas = np.stack(texture_atlas_list).astype(np.float32)
    else:
        # Dummy 1x1x1x3 atlas to avoid Numba errors if no textures
        texture_atlas = np.zeros((1, 512, 512, 3), dtype=np.float32)
    
    return {
        'num_objects': len(object_types),
        'object_types': np.array(object_types, dtype=np.int32),
        'inv_matrices': np.array(inv_matrices, dtype=np.float32),
        'scales': np.array(scales, dtype=np.float32),
        'material_ids': np.array(material_ids, dtype=np.float32),
        'operations': np.array(operations, dtype=np.float32),
        'aabb_mins': np.array(aabb_mins, dtype=np.float32),
        'aabb_maxs': np.array(aabb_maxs, dtype=np.float32),
        'materials': np.array(materials, dtype=np.float32),
        'lights': np.array(lights, dtype=np.float32),
        'env_map': env_map,
        'texture_atlas': texture_atlas,
        'texture_paths': texture_paths,
        'camera': data.get('Camera', {'Pos': [0, 0, -3], 'LookAt': [0, 0, 0], 'FOV': 45})
    }
