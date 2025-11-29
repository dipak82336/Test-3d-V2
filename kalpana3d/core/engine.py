import numpy as np
from numba import njit, prange
from .math import vec3, normalize, dot, cross, length, mix, clamp, reflect
from ..shapes.sdf import sdSphere, sdBox, sdRoundBox, sdCylinder, sdPlane, sdTorus, sdCapsule, sdCone, sdHexPrism
from ..shapes.sdf_extras import sdMandelbulb, sdSierpinski

# CONSTANTS
MAX_STEPS = 256
MAX_DIST = 100.0
SURF_DIST = 0.001
PI = 3.14159265359

# SHAPE TYPES
SHAPE_SPHERE = 0
SHAPE_BOX = 1
SHAPE_CYLINDER = 2
SHAPE_ROUNDBOX = 3
SHAPE_PLANE = 4
SHAPE_TORUS = 5
SHAPE_CAPSULE = 6
SHAPE_CONE = 7
SHAPE_HEXPRISM = 8
SHAPE_MANDELBULB = 9
SHAPE_SIERPINSKI = 10

# OPERATIONS
OP_UNION = 0
OP_SUBTRACT = 1
OP_INTERSECT = 2
OP_SMOOTH_UNION = 3
OP_SMOOTH_SUBTRACT = 4
OP_SMOOTH_INTERSECT = 5

@njit
def smin(a, b, k):
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - h * h * k * (1.0 / 4.0)

@njit
def transform_point(inv_mat, p):
    x = inv_mat[0,0]*p[0] + inv_mat[0,1]*p[1] + inv_mat[0,2]*p[2] + inv_mat[0,3]
    y = inv_mat[1,0]*p[0] + inv_mat[1,1]*p[1] + inv_mat[1,2]*p[2] + inv_mat[1,3]
    z = inv_mat[2,0]*p[0] + inv_mat[2,1]*p[1] + inv_mat[2,2]*p[2] + inv_mat[2,3]
    return vec3(x, y, z)

@njit
def map_scene(p, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, aabb_mins, aabb_maxs):
    d_final = MAX_DIST
    m_final = -1.0

    for i in range(num_objects):
        op = int(operations[i])

        # AABB Culling (Optimization)
        # Only for UNION (0) as it's the most common and safe to cull
        if op == 0:
            b_min = aabb_mins[i]
            b_max = aabb_maxs[i]

            # Distance to AABB
            # max(max(min - p, p - max), 0)
            dx = max(max(b_min[0] - p[0], p[0] - b_max[0]), 0.0)
            dy = max(max(b_min[1] - p[1], p[1] - b_max[1]), 0.0)
            dz = max(max(b_min[2] - p[2], p[2] - b_max[2]), 0.0)
            d_aabb = np.sqrt(dx*dx + dy*dy + dz*dz)

            if d_aabb > d_final:
                continue

        local_p = transform_point(inv_matrices[i], p)

        # Domain Repetition (Infinite Grid)
        if op == 6: # OP_REPEAT
            # Repeat every 4.0 units
            s = 4.0
            local_p[0] = (local_p[0] + s*0.5) % s - s*0.5
            local_p[1] = (local_p[1] + s*0.5) % s - s*0.5
            local_p[2] = (local_p[2] + s*0.5) % s - s*0.5

        d = MAX_DIST
        typ = object_types[i]
        scale = scales[i]

        if typ == SHAPE_SPHERE: d = sdSphere(local_p, scale[0])
        elif typ == SHAPE_BOX: d = sdBox(local_p, scale)
        elif typ == SHAPE_CYLINDER: d = sdCylinder(local_p, scale[1], scale[0])
        elif typ == SHAPE_ROUNDBOX: d = sdRoundBox(local_p, scale, 0.1)
        elif typ == SHAPE_PLANE: d = local_p[1]
        elif typ == SHAPE_TORUS: d = sdTorus(local_p, vec3(scale[0], scale[1], 0))
        elif typ == SHAPE_CAPSULE: d = sdCapsule(local_p, vec3(0, -scale[1]/2, 0), vec3(0, scale[1]/2, 0), scale[0])
        elif typ == SHAPE_CONE: d = sdCone(local_p, vec3(scale[0], scale[1], 0), scale[2])
        elif typ == SHAPE_HEXPRISM: d = sdHexPrism(local_p, vec3(scale[0], scale[1], 0))

        mat = material_ids[i]

        # APPLY DISPLACEMENT (Geometric Deformation)
        mat_idx = int(mat)
        if mat_idx >= 0 and mat_idx < materials.shape[0]:
            disp_strength = materials[mat_idx, 17]  # Displacement parameter

            if disp_strength > 0.0:
                # Optimization: Only calculate noise if close to surface
                # Conservative bound: d - max_disp
                # If d > 1.0, we are far enough to skip detailed noise
                if d < 1.0:
                    # 3D Value Noise for displacement
                    noise_scale = 3.0  # Controls detail frequency
                    noise_val = value_noise(scale_vec3(local_p, noise_scale))

                    # Offset noise to -1..1 range, then scale
                    displacement = (noise_val - 0.5) * 2.0 * disp_strength
                    d += displacement
                else:
                    d -= disp_strength # Conservative step

        if i == 0:
            d_final = d
            m_final = mat
        else:
            if op == OP_UNION or op == 6: # Treat Repeat as Union
                if d < d_final:
                    d_final = d
                    m_final = mat
            elif op == OP_SUBTRACT:
                if -d > d_final:
                    d_final = -d
            elif op == OP_INTERSECT:
                if d > d_final:
                    d_final = d
                    m_final = mat
            elif op == OP_SMOOTH_UNION:
                k = 0.1 # Smoothness factor
                d_final = smin(d_final, d, k)
                if d < d_final + 0.1: # Approximate material blending
                     m_final = mat
            elif op == OP_SMOOTH_SUBTRACT:
                k = 0.1
                h = max(k - abs(-d - d_final), 0.0) / k
                d_final = max(-d, d_final) + h * h * k * (1.0 / 4.0)
            elif op == OP_SMOOTH_INTERSECT:
                k = 0.1
                h = max(k - abs(d - d_final), 0.0) / k
                d_final = max(d, d_final) - h * h * k * (1.0 / 4.0)
                if d > d_final - 0.1:
                    m_final = mat

    return d_final, m_final

@njit
def calc_normal(p, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, aabb_mins, aabb_maxs):
    e = vec3(SURF_DIST, 0.0, 0.0)
    d = map_scene(p, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, aabb_mins, aabb_maxs)[0]
    n = vec3(
        d - map_scene(p - vec3(e[0], 0, 0), num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, aabb_mins, aabb_maxs)[0],
        d - map_scene(p - vec3(0, e[0], 0), num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, aabb_mins, aabb_maxs)[0],
        d - map_scene(p - vec3(0, 0, e[0]), num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, aabb_mins, aabb_maxs)[0]
    )
    return normalize(n)

@njit
def calc_softshadow(ro, rd, tmin, tmax, k, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, aabb_mins, aabb_maxs):
    res = 1.0
    t = tmin
    for i in range(16):
        h = map_scene(ro + rd * t, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, aabb_mins, aabb_maxs)[0]
        res = min(res, k * h / t)
        t += clamp(h, 0.02, 0.2)
        if res < 0.005 or t > tmax: break
    return clamp(res, 0.0, 1.0)

@njit
def calc_ao(p, n, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, aabb_mins, aabb_maxs):
    occ = 0.0
    sca = 1.0
    for i in range(5):
        h = 0.01 + 0.12 * float(i) / 4.0
        d = map_scene(p + n * h, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, aabb_mins, aabb_maxs)[0]
        occ += (h - d) * sca
        sca *= 0.95
        if sca < 0.1: break
    return clamp(1.0 - 3.0 * occ, 0.0, 1.0)

@njit
def floor_vec3(v):
    return vec3(np.floor(v[0]), np.floor(v[1]), np.floor(v[2]))

@njit
def fract_vec3(v):
    return v - floor_vec3(v)

@njit
def hash33(p):
    p3 = fract_vec3(mul_vec3(p, np.array([.1031, .1030, .0973], dtype=np.float32)))
    p3_yzx = np.array([p3[1], p3[2], p3[0]], dtype=np.float32)
    dot_val = dot(p3, vec3(p3_yzx[0] + 33.33, p3_yzx[1] + 33.33, p3_yzx[2] + 33.33))
    p3 = vec3(p3[0] + dot_val, p3[1] + dot_val, p3[2] + dot_val)
    p3_xxy = np.array([p3[0], p3[0], p3[1]], dtype=np.float32)
    p3_yzz = np.array([p3[1], p3[2], p3[2]], dtype=np.float32)
    p3_zyx = np.array([p3[2], p3[1], p3[0]], dtype=np.float32)
    return fract_vec3(vec3((p3_xxy[0] + p3_yzz[0]) * p3_zyx[0], (p3_xxy[1] + p3_yzz[1]) * p3_zyx[1], (p3_xxy[2] + p3_yzz[2]) * p3_zyx[2]))

@njit
def mul_vec3(a, b):
    return vec3(a[0]*b[0], a[1]*b[1], a[2]*b[2])

@njit
def add_vec3(a, b):
    return vec3(a[0]+b[0], a[1]+b[1], a[2]+b[2])

@njit
def scale_vec3(v, s):
    return vec3(v[0]*s, v[1]*s, v[2]*s)

@njit
def lerp(a, b, t):
    return vec3(a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t, a[2] + (b[2]-a[2])*t)

@njit
def value_noise(p):
    i = floor_vec3(p)
    f = fract_vec3(p)
    u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)
    a = hash33(i)[0]
    b = hash33(add_vec3(i, np.array([1.0, 0.0, 0.0], dtype=np.float32)))[0]
    c = hash33(add_vec3(i, np.array([0.0, 1.0, 0.0], dtype=np.float32)))[0]
    d = hash33(add_vec3(i, np.array([1.0, 1.0, 0.0], dtype=np.float32)))[0]
    e = hash33(add_vec3(i, np.array([0.0, 0.0, 1.0], dtype=np.float32)))[0]
    f_val = hash33(add_vec3(i, np.array([1.0, 0.0, 1.0], dtype=np.float32)))[0]
    g = hash33(add_vec3(i, np.array([0.0, 1.0, 1.0], dtype=np.float32)))[0]
    h = hash33(add_vec3(i, np.array([1.0, 1.0, 1.0], dtype=np.float32)))[0]
    k0 = a
    k1 = b - a
    k2 = c - a
    k3 = e - a
    k4 = a - b - c + d
    k5 = a - c - e + g
    k6 = a - b - e + f_val
    k7 = -a + b + c - d + e - f_val - g + h
    return k0 + k1*u[0] + k2*u[1] + k3*u[2] + k4*u[0]*u[1] + k5*u[1]*u[2] + k6*u[2]*u[0] + k7*u[0]*u[1]*u[2]

@njit
def fbm(p):
    v = 0.0
    a = 0.5
    shift = np.array([100.0, 100.0, 100.0], dtype=np.float32)
    for i in range(3):
        v += a * value_noise(p)
        p = add_vec3(scale_vec3(p, 2.0), shift)
        a *= 0.5
    return v

@njit
def get_sky(rd, roughness):
    # Simple sky for fallback
    t = rd[1] * 0.5 + 0.5
    return mix(np.array([0.5, 0.7, 1.0], dtype=np.float32), np.array([1.0, 1.0, 1.0], dtype=np.float32), t)

@njit
def sample_env_map(rd, env_map_flat, width, height):
    if width <= 1:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    phi = np.arctan2(rd[2], rd[0])
    theta = np.arcsin(rd[1])

    u = 0.5 + phi / (2.0 * np.pi)
    v = 0.5 - theta / np.pi

    x = int(u * width) % width
    y = int(v * height) % height

    idx = (y * width + x) * 3
    if idx < 0 or idx >= env_map_flat.size - 2:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    return np.array([env_map_flat[idx], env_map_flat[idx+1], env_map_flat[idx+2]], dtype=np.float32)

@njit
def calc_bump_normal(p, n, strength):
    e = 0.001
    scale = 20.0
    n_x = value_noise(add_vec3(p, np.array([e, 0.0, 0.0], dtype=np.float32)) * scale) - value_noise(add_vec3(p, np.array([-e, 0.0, 0.0], dtype=np.float32)) * scale)
    n_y = value_noise(add_vec3(p, np.array([0.0, e, 0.0], dtype=np.float32)) * scale) - value_noise(add_vec3(p, np.array([0.0, -e, 0.0], dtype=np.float32)) * scale)
    n_z = value_noise(add_vec3(p, np.array([0.0, 0.0, e], dtype=np.float32)) * scale) - value_noise(add_vec3(p, np.array([0.0, 0.0, -e], dtype=np.float32)) * scale)
    grad = np.array([n_x, n_y, n_z], dtype=np.float32)
    return normalize(add_vec3(n, scale_vec3(grad, -strength * 5.0)))

@njit
def calc_clearcoat(n, rd, sun_dir, sun_col, sun_int, shadow, clear_coat):
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)

@njit
def sample_atlas_2d(uv, idx, atlas):
    u = uv[0] - np.floor(uv[0])
    v = uv[1] - np.floor(uv[1])

    size = 512 # Fixed for now
    x = int(u * size) % size
    y = int(v * size) % size
    layer = int(idx)

    # Check bounds
    if layer < 0 or layer >= atlas.shape[0]:
        return np.array([1.0, 0.0, 1.0], dtype=np.float32)

    return atlas[layer, y, x]

@njit
def triplanar_sample(p, n, tex_idx, atlas, scale):
    w = np.abs(n)
    w = w / (w[0] + w[1] + w[2])

    uv_x = vec3(p[2] * scale, p[1] * scale, 0)
    uv_y = vec3(p[0] * scale, p[2] * scale, 0)
    uv_z = vec3(p[0] * scale, p[1] * scale, 0)

    col_x = sample_atlas_2d(uv_x, tex_idx, atlas)
    col_y = sample_atlas_2d(uv_y, tex_idx, atlas)
    col_z = sample_atlas_2d(uv_z, tex_idx, atlas)

    return add_vec3(add_vec3(scale_vec3(col_x, w[0]), scale_vec3(col_y, w[1])), scale_vec3(col_z, w[2]))

@njit
def triplanar_normal(p, n, tex_idx, atlas, scale):
    # Simplified normal mapping (perturbation)
    # Correct tangent space triplanar is complex, we'll use a perturbation approximation
    # Sample intensity/height from normal map (assuming blue channel dominant or converting to greyscale)
    # Actually, if we have a normal map, we should use it.

    w = np.abs(n)
    w = w / (w[0] + w[1] + w[2])

    uv_x = vec3(p[2] * scale, p[1] * scale, 0)
    uv_y = vec3(p[0] * scale, p[2] * scale, 0)
    uv_z = vec3(p[0] * scale, p[1] * scale, 0)

    # Sample normals from atlas (in 0..1 range)
    c_x = sample_atlas_2d(uv_x, tex_idx, atlas)
    c_y = sample_atlas_2d(uv_y, tex_idx, atlas)
    c_z = sample_atlas_2d(uv_z, tex_idx, atlas)

    # Unpack to -1..1
    n_x = add_vec3(scale_vec3(c_x, 2.0), np.array([-1.0, -1.0, -1.0], dtype=np.float32))
    n_y = add_vec3(scale_vec3(c_y, 2.0), np.array([-1.0, -1.0, -1.0], dtype=np.float32))
    n_z = add_vec3(scale_vec3(c_z, 2.0), np.array([-1.0, -1.0, -1.0], dtype=np.float32))

    # Blend (this is mathematically incorrect but visually acceptable for noise/rough surfaces)
    # A better way is Whiteout blending or UDN blending
    blended = add_vec3(add_vec3(scale_vec3(n_x, w[0]), scale_vec3(n_y, w[1])), scale_vec3(n_z, w[2]))
    blended = normalize(blended)

    # Reorient to surface normal 'n'
    # This is a hacky way to apply the detail normal to the surface normal
    # We treat 'blended' as a perturbation in tangent space, but we don't have tangent.
    # So we just add it to n and normalize (very hacky)
    return normalize(add_vec3(n, scale_vec3(blended, 0.5)))

@njit
def get_lighting(p, n, rd, mat_id, materials, num_objects, object_types, inv_matrices, scales, material_ids, operations, env_map_flat, env_map_w, env_map_h, texture_atlas, aabb_mins, aabb_maxs):
    mat_idx = int(mat_id)

    # Manual Array Access
    r = materials[mat_idx, 0]
    g = materials[mat_idx, 1]
    b = materials[mat_idx, 2]
    albedo = np.array([r, g, b], dtype=np.float32)

    metallic = materials[mat_idx, 3]
    roughness = materials[mat_idx, 4]

    em_r = materials[mat_idx, 5]
    em_g = materials[mat_idx, 6]
    em_b = materials[mat_idx, 7]
    emission = np.array([em_r, em_g, em_b], dtype=np.float32)

    transmission = materials[mat_idx, 8]
    ior = materials[mat_idx, 9]

    sss_strength = materials[mat_idx, 10]
    bump_strength = materials[mat_idx, 11]
    clear_coat = materials[mat_idx, 12]
    aniso = materials[mat_idx, 13]
    aniso_rot = materials[mat_idx, 14]
    sheen = materials[mat_idx, 15]
    sheen_tint = materials[mat_idx, 16]

    tex_albedo_id = materials[mat_idx, 18]
    tex_rough_id = materials[mat_idx, 19]
    tex_normal_id = materials[mat_idx, 20]
    uv_scale = materials[mat_idx, 21]

    # Sample Textures
    if tex_albedo_id >= 0.0:
        tex_col = triplanar_sample(p, n, tex_albedo_id, texture_atlas, uv_scale)
        albedo = mul_vec3(albedo, tex_col)

    if tex_rough_id >= 0.0:
        tex_r = triplanar_sample(p, n, tex_rough_id, texture_atlas, uv_scale)
        roughness = roughness * tex_r[0] # Use R channel

    # APPLY NORMAL MAPPING
    if tex_normal_id >= 0.0:
        n = triplanar_normal(p, n, tex_normal_id, texture_atlas, uv_scale)

    # APPLY BUMP MAPPING (Procedural)
    if bump_strength > 0.0:
        n = calc_bump_normal(p, n, bump_strength)

    # CLEAR COAT (Secondary Specular Layer)
    cc_specular = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cc_fresnel = 0.0

    col = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # EMISSION - "HOT CORE" NEON EFFECT
    if emission[0] > 0.0 or emission[1] > 0.0 or emission[2] > 0.0:
        view_dir = scale_vec3(rd, -1.0)
        ndotv = max(dot(n, view_dir), 0.0)
        em_intensity = max(emission[0], max(emission[1], emission[2]))

        if em_intensity > 4.0:
            core_mask = float(pow(ndotv, 8.0))
            white_hot = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            vibrant_col = normalize(emission)
            final_col = add_vec3(scale_vec3(vibrant_col, 1.0 - core_mask), scale_vec3(white_hot, core_mask))
            col = add_vec3(col, scale_vec3(final_col, 12.0))
            return col
        else:
            fog_factor = 0.2 + 0.8 * ndotv
            col = add_vec3(col, scale_vec3(emission, 5.0 * fog_factor))
            if transmission > 0.0:
                 pass

    # Ambient Occlusion
    ao = calc_ao(p, n, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, aabb_mins, aabb_maxs)

    # Ambient Lighting
    ambient_strength = 0.18
    ambient_light = np.array([ambient_strength, ambient_strength, ambient_strength * 1.1], dtype=np.float32)

    metal_ambient = mul_vec3(scale_vec3(albedo, metallic), ambient_light)
    dielectric_ambient = mul_vec3(scale_vec3(albedo, 1.0 - metallic), ambient_light)
    col = add_vec3(col, scale_vec3(add_vec3(metal_ambient, dielectric_ambient), ao))

    # Direct Light (Sun)
    sun_dir = normalize(np.array([0.5, 0.8, 0.5], dtype=np.float32))
    sun_col = np.array([1.0, 0.95, 0.9], dtype=np.float32)
    sun_int = 2.5

    # Shadows
    shadow = calc_softshadow(p, sun_dir, 0.02, 3.0, 8.0, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, aabb_mins, aabb_maxs)

    # Calculate F0
    f0 = mix(np.array([0.04, 0.04, 0.04], dtype=np.float32), albedo, metallic)

    # Specular Highlights
    h = normalize(add_vec3(sun_dir, scale_vec3(rd, -1.0)))
    h_dot_sun = clamp(dot(h, sun_dir), 0.0, 1.0)
    ndoth = max(dot(n, h), 0.0)
    ndotl = max(dot(n, sun_dir), 0.0)

    spec = 0.0

    if aniso > 0.0:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(n[1]) > 0.9:
            up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        t = normalize(cross(n, up))
        b = normalize(cross(n, t))

        ax = max(roughness * (1.0 + aniso), 0.001)
        ay = max(roughness * (1.0 - aniso), 0.001)

        dot_th = dot(t, h)
        dot_bh = dot(b, h)
        dot_nh = ndoth

        denom = dot_th*dot_th/(ax*ax) + dot_bh*dot_bh/(ay*ay) + dot_nh*dot_nh
        d_aniso = 1.0 / (3.14159 * ax * ay * denom * denom)
        spec = d_aniso * 0.5
    else:
        spec_pow = 2048.0 * (1.0 - roughness) + 4.0
        spec = float(pow(float(ndoth), float(spec_pow))) * (1.0 - roughness) * 1.5

    fresnel_term = float(pow(1.0 - h_dot_sun, 5.0))
    f_schlick = add_vec3(f0, scale_vec3(add_vec3(np.array([1.0, 1.0, 1.0], dtype=np.float32), scale_vec3(f0, -1.0)), fresnel_term))

    diffuse = scale_vec3(mul_vec3(albedo, scale_vec3(np.array([1.0, 1.0, 1.0], dtype=np.float32), 1.0 - metallic)), 1.0 - transmission)
    diffuse = scale_vec3(diffuse, ndotl)

    specular = scale_vec3(f_schlick, spec * ndotl)

    col = add_vec3(col, scale_vec3(mul_vec3(add_vec3(diffuse, specular), sun_col), sun_int * shadow))

    # CLEAR COAT
    if clear_coat > 0.0:
        cc_roughness = 0.05
        cc_f0 = 0.04
        cc_fresnel_term = float(pow(1.0 - h_dot_sun, 5.0))
        cc_fresnel = cc_f0 + (1.0 - cc_f0) * cc_fresnel_term
        cc_spec_power = 2048.0
        cc_spec_intensity = float(pow(float(ndoth), float(cc_spec_power))) * cc_fresnel
        cc_specular = np.array([cc_spec_intensity, cc_spec_intensity, cc_spec_intensity], dtype=np.float32)
        cc_specular = scale_vec3(cc_specular, ndotl * shadow * sun_int * clear_coat)
        attenuation = 1.0 - (cc_fresnel * clear_coat * 0.5)
        col = scale_vec3(col, attenuation)
        col = add_vec3(col, cc_specular)

    # SHEEN
    if sheen > 0.0:
        view_dir = scale_vec3(rd, -1.0)
        ndotv = max(dot(n, view_dir), 0.0)
        alpha = 0.5
        inv_alpha = 1.0 / alpha
        sin_theta = float(pow(1.0 - ndotv * ndotv, 0.5))
        sheen_dist = (2.0 + inv_alpha) * float(pow(sin_theta, inv_alpha)) / (2.0 * 3.14159)
        white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        sheen_color = add_vec3(scale_vec3(white, 1.0 - sheen_tint), scale_vec3(albedo, sheen_tint))
        sheen_contrib = scale_vec3(sheen_color, sheen_dist * sheen * ndotl * 0.5)
        col = add_vec3(col, sheen_contrib)

    # SSS
    if sss_strength > 0.0:
        distortion = 0.5
        trans_light = add_vec3(sun_dir, scale_vec3(n, distortion))
        trans_dot = float(pow(max(dot(scale_vec3(rd, -1.0), trans_light), 0.0), 4.0))
        ambient_sss = (1.0 - ao) * 0.5
        sss_color = mul_vec3(albedo, np.array([1.2, 1.0, 0.8], dtype=np.float32))
        sss_final = scale_vec3(sss_color, (trans_dot + ambient_sss) * sss_strength * 2.0)
        col = add_vec3(col, sss_final)

    # REFRACTION
    if transmission > 0.0:
        edge_factor = 1.0 - abs(dot(n, rd))
        edge_opacity = pow(edge_factor, 2.0) * 0.3

        ior_r = ior * 0.98
        ior_g = ior
        ior_b = ior * 1.02

        refract_col = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Red
        eta_r = 1.0 / ior_r
        cos_theta = dot(scale_vec3(rd, -1.0), n)
        k_r = 1.0 - eta_r * eta_r * (1.0 - cos_theta * cos_theta)
        if k_r >= 0.0:
            # refract_dir_r = normalize(add_vec3(scale_vec3(rd, eta_r), scale_vec3(n, eta_r * cos_theta - np.sqrt(k_r))))
            refract_dir_r = normalize(add_vec3(scale_vec3(rd, eta_r), scale_vec3(n, eta_r * cos_theta - np.sqrt(k_r))))
            refract_dir_r_f32 = np.array([refract_dir_r[0], refract_dir_r[1], refract_dir_r[2]], dtype=np.float32)
            bg_r = sample_env_map(refract_dir_r_f32, env_map_flat, env_map_w, env_map_h)
            # bg_r = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            refract_col = add_vec3(refract_col, np.array([bg_r[0], 0.0, 0.0], dtype=np.float32))

        # Green
        eta_g = 1.0 / ior_g
        k_g = 1.0 - eta_g * eta_g * (1.0 - cos_theta * cos_theta)
        if k_g >= 0.0:
            # refract_dir_g = normalize(add_vec3(scale_vec3(rd, eta_g), scale_vec3(n, eta_g * cos_theta - np.sqrt(k_g))))
            refract_dir_g = normalize(add_vec3(scale_vec3(rd, eta_g), scale_vec3(n, eta_g * cos_theta - np.sqrt(k_g))))
            refract_dir_g_f32 = np.array([refract_dir_g[0], refract_dir_g[1], refract_dir_g[2]], dtype=np.float32)
            bg_g = sample_env_map(refract_dir_g_f32, env_map_flat, env_map_w, env_map_h)
            # bg_g = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            refract_col = add_vec3(refract_col, np.array([0.0, bg_g[1], 0.0], dtype=np.float32))

        # Blue
        eta_b = 1.0 / ior_b
        k_b = 1.0 - eta_b * eta_b * (1.0 - cos_theta * cos_theta)
        if k_b >= 0.0:
            # refract_dir_b = normalize(add_vec3(scale_vec3(rd, eta_b), scale_vec3(n, eta_b * cos_theta - np.sqrt(k_b))))
            refract_dir_b = normalize(add_vec3(scale_vec3(rd, eta_b), scale_vec3(n, eta_b * cos_theta - np.sqrt(k_b))))
            refract_dir_b_f32 = np.array([refract_dir_b[0], refract_dir_b[1], refract_dir_b[2]], dtype=np.float32)
            bg_b = sample_env_map(refract_dir_b_f32, env_map_flat, env_map_w, env_map_h)
            # bg_b = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            refract_col = add_vec3(refract_col, np.array([0.0, 0.0, bg_b[2]], dtype=np.float32))

        tint = lerp(np.array([1.0, 1.0, 1.0], dtype=np.float32), albedo, 0.2)
        refract_col = mul_vec3(refract_col, tint)

        ref_dir = reflect(rd, n)
        ref_dir_f32 = np.array([ref_dir[0], ref_dir[1], ref_dir[2]], dtype=np.float32)
        sky_ref = sample_env_map(ref_dir_f32, env_map_flat, env_map_w, env_map_h)
        # sky_ref = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        rd_dot_n = clamp(1.0 + dot(rd, n), 0.0, 1.0)
        f_val = 0.3 + 0.7 * float(pow(rd_dot_n, 5.0))

        effective_transmission = transmission * (1.0 - edge_opacity)
        col = add_vec3(scale_vec3(col, 1.0 - effective_transmission), scale_vec3(refract_col, effective_transmission))
        col = add_vec3(col, scale_vec3(sky_ref, f_val * transmission * 3.0))

        if edge_opacity > 0.01:
            edge_color = np.array([0.9, 0.95, 1.0], dtype=np.float32)
            col = add_vec3(col, scale_vec3(edge_color, edge_opacity * 2.0))

    col = scale_vec3(col, 1.0 - cc_fresnel)
    col = add_vec3(col, cc_specular)

    return col

@njit
def raymarch_kernel(ro, rd, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, env_map_flat, env_map_w, env_map_h, texture_atlas, aabb_mins, aabb_maxs):
    dO = 0.0
    glow_acc = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    for i in range(MAX_STEPS):
        p = ro + rd * dO
        d, m = map_scene(p, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, aabb_mins, aabb_maxs)

        # VOLUMETRIC GLOW ACCUMULATION
        # If we are close to an object, accumulate its emission as "Glow"
        if d < 0.5: # Glow radius
            # Find which object is closest to get its emission
            # This is expensive, so maybe we can approximate?
            # For now, let's just use the material ID returned by map_scene?
            # BUT map_scene returns the material of the CLOSEST object.
            # If we are in the "glow zone" but not hitting it yet, 'm' is correct.

            mat_idx = int(m)
            # Check emission
            em_r = materials[mat_idx, 5]
            em_g = materials[mat_idx, 6]
            em_b = materials[mat_idx, 7]

            emission = np.array([em_r, em_g, em_b], dtype=np.float32)

            # Accumulate glow based on distance (inverse square falloff)
            # glow_strength = 1.0 / (d * d + 0.1) * 0.01
            glow_strength = max(0.0, (0.5 - d) * 0.02)

            glow_acc = add_vec3(glow_acc, scale_vec3(emission, glow_strength))

        if d < SURF_DIST:
            n = calc_normal(p, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, aabb_mins, aabb_maxs)
            col = get_lighting(p, n, rd, m, materials, num_objects, object_types, inv_matrices, scales, material_ids, operations, env_map_flat, env_map_w, env_map_h, texture_atlas, aabb_mins, aabb_maxs)
            # Add accumulated glow to the surface color too
            return add_vec3(col, glow_acc)

        dO += d
        if dO > MAX_DIST:
            break

    rd_arr = np.array([rd[0], rd[1], rd[2]], dtype=np.float32)
    bg_col = sample_env_map(rd_arr, env_map_flat, env_map_w, env_map_h)

    # Add accumulated glow to background
    return add_vec3(bg_col, glow_acc)

@njit(parallel=True)
def render_pixels(width, height, cam_pos, cam_target, fov,
                 num_objects, object_types, inv_matrices, scales, material_ids, operations,
                 materials, lights, env_map_flat, env_map_w, env_map_h, texture_atlas, aabb_mins, aabb_maxs, output_buffer):

    ro = cam_pos
    lookat = cam_target
    f = normalize(lookat - ro)
    r = normalize(cross(f, vec3(0, 1, 0)))
    u = cross(r, f)
    zoom = 1.0 / np.tan((fov * np.pi / 180.0) / 2.0)

    for y in prange(height):
        for x in range(width):
            # Accumulators for components
            r_acc = 0.0
            g_acc = 0.0
            b_acc = 0.0

            # 4x Super-Sampling (SSAA)
            # Sample 1
            uv_x = ((x + 0.25) - width / 2.0) / height
            uv_y = -((y + 0.25) - height / 2.0) / height
            dir_vec = f * zoom + r * uv_x + u * uv_y
            rd = normalize(dir_vec.astype(np.float32))
            col = raymarch_kernel(ro, rd, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, env_map_flat, env_map_w, env_map_h, texture_atlas, aabb_mins, aabb_maxs)
            r_acc += col[0]
            g_acc += col[1]
            b_acc += col[2]

            # Sample 2
            uv_x = ((x + 0.75) - width / 2.0) / height
            uv_y = -((y + 0.25) - height / 2.0) / height
            dir_vec = f * zoom + r * uv_x + u * uv_y
            rd = normalize(dir_vec.astype(np.float32))
            col = raymarch_kernel(ro, rd, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, env_map_flat, env_map_w, env_map_h, texture_atlas, aabb_mins, aabb_maxs)
            r_acc += col[0]
            g_acc += col[1]
            b_acc += col[2]

            # Sample 3
            uv_x = ((x + 0.25) - width / 2.0) / height
            uv_y = -((y + 0.75) - height / 2.0) / height
            dir_vec = f * zoom + r * uv_x + u * uv_y
            rd = normalize(dir_vec.astype(np.float32))
            col = raymarch_kernel(ro, rd, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, env_map_flat, env_map_w, env_map_h, texture_atlas, aabb_mins, aabb_maxs)
            r_acc += col[0]
            g_acc += col[1]
            b_acc += col[2]

            # Sample 4
            uv_x = ((x + 0.75) - width / 2.0) / height
            uv_y = -((y + 0.75) - height / 2.0) / height
            dir_vec = f * zoom + r * uv_x + u * uv_y
            rd = normalize(dir_vec.astype(np.float32))
            col = raymarch_kernel(ro, rd, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, env_map_flat, env_map_w, env_map_h, texture_atlas, aabb_mins, aabb_maxs)
            r_acc += col[0]
            g_acc += col[1]
            b_acc += col[2]

            # Average
            r_val = r_acc * 0.25
            g_val = g_acc * 0.25
            b_val = b_acc * 0.25

            # Tone Mapping (Reinhard)
            r_val = r_val / (r_val + 1.0)
            g_val = g_val / (g_val + 1.0)
            b_val = b_val / (b_val + 1.0)

            # Gamma Correction
            r_val = pow(r_val, 1.0/2.2)
            g_val = pow(g_val, 1.0/2.2)
            b_val = pow(b_val, 1.0/2.2)

            idx = (y * width + x) * 3
            output_buffer[idx] = r_val
            output_buffer[idx+1] = g_val
            output_buffer[idx+2] = b_val

def render_scene(width, height, cam_pos, cam_target, fov,
                 num_objects, object_types, inv_matrices, scales, material_ids, operations,
                 materials, lights, env_map_flat, env_map_w, env_map_h, texture_atlas, aabb_mins, aabb_maxs, output_buffer):
    # Just call the parallel kernel
    render_pixels(width, height, cam_pos, cam_target, fov,
                 num_objects, object_types, inv_matrices, scales, material_ids, operations,
                 materials, lights, env_map_flat, env_map_w, env_map_h, texture_atlas, aabb_mins, aabb_maxs, output_buffer)
