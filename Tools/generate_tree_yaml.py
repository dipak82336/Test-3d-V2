import yaml
import numpy as np
import math
import random

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: return v
    return v / norm

def direction_to_euler(d):
    # Align Y axis (0,1,0) to vector d
    d = normalize(d)

    # Standard conversion
    # Pitch (X rotation) and Yaw (Z rotation) - Wait, standard Euler is usually ZYX or XYZ
    # Kalpana3D uses Rz @ Ry @ Rx

    # Let's try to find Rx, Ry, Rz such that R * (0,1,0) = d
    # (0,1,0) rotated by Rx -> (0, cos, sin)
    # Then Ry -> (sin*sin_y, cos, sin*cos_y)
    # This is getting complicated.

    # Alternative: LookAt matrix logic
    up = np.array([0, 1, 0])

    # If d is parallel to up
    if abs(np.dot(d, up)) > 0.999:
        if d[1] > 0: return [0, 0, 0]
        else: return [math.pi, 0, 0]

    # Axis to rotate around
    axis = np.cross(up, d)
    axis = normalize(axis)

    # Angle
    angle = math.acos(np.dot(up, d))

    # Convert Axis-Angle to Euler?
    # Or Matrix to Euler.
    # Construct rotation matrix from axis-angle
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c
    x, y, z = axis

    R = np.array([
        [t*x*x + c,   t*x*y - z*s, t*x*z + y*s],
        [t*x*y + z*s, t*y*y + c,   t*y*z - x*s],
        [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
    ])

    # Extract Euler angles (Z-Y-X convention matching make_rotation in parser.py)
    # R = Rz @ Ry @ Rx
    # R[0,2] = -sin(ry)
    # ry = -asin(R[0,2])

    sy = -R[0,2]
    cy = math.sqrt(1 - sy*sy)

    ry = math.atan2(-R[0,2], math.sqrt(R[0,0]**2 + R[0,1]**2)) # This is safer

    if abs(math.cos(ry)) > 1e-6:
        rx = math.atan2(R[1,2], R[2,2])
        rz = math.atan2(R[0,1], R[0,0])
    else:
        # Gimbal lock
        rx = math.atan2(-R[2,1], R[1,1])
        rz = 0

    return [float(rx), float(ry), float(rz)]

def generate_tree():
    objects = []

    # Tree Parameters
    MAX_DEPTH = 5
    MIN_RADIUS = 0.02

    segments = []

    # Queue for BFS: (pos, dir, length, radius, depth)
    # Start with trunk
    queue = [ (np.array([0, 0, 0]), np.array([0, 1, 0]), 2.0, 0.4, 0) ]

    # Roots
    root_count = 5
    for i in range(root_count):
        angle = (i / root_count) * 2 * math.pi
        d = np.array([math.cos(angle), -0.5, math.sin(angle)])
        queue.append( (np.array([0, 0.2, 0]), normalize(d), 1.5, 0.3, 0) )

    while queue:
        pos, direction, length, radius, depth = queue.pop(0)

        if depth > MAX_DEPTH or radius < MIN_RADIUS:
            continue

        end_pos = pos + direction * length
        center = (pos + end_pos) / 2.0

        # Add capsule
        rot = direction_to_euler(direction)

        # Variation in radius along length? No, Capsule is constant radius.
        # But we can use Cone? Cone is tapered.
        # Let's use Cone for better look? Or Capsule is fine.
        # User said "High Quality Detail". Tapered branches are better.
        # But SDF Cone is infinite or specific. `sdRoundCone` is better for trees.
        # Engine has `sdCone` and `sdCapsule`. `sdCone` in sdf.py takes angle.
        # Let's stick to Capsule for smooth blending.

        objects.append({
            'Type': 6, # Capsule
            'Pos': [float(center[0]), float(center[1]), float(center[2])],
            'Rot': rot,
            'Scale': [float(radius), float(length), 0.0],
            'Mat': 0,
            'Op': 3 # Smooth Union
        })

        # Branching
        num_branches = 0
        if depth < 2:
            num_branches = random.randint(2, 3)
        else:
            num_branches = random.randint(1, 3)

        # Continue main branch?
        # A tree usually splits.

        for i in range(num_branches):
            # Rotate direction
            # Random vector perturbation
            perturb = np.random.uniform(-0.5, 0.5, 3)
            new_dir = normalize(direction + perturb * (0.5 + depth * 0.1))

            # Reduce
            new_len = length * random.uniform(0.7, 0.9)
            new_rad = radius * random.uniform(0.6, 0.8)

            queue.append((end_pos, new_dir, new_len, new_rad, depth + 1))

    return objects

def main():
    objects = generate_tree()

    # Create Material
    # Bark material
    # We can use Procedural Noise (Bump/Displacement)
    materials = [{
        'Color': [0.15, 0.1, 0.05], # Dark Brown
        'Roughness': 0.9,
        'Bump': 0.5,
        'Displacement': 0.02, # Small displacement for bark detail
        'RoughnessMap': 'textures/brick_roughness.png', # Reusing existing textures for noise
        'NormalMap': 'textures/brick_normal.png',
        'UVScale': 2.0
    }]

    scene = {
        'Camera': {
            'Pos': [0, 4, 8],
            'LookAt': [0, 2, 0],
            'FOV': 45.0
        },
        'EnvironmentMap': '../textures/studio_small.jpg', # Assuming relative path
        'Materials': materials,
        'Objects': objects
    }

    with open('example/tree_design.yaml', 'w') as f:
        yaml.dump(scene, f)

    print(f"Generated {len(objects)} branches.")

if __name__ == "__main__":
    main()
