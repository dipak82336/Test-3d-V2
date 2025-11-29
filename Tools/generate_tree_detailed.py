from kalpana3d import render_yaml
import os
import sys
import yaml
import math
import numpy as np

def rotation_matrix_to_euler(R):
    # R is a 3x3 rotation matrix
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return [float(x), float(y), float(z)]

def get_rotation_from_vector(v):
    v = v / np.linalg.norm(v)
    up = np.array([0, 1, 0], dtype=float)

    dot = np.dot(v, up)
    if abs(dot) > 0.999:
        if dot > 0:
            return [0.0, 0.0, 0.0]
        else:
            return [float(np.pi), 0.0, 0.0]

    axis = np.cross(up, v)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(dot)

    # Rodrigues formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    I = np.eye(3)
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    return rotation_matrix_to_euler(R)

def generate_tree_yaml(output_path):
    objects = []

    def add_segment(p1, p2, radius):
        center = (p1 + p2) * 0.5
        length = np.linalg.norm(p2 - p1)
        if length < 0.001: return

        direction = (p2 - p1) / length
        rot = get_rotation_from_vector(direction)

        objects.append({
            'Type': 6, # Capsule
            'Pos': center.tolist(),
            'Rot': rot,
            'Scale': [float(radius), float(length), float(radius)],
            'Mat': 0,
            'Op': 3 # Smooth Union
        })

    def grow_branch(pos, direction, length, radius, level, max_levels):
        if level > max_levels or radius < 0.02:
            return

        # Curvature/Wobble
        # Divide branch into segments
        num_segments = max(1, int(length * 3)) # 3 segments per unit length
        seg_len = length / num_segments

        curr_pos = pos
        curr_dir = direction
        curr_rad = radius

        # Tapering
        taper_factor = 0.8 # Radius at end of branch relative to start
        rad_step = (radius * (1.0 - taper_factor)) / num_segments

        for i in range(num_segments):
            # Wobble
            if level < 2: # Trunk is stiffer
                wobble = np.random.uniform(-0.05, 0.05, 3)
            else:
                wobble = np.random.uniform(-0.2, 0.2, 3)

            next_dir = curr_dir + wobble
            next_dir = next_dir / np.linalg.norm(next_dir)

            # Gravity influence on thin branches
            if level > 3:
                next_dir[1] -= 0.1
                next_dir = next_dir / np.linalg.norm(next_dir)

            next_pos = curr_pos + next_dir * seg_len
            next_rad = max(0.01, curr_rad - rad_step)

            add_segment(curr_pos, next_pos, (curr_rad + next_rad)*0.5)

            curr_pos = next_pos
            curr_dir = next_dir
            curr_rad = next_rad

        # Branching at the end
        if level < max_levels:
            # Determine number of child branches
            # Trunk (level 0) -> 2-3 main branches
            # Branches -> 2 children
            num_children = 0
            if level == 0: num_children = np.random.randint(2, 4)
            else: num_children = np.random.randint(1, 3)

            # Coordinate frame
            if abs(curr_dir[1]) < 0.9:
                tangent = np.cross(curr_dir, np.array([0,1,0]))
            else:
                tangent = np.cross(curr_dir, np.array([1,0,0]))
            tangent = tangent / np.linalg.norm(tangent)
            bitangent = np.cross(curr_dir, tangent)

            for i in range(num_children):
                angle_offset = (i / num_children) * 2 * np.pi + np.random.uniform(0, 1)
                spread = np.random.uniform(0.5, 1.0) # 30-60 degrees

                child_dir = curr_dir * np.cos(spread) + \
                            (tangent * np.cos(angle_offset) + bitangent * np.sin(angle_offset)) * np.sin(spread)
                child_dir = child_dir / np.linalg.norm(child_dir)

                # Parameters for child
                child_len = length * np.random.uniform(0.6, 0.8)
                child_rad = curr_rad * np.random.uniform(0.6, 0.8)

                grow_branch(curr_pos, child_dir, child_len, child_rad, level + 1, max_levels)

    # --- Generation ---

    origin = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])

    # Roots
    num_roots = 7
    for i in range(num_roots):
        angle = (i / num_roots) * 2 * np.pi + np.random.uniform(-0.1, 0.1)
        root_dir = np.array([np.cos(angle), -0.5, np.sin(angle)])
        root_dir = root_dir / np.linalg.norm(root_dir)
        grow_branch(origin, root_dir, 1.2, 0.35, 3, 5) # Treat as level 3 to limit recursion

    # Main Trunk
    grow_branch(origin, up, 1.8, 0.45, 0, 5) # Start level 0, max 5

    scene = {
        'Camera': {
            'Pos': [0.0, 3.0, -6.0],
            'LookAt': [0.0, 2.0, 0.0],
            'FOV': 45.0
        },
        'EnvironmentMap': '../textures/studio_small.jpg',
        'Materials': [
            {
                'Color': [0.45, 0.35, 0.25],
                'Roughness': 0.8,
                'Metallic': 0.0,
                'Displacement': 0.02, # Subtle bark
                'Bump': 0.2,
                'UVScale': 2.0
            }
        ],
        'Objects': objects
    }

    with open(output_path, 'w') as f:
        yaml.dump(scene, f)

    print(f"Generated {len(objects)} objects")

if __name__ == "__main__":
    generate_tree_yaml("example/tree_generated.yaml")
