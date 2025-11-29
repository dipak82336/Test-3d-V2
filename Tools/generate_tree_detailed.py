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

    return [x, y, z]

def get_rotation_from_vector(v):
    # Default vector is Y (0,1,0)
    # Target is v
    v = v / np.linalg.norm(v)
    up = np.array([0, 1, 0], dtype=float)

    if np.abs(np.dot(v, up)) > 0.999:
        if np.dot(v, up) > 0:
            return [0, 0, 0]
        else:
            return [np.pi, 0, 0]

    axis = np.cross(up, v)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.dot(up, v))

    # Rodrigues formula to get rotation matrix
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

    # Recursive function to generate branches
    def add_branch(pos, direction, length, radius, level):
        if level <= 0 or radius < 0.01:
            return

        direction = direction / np.linalg.norm(direction)
        end_pos = pos + direction * length

        # Midpoint
        mid_pos = (pos + end_pos) * 0.5

        # Rotation
        rot = get_rotation_from_vector(direction)

        # Capsule segment
        objects.append({
            'Type': 6, # Capsule
            'Pos': mid_pos.tolist(),
            'Rot': rot,
            'Scale': [radius, length, radius],
            'Mat': 0,
            'Op': 3 # Smooth Union
        })

        # Increase branching logic
        if level > 1:
            # Force more branches at lower levels to fill the tree
            if level >= 5:
                num_branches = np.random.randint(3, 5)
            else:
                num_branches = np.random.randint(2, 4)

            # Create a coordinate system for branching
            if abs(direction[1]) < 0.9:
                tangent = np.cross(direction, np.array([0,1,0]))
            else:
                tangent = np.cross(direction, np.array([1,0,0]))
            tangent = tangent / np.linalg.norm(tangent)
            bitangent = np.cross(direction, tangent)

            for i in range(num_branches):
                # Spread branches distribution
                angle_offset = (i / num_branches) * 2 * np.pi + np.random.uniform(-0.5, 0.5)
                spread_angle = np.random.uniform(0.4, 0.9)

                # Perturb direction
                new_dir = direction * np.cos(spread_angle) + \
                          (tangent * np.cos(angle_offset) + bitangent * np.sin(angle_offset)) * np.sin(spread_angle)

                new_dir = new_dir + np.array([0, 0.3, 0]) # More up bias
                new_dir = new_dir / np.linalg.norm(new_dir)

                # Decay
                length_decay = np.random.uniform(0.7, 0.9)
                radius_decay = np.random.uniform(0.6, 0.75)

                new_length = length * length_decay
                new_radius = radius * radius_decay

                add_branch(end_pos, new_dir, new_length, new_radius, level - 1)

    # Root Trunk
    start_pos = np.array([0, 0, 0], dtype=float)
    up_dir = np.array([0, 1, 0], dtype=float)

    # Base roots
    num_roots = 8 # More roots
    for i in range(num_roots):
        angle = (i / num_roots) * 2 * np.pi + np.random.uniform(-0.2, 0.2)
        root_dir = np.array([np.cos(angle), -0.4, np.sin(angle)])
        root_dir = root_dir / np.linalg.norm(root_dir)

        curr_pos = start_pos.copy()
        curr_dir = root_dir
        curr_len = 0.8
        curr_rad = 0.35

        for r_level in range(3):
             end_p = curr_pos + curr_dir * curr_len
             mid_p = (curr_pos + end_p) * 0.5
             rot = get_rotation_from_vector(curr_dir)
             objects.append({
                'Type': 6,
                'Pos': mid_p.tolist(),
                'Rot': rot,
                'Scale': [curr_rad, curr_len, curr_rad],
                'Mat': 0,
                'Op': 3
             })
             curr_pos = end_p
             noise = np.random.uniform(-0.3, 0.3, 3)
             curr_dir = curr_dir + noise
             curr_dir[1] = -abs(curr_dir[1]) * 0.5
             curr_dir = curr_dir / np.linalg.norm(curr_dir)
             curr_len *= 0.8
             curr_rad *= 0.6

    # Main Trunk
    trunk_segments = 4
    curr_pos = start_pos.copy()
    curr_dir = up_dir
    curr_len = 0.6
    curr_rad = 0.45

    for i in range(trunk_segments):
        end_p = curr_pos + curr_dir * curr_len
        mid_p = (curr_pos + end_p) * 0.5
        rot = get_rotation_from_vector(curr_dir)
        objects.append({
            'Type': 6,
            'Pos': mid_p.tolist(),
            'Rot': rot,
            'Scale': [curr_rad, curr_len, curr_rad],
            'Mat': 0,
            'Op': 3
        })
        curr_pos = end_p
        curr_dir = curr_dir + np.random.uniform(-0.1, 0.1, 3)
        curr_dir = curr_dir / np.linalg.norm(curr_dir)
        curr_rad *= 0.9

    # Start branching (Increase depth to 6)
    add_branch(curr_pos, curr_dir, 1.0, curr_rad, 6)

    scene = {
        'Camera': {
            'Pos': [0.0, 3.5, -6.0],
            'LookAt': [0.0, 3.0, 0.0],
            'FOV': 50.0
        },
        'EnvironmentMap': '../textures/studio_small.jpg',
        'Materials': [
            {
                'Color': [0.4, 0.3, 0.2],
                'Roughness': 0.9,
                'Metallic': 0.0,
                'Displacement': 0.08,
                'Bump': 0.4,
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
