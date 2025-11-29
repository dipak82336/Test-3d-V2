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
        if level <= 0:
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

        # Branching
        if level > 1:
            num_branches = 3 if level > 2 else 2

            # Create a coordinate system for branching
            # Tangent and Bitangent
            if abs(direction[1]) < 0.9:
                tangent = np.cross(direction, np.array([0,1,0]))
            else:
                tangent = np.cross(direction, np.array([1,0,0]))
            tangent = tangent / np.linalg.norm(tangent)
            bitangent = np.cross(direction, tangent)

            for i in range(num_branches):
                # Spread branches
                angle_offset = (i / num_branches) * 2 * np.pi + (level * 1.5)
                spread_angle = 0.5 + (0.2 * np.random.random()) # ~30-40 degrees

                # Perturb direction
                new_dir = direction * np.cos(spread_angle) + \
                          (tangent * np.cos(angle_offset) + bitangent * np.sin(angle_offset)) * np.sin(spread_angle)
                new_dir = new_dir / np.linalg.norm(new_dir)

                # Decay
                new_length = length * 0.75
                new_radius = radius * 0.7

                add_branch(end_pos, new_dir, new_length, new_radius, level - 1)

    # Root Trunk
    start_pos = np.array([0, 0, 0], dtype=float)
    up_dir = np.array([0, 1, 0], dtype=float)

    # Base roots (optional, like the image)
    # The image has roots spreading out
    num_roots = 5
    for i in range(num_roots):
        angle = (i / num_roots) * 2 * np.pi
        root_dir = np.array([np.cos(angle), -0.5, np.sin(angle)])
        root_dir = root_dir / np.linalg.norm(root_dir)
        add_branch(start_pos, root_dir, 1.5, 0.25, 3)

    # Main Trunk
    add_branch(start_pos, up_dir, 2.0, 0.35, 6)

    scene = {
        'Camera': {
            'Pos': [0.0, 3.0, -6.0],
            'LookAt': [0.0, 2.0, 0.0],
            'FOV': 45.0
        },
        'EnvironmentMap': '../textures/studio_small.jpg',
        'Materials': [
            {
                'Color': [0.55, 0.27, 0.07], # Brown
                'Roughness': 0.9,
                'Metallic': 0.0,
                'Displacement': 0.05, # Bark effect
                'Bump': 0.2
            }
        ],
        'Objects': objects
    }

    with open(output_path, 'w') as f:
        yaml.dump(scene, f)

    print(f"Generated {len(objects)} objects")

if __name__ == "__main__":
    generate_tree_yaml("example/tree_generated.yaml")
