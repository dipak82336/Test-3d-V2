from kalpana3d import render_yaml
import os
import sys
import yaml
import math
import numpy as np

def generate_tree_yaml(output_path):
    # Procedural generation of a tree

    # Tree parameters
    trunk_radius = 0.2
    trunk_height = 2.0
    branch_levels = 5

    objects = []

    # Recursive function to generate branches
    def add_branch(pos, direction, length, radius, level):
        if level <= 0:
            return

        # Add branch segment (Capsule)
        end_pos = [
            pos[0] + direction[0] * length,
            pos[1] + direction[1] * length,
            pos[2] + direction[2] * length
        ]

        # Calculate rotation for capsule
        # Capsule is aligned with Y axis by default
        # We need to rotate it to match 'direction'
        # Simplified: Just place it and use rotation in YAML if we can calculate Euler angles
        # OR: Just place capsules using endpoints?
        # The library sdCapsule takes (p, a, b, r) where a and b are endpoints in LOCAL space.
        # But our objects have Pos, Rot, Scale.
        # If we use sdCapsule from library, it is aligned Y with height 'Scale[1]' and radius 'Scale[0]' centered at 0.
        # So it goes from (0, -h/2, 0) to (0, h/2, 0).

        # We need to position and rotate this capsule so it goes from 'pos' to 'end_pos'.
        mid_pos = [
            (pos[0] + end_pos[0]) / 2,
            (pos[1] + end_pos[1]) / 2,
            (pos[2] + end_pos[2]) / 2
        ]

        # Calculate rotation (Quaternion or Euler)
        # Direction vector
        d = np.array(direction)
        up = np.array([0, 1, 0])

        # Rotation axis
        axis = np.cross(up, d)
        norm_axis = np.linalg.norm(axis)

        rot = [0, 0, 0] # Euler angles

        if norm_axis < 0.001:
            # Parallel
            if np.dot(up, d) < 0:
                rot = [np.pi, 0, 0]
        else:
            angle = np.arccos(np.dot(up, d))
            axis = axis / norm_axis
            # Convert Axis-Angle to Euler (XYZ) is painful.
            # But the parser takes 'Rot' as Euler angles (radians).
            # Let's approximate or use a library if available.
            # Or better: Update library to support 'Capsule endpoints' directly?
            # No, standard transform is better.

            # Simple workaround: Just use many small spheres or overlapped cylinders? No, that's ugly.
            # Let's implement LookAt rotation for the object.

            # Use scipy rotation if available?
            # Or write a small helper.
            pass

        # Actually, let's keep it simple.
        # We will assume we can calculate the rotation.
        # For now, let's just build a vertical trunk to test.

        objects.append({
            'Type': 6, # Capsule
            'Pos': mid_pos,
            'Rot': [0, 0, 0], # Placeholder
            'Scale': [radius, length, radius],
            'Mat': 0,
            'Op': 3 # Smooth Union
        })

        # Recursion
        # ...

    # Let's just create a static detailed tree structure manually for now using simple math
    # Root
    objects.append({
        'Type': 6, # Capsule
        'Pos': [0, 1.0, 0],
        'Rot': [0, 0, 0],
        'Scale': [0.15, 2.0, 0.15],
        'Mat': 0,
        'Op': 0 # Union
    })

    # Branches (hardcoded for test)
    # Branch 1 (Right)
    # Rotated 45 deg Z
    objects.append({
        'Type': 6,
        'Pos': [0.5, 1.5, 0],
        'Rot': [0, 0, -0.785],
        'Scale': [0.1, 1.5, 0.1],
        'Mat': 0,
        'Op': 3
    })

     # Branch 2 (Left)
    objects.append({
        'Type': 6,
        'Pos': [-0.5, 1.5, 0],
        'Rot': [0, 0, 0.785],
        'Scale': [0.1, 1.5, 0.1],
        'Mat': 0,
        'Op': 3
    })

    scene = {
        'Camera': {
            'Pos': [0.0, 2.0, -4.0],
            'LookAt': [0.0, 1.5, 0.0],
            'FOV': 45.0
        },
        'EnvironmentMap': '../textures/studio_small.jpg',
        'Materials': [
            {
                'Color': [0.55, 0.27, 0.07], # Brown
                'Roughness': 0.8,
                'Metallic': 0.0,
                'Displacement': 0.02 # Bark effect
            }
        ],
        'Objects': objects
    }

    with open(output_path, 'w') as f:
        yaml.dump(scene, f)

if __name__ == "__main__":
    generate_tree_yaml("example/tree_generated.yaml")
