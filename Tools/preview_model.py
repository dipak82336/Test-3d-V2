import os
import sys
import argparse
import numpy as np
from PIL import Image
import yaml

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kalpana3d.utils.parser import parse_scene
from kalpana3d.core.engine import render_scene

def render_view(scene_data, width, height, cam_pos, cam_target, output_path):
    print(f"Rendering view to {output_path}...")

    # Create output buffer
    output_buffer = np.zeros(width * height * 3, dtype=np.float32)

    # Render
    render_scene(
        width, height,
        np.array(cam_pos, dtype=np.float32),
        np.array(cam_target, dtype=np.float32),
        float(scene_data['camera']['FOV']),
        scene_data['num_objects'],
        scene_data['object_types'],
        scene_data['inv_matrices'],
        scene_data['scales'],
        scene_data['material_ids'],
        scene_data['operations'],
        scene_data['materials'],
        scene_data['lights'],
        scene_data['env_map'].reshape(-1), # flatten for numba
        scene_data['env_map'].shape[1], # width
        scene_data['env_map'].shape[0], # height
        scene_data['texture_atlas'],
        scene_data['aabb_mins'],
        scene_data['aabb_maxs'],
        output_buffer
    )

    # Reshape and save
    img_data = output_buffer.reshape(height, width, 3)
    img_data = np.clip(img_data * 255, 0, 255).astype(np.uint8)

    img = Image.fromarray(img_data)
    img.save(output_path)
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Render preview images for a Kalpana3D model")
    parser.add_argument("yaml_file", help="Path to the YAML scene file")
    parser.add_argument("--width", type=int, default=512, help="Output width")
    parser.add_argument("--height", type=int, default=512, help="Output height")
    parser.add_argument("--dist", type=float, default=5.0, help="Camera distance")

    args = parser.parse_args()

    if not os.path.exists(args.yaml_file):
        print(f"Error: File {args.yaml_file} not found")
        return

    # Create output directory
    output_dir = os.path.join(os.path.dirname(args.yaml_file), "image")
    os.makedirs(output_dir, exist_ok=True)

    # Parse scene
    print("Parsing scene...")
    scene_data = parse_scene(args.yaml_file)

    center = np.array([0, 0, 0])

    # Render views
    views = {
        "main": [0, 0, args.dist], # Front (Z+)
        "front": [0, 0, args.dist],
        "back": [0, 0, -args.dist],
        "left": [-args.dist, 0, 0],
        "right": [args.dist, 0, 0]
    }

    # Adjust lookat to center of bounding box?
    # For now assume model is centered at 0,0,0

    for name, pos in views.items():
        # If camera is defined in YAML, use it for "main"?
        # User requested specific views.

        # Adjust Y to look slightly down
        cam_pos = np.array(pos)
        cam_pos[1] += args.dist * 0.2

        output_path = os.path.join(output_dir, f"{name}.png")
        render_view(scene_data, args.width, args.height, cam_pos, center, output_path)

if __name__ == "__main__":
    main()
