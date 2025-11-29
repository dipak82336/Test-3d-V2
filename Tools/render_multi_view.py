from kalpana3d import render_yaml
import os
import sys
import yaml
import numpy as np

def main():
    if len(sys.argv) < 3:
        print("Usage: python render_multi_view.py <yaml_file> <output_dir>")
        return

    yaml_file = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(yaml_file, 'r') as f:
        scene = yaml.safe_load(f)

    # Get center of tree (approx height 2.5 based on generation)
    center = [0.0, 2.5, 0.0]
    if 'LookAt' in scene.get('Camera', {}):
        center = scene['Camera']['LookAt']

    dist = 9.0
    height = 3.5

    # Views: main, front, back, left, right
    # Assuming 'front' is +Z or -Z. Let's say Front is viewing from +Z looking at -Z?
    # Standard: Front view usually looks along -Z or +Z.
    # Let's align with the generated camera pos [0, 3, -5] (which is looking towards +Z if center is 0)
    # Wait, [0, 3, -5] looking at [0, 2.5, 0] is looking towards +Z.

    views = {
        'main': [0, height, -dist], # Same as default camera
        'front': [0, height, -dist],
        'back': [0, height, dist],
        'left': [-dist, height, 0],
        'right': [dist, height, 0]
    }

    # Keep quality low for preview speed
    width = 320
    height_px = 240

    for name, pos in views.items():
        out_img = f"{output_dir}/{name}.png"
        if os.path.exists(out_img):
            print(f"Skipping {name} (already exists)")
            continue

        print(f"Rendering {name} view...")
        scene['Camera']['Pos'] = pos
        scene['Camera']['LookAt'] = center

        temp_yaml = f"{output_dir}/temp_{name}.yaml"
        with open(temp_yaml, 'w') as f:
            yaml.dump(scene, f)

        render_yaml(temp_yaml, out_img, width=width, height=height_px)

        if os.path.exists(temp_yaml):
            os.remove(temp_yaml)

if __name__ == "__main__":
    main()
