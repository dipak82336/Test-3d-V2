import sys
import os
import numpy as np
import time

# Add Library to Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kalpana3d.utils.parser import parse_scene
from kalpana3d.core.engine import render_scene

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("PIL not found, falling back to PPM format.")

def save_image(filename, width, height, buffer):
    if HAS_PIL:
        print(f"Saving to {filename}...")
        img_data = (buffer.reshape(height, width, 3) * 255).astype(np.uint8)
        img = Image.fromarray(img_data)
        img.save(filename)
    else:
        ppm_filename = filename.rsplit('.', 1)[0] + '.ppm'
        print(f"Saving to {ppm_filename}...")
        with open(ppm_filename, 'w') as f:
            f.write(f"P3\n{width} {height}\n255\n")
            for i in range(width * height):
                r = int(min(max(buffer[i*3], 0), 1) * 255)
                g = int(min(max(buffer[i*3+1], 0), 1) * 255)
                b = int(min(max(buffer[i*3+2], 0), 1) * 255)
                f.write(f"{r} {g} {b} ")
            f.write("\n")

def render_view(scene, angle_name, cam_pos_override=None, output_path="output.png"):
    width = 320 # Low quality for speed as requested
    height = 240

    cam = scene['camera']
    if cam_pos_override is not None:
        cam_pos = np.array(cam_pos_override, dtype=np.float32)
    else:
        cam_pos = np.array(cam['Pos'], dtype=np.float32)

    cam_target = np.array(cam['LookAt'], dtype=np.float32)
    fov = float(cam['FOV'])

    output_buffer = np.zeros(width * height * 3, dtype=np.float32)

    print(f"Rendering {angle_name}...")

    # 3-Point Lighting Setup
    key_light = np.array([[5.0, 8.0, 5.0, 1.0, 0.9, 0.8, 1.2]], dtype=np.float32)
    fill_light = np.array([[-6.0, 4.0, 2.0, 0.6, 0.7, 0.9, 0.5]], dtype=np.float32)
    rim_light = np.array([[0.0, 5.0, -6.0, 0.9, 0.9, 1.0, 1.0]], dtype=np.float32)
    scene['lights'] = np.vstack((key_light, fill_light, rim_light)).astype(np.float32)

    env_map = scene['env_map']
    env_map_h, env_map_w, _ = env_map.shape
    env_map_flat = env_map.flatten().astype(np.float32)

    texture_atlas = scene['texture_atlas']
    aabb_mins = scene['aabb_mins']
    aabb_maxs = scene['aabb_maxs']

    start_time = time.time()
    render_scene(width, height, cam_pos, cam_target, fov,
                 scene['num_objects'], scene['object_types'],
                 scene['inv_matrices'], scene['scales'],
                 scene['material_ids'], scene['operations'],
                 scene['materials'], scene['lights'],
                 env_map_flat, env_map_w, env_map_h,
                 texture_atlas,
                 aabb_mins, aabb_maxs,
                 output_buffer)

    print(f"Rendered in {time.time() - start_time:.2f}s")
    save_image(output_path, width, height, output_buffer)

def main():
    # Detect which scene to render based on existence, default to realistic
    # Or just render both if they exist?
    # User asked for realistic one.

    target_scene = 'tree_realistic.yaml'

    scene_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f'../example/{target_scene}'))
    if not os.path.exists(scene_path):
        print(f"Scene not found: {scene_path}, falling back to tree.yaml")
        scene_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../example/tree.yaml'))

    print(f"Loading Scene: {scene_path}")
    scene = parse_scene(scene_path)

    # Angles: Front, Back, Left, Right
    # Assuming object is at 0,0,0
    dist = 6.0
    height = 3.5

    views = {
        "main": [0.0, height, dist],       # Front (Z+)
        "back": [0.0, height, -dist],      # Back (Z-)
        "left": [-dist, height, 0.0],      # Left (X-)
        "right": [dist, height, 0.0]       # Right (X+)
    }

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../example/image'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for name, pos in views.items():
        output_path = os.path.join(output_dir, f"{name}.png")
        render_view(scene, name, pos, output_path)

    print("All previews generated.")

if __name__ == "__main__":
    main()
