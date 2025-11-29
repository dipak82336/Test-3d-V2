from .core.engine import render_scene
from .utils.parser import parse_scene
import numpy as np
from PIL import Image
import time

def render_yaml(yaml_path, output_path, width=800, height=600):
    print(f"Loading Scene: {yaml_path}")
    scene = parse_scene(yaml_path)

    print("Compiling Engine...")
    buffer = np.zeros((height * width * 3), dtype=np.float32)

    # Flatten Env Map
    env_map = scene['env_map']
    env_map_h, env_map_w, _ = env_map.shape
    env_map_flat = env_map.reshape(-1)

    start = time.time()
    render_scene(
        width, height,
        np.array(scene['camera']['Pos'], dtype=np.float32),
        np.array(scene['camera']['LookAt'], dtype=np.float32),
        float(scene['camera']['FOV']),
        scene['num_objects'],
        scene['object_types'],
        scene['inv_matrices'],
        scene['scales'],
        scene['material_ids'],
        scene['operations'],
        scene['materials'],
        scene['lights'],
        env_map_flat,
        env_map_w,
        env_map_h,
        scene['texture_atlas'],
        scene['aabb_mins'],
        scene['aabb_maxs'],
        buffer
    )
    end = time.time()
    print(f"Render Time: {end - start:.4f}s")

    img_data = (buffer.reshape((height, width, 3)) * 255).astype(np.uint8)
    Image.fromarray(img_data).save(output_path)
    print(f"Saved: {output_path}")
