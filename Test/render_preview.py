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

def save_ppm(filename, width, height, buffer):
    with open(filename, 'w') as f:
        f.write(f"P3\n{width} {height}\n255\n")
        for i in range(width * height):
            r = int(min(max(buffer[i*3], 0), 1) * 255)
            g = int(min(max(buffer[i*3+1], 0), 1) * 255)
            b = int(min(max(buffer[i*3+2], 0), 1) * 255)
            f.write(f"{r} {g} {b} ")
        f.write("\n")

def main():
    scenes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scenes/Materials'))
    scene_path = 'scenes/Materials/textured_brick.yaml'
    input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', scene_path))
    
    print(f"Loading Scene: {input_path}")
    scene = parse_scene(input_path)
    print(f"Materials Shape: {scene['materials'].shape}")
    
    width = 640
    height = 480
    
    cam = scene['camera']
    cam_pos = np.array(cam['Pos'], dtype=np.float32)
    cam_target = np.array(cam['LookAt'], dtype=np.float32)
    fov = float(cam['FOV'])
    
    output_buffer = np.zeros(width * height * 3, dtype=np.float32)
    
    print("Rendering Preview...")
    
    # 3-Point Lighting Setup for Studio Realism
    # 1. Key Light (Main Sun - Warm, Casting Shadows)
    key_light = np.array([[5.0, 8.0, 5.0, 1.0, 0.9, 0.8, 1.2]], dtype=np.float32)
    
    # 2. Fill Light (Side - Cool, Soft, No Shadows)
    fill_light = np.array([[-6.0, 4.0, 2.0, 0.6, 0.7, 0.9, 0.5]], dtype=np.float32)
    
    # 3. Rim Light (Back - Bright, Highlights Edges)
    rim_light = np.array([[0.0, 5.0, -6.0, 0.9, 0.9, 1.0, 1.0]], dtype=np.float32)
    
    # Combine lights (Overwrite scene lights)
    scene['lights'] = np.vstack((key_light, fill_light, rim_light)).astype(np.float32)
    
    # Flatten Env Map for Numba
    env_map = scene['env_map']
    env_map_h, env_map_w, _ = env_map.shape
    env_map_flat = env_map.flatten().astype(np.float32)
    
    # Dummy small map
    # env_map_flat = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    # env_map_w = 1
    # env_map_h = 1
    
    texture_atlas = scene['texture_atlas']
    aabb_mins = scene['aabb_mins']
    aabb_maxs = scene['aabb_maxs']
    
    # print(f"Env Map Shape: {env_map.shape}, Dtype: {env_map.dtype}")
    
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
    
    if HAS_PIL:
        output_file = os.path.join(os.path.dirname(__file__), 'preview.png')
        print(f"Saving to {output_file}...")
        img_data = (output_buffer.reshape(height, width, 3) * 255).astype(np.uint8)
        img = Image.fromarray(img_data)
        img.save(output_file)
    else:
        output_file = os.path.join(os.path.dirname(__file__), 'preview.ppm')
        print(f"Saving to {output_file}...")
        save_ppm(output_file, width, height, output_buffer)
        
    print("Done!")

if __name__ == "__main__":
    main()
