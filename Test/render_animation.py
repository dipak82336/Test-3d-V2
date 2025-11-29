import sys
import os
import numpy as np
import math
from PIL import Image

# Add Library to Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kalpana3d.utils.parser import parse_scene, make_translation, make_rotation
from kalpana3d.core.engine import render_scene as engine_render

def render_animation(yaml_path, output_dir, frames=60, fps=30):
    print(f"Loading Scene: {yaml_path}")
    scene = parse_scene(yaml_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    width = 400
    height = 300
    buffer = np.zeros((height * width * 3), dtype=np.float32)
    
    print(f"Rendering {frames} frames...")
    
    for i in range(frames):
        t = i / float(frames) # 0.0 to 1.0
        angle = t * 2.0 * math.pi # 0 to 2pi
        
        # --- ANIMATION LOGIC ---
        
        # Object 2: Ring 1 (Rotate around Z)
        # Original: Pos [0,0,0], Rot [0,0,0.5]
        # We want to spin it.
        rot_z = 0.5 + angle
        T = make_translation(0, 0, 0)
        R = make_rotation(0, 0, rot_z)
        mat = T @ R
        scene['inv_matrices'][2] = np.linalg.inv(mat).astype(np.float32)
        
        # Object 3: Ring 2 (Rotate around X)
        rot_x = 0.5 - angle * 2.0 # Spin faster opposite way
        T = make_translation(0, 0, 0)
        R = make_rotation(rot_x, 0, 0)
        mat = T @ R
        scene['inv_matrices'][3] = np.linalg.inv(mat).astype(np.float32)
        
        # Object 1: Core (Pulse Scale)
        # Scale is in scene['scales'][1]
        pulse = 0.8 + 0.1 * math.sin(angle * 4.0)
        scene['scales'][1] = np.array([pulse, pulse, pulse], dtype=np.float32)
        
        # Camera Orbit
        cam_radius = 5.0
        cam_x = math.sin(angle) * cam_radius
        cam_z = math.cos(angle) * cam_radius
        cam_pos = np.array([cam_x, 3.0, cam_z], dtype=np.float32)
        
        # -----------------------
        
        engine_render(
            width, height, 
            cam_pos,
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
            buffer
        )
        
        # --- POST PROCESSING: BLOOM ---
        img = buffer.reshape((height, width, 3))
        
        # 1. Extract bright areas (Threshold)
        threshold = 1.0
        bright = np.maximum(img - threshold, 0.0)
        
        # 2. Blur (Simple Box Blur for speed, repeated for Gaussian approx)
        # We can't use scipy, so we do manual simple blur
        def simple_blur(arr, r=2):
            # Very basic separable blur
            h, w, c = arr.shape
            out = np.copy(arr)
            # Horizontal
            for y in range(h):
                for x in range(r, w-r):
                    out[y, x] = np.mean(arr[y, x-r:x+r+1], axis=0)
            # Vertical
            temp = np.copy(out)
            for x in range(w):
                for y in range(r, h-r):
                    out[y, x] = np.mean(temp[y-r:y+r+1, x], axis=0)
            return out

        # Multi-pass blur for better quality
        blur1 = simple_blur(bright, r=2)
        blur2 = simple_blur(blur1, r=4)
        
        # 3. Add back to original
        final_img = img + blur2 * 2.0 # Intensity
        
        # Tone Map again? No, already tone mapped in engine. 
        # Just clamp.
        final_img = np.clip(final_img, 0.0, 1.0)
        
        img_data = (final_img * 255).astype(np.uint8)
        out_file = os.path.join(output_dir, f"frame_{i:03d}.png")
        Image.fromarray(img_data).save(out_file)
        print(f"  Rendered Frame {i+1}/{frames}")

if __name__ == "__main__":
    scenes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scenes'))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../gallery/animation'))
    
    input_path = os.path.join(scenes_dir, '08_complex_structure.yaml')
    render_animation(input_path, output_dir, frames=5) # 5 frames for testing bloom
