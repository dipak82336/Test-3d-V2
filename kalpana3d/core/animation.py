import os
import numpy as np
from PIL import Image
import time
from .engine import render_scene
from ..utils.parser import parse_scene

def render_animation(yaml_path, output_dir, frames=60, width=800, height=600):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Loading Scene: {yaml_path}")
    scene = parse_scene(yaml_path)
    
    print(f"Rendering {frames} Frames...")
    
    buffer = np.zeros((height * width * 3), dtype=np.float32)
    
    base_cam_pos = np.array(scene['camera']['Pos'], dtype=np.float32)
    look_at = np.array(scene['camera']['LookAt'], dtype=np.float32)
    
    for i in range(frames):
        t = i / float(frames)
        angle = t * np.pi * 2.0
        
        # Orbit Camera
        radius = np.sqrt(base_cam_pos[0]**2 + base_cam_pos[2]**2)
        cam_x = np.sin(angle) * radius
        cam_z = np.cos(angle) * radius
        cam_pos = np.array([cam_x, base_cam_pos[1], cam_z], dtype=np.float32)
        
        # Render
        print(f"Frame {i+1}/{frames}")
        render_scene(
            width, height, 
            cam_pos, look_at, float(scene['camera']['FOV']),
            scene['num_objects'], scene['object_types'], 
            scene['inv_matrices'], scene['scales'], 
            scene['material_ids'], scene['materials'], scene['lights'], 
            buffer
        )
        
        # Save
        img_data = (buffer.reshape((height, width, 3)) * 255).astype(np.uint8)
        Image.fromarray(img_data).save(f"{output_dir}/frame_{i:03d}.png")
        
    print("Animation Complete.")
