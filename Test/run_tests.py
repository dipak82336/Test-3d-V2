import sys
import os
import numpy as np

# Add Library to Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kalpana3d import render_yaml
from kalpana3d.utils.parser import parse_scene
from kalpana3d.core.engine import render_scene as engine_render
from PIL import Image

def render_multi_view(yaml_path, output_base, width=400, height=300):
    print(f"Loading Scene: {yaml_path}")
    scene = parse_scene(yaml_path)
    
    # Extract new data
    env_map = scene['env_map']
    env_map_h, env_map_w, _ = env_map.shape
    env_map_flat = env_map.flatten().astype(np.float32)
    
    texture_atlas = scene['texture_atlas']
    aabb_mins = scene['aabb_mins']
    aabb_maxs = scene['aabb_maxs']
    
    # Views
    views = [
        ('main', scene['camera']['Pos'], scene['camera']['LookAt']),
        # ('front', [0, 2, 8], [0, 0, 0]),
        # ('back', [0, 2, -8], [0, 0, 0]),
        # ('left', [-8, 2, 0], [0, 0, 0]),
        # ('right', [8, 2, 0], [0, 0, 0])
    ]
    
    buffer = np.zeros((height * width * 3), dtype=np.float32)
    
    for name, pos, target in views:
        print(f"  Rendering View: {name}")
        
        engine_render(
            width, height, 
            np.array(pos, dtype=np.float32),
            np.array(target, dtype=np.float32),
            float(scene['camera']['FOV']),
            scene['num_objects'],
            scene['object_types'],
            scene['inv_matrices'],
            scene['scales'],
            scene['material_ids'],
            scene['operations'],
            scene['materials'],
            scene['lights'],
            env_map_flat, env_map_w, env_map_h,
            texture_atlas,
            aabb_mins, aabb_maxs,
            buffer
        )
        
        img_data = (buffer.reshape((height, width, 3)) * 255).astype(np.uint8)
        out_file = f"{output_base}_{name}.png"
        Image.fromarray(img_data).save(out_file)

def main():
    scenes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scenes'))
    # Output to scenes/Materials as requested
    output_dir = os.path.join(scenes_dir, 'Materials')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Scan for Material Tests
    materials_dir = os.path.join(scenes_dir, 'Materials')
    tests = []
    if os.path.exists(materials_dir):
        for f in os.listdir(materials_dir):
            if f.endswith('.yaml') and f.startswith('test_'):
                tests.append(os.path.join('Materials', f))
    
    print(f"Found {len(tests)} Material Tests")
    
    for test in tests:
        input_path = os.path.join(scenes_dir, test)
        # Save directly in Materials folder with prefix
        output_base = os.path.join(scenes_dir, test.replace('.yaml', '_render'))
        
        if os.path.exists(input_path):
            print(f"Running Test Suite: {test}")
            try:
                render_multi_view(input_path, output_base)
            except Exception as e:
                print(f"FAILED: {test} - {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"MISSING: {input_path}")

if __name__ == "__main__":
    main()
