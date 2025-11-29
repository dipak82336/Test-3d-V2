import sys
import os
import numpy as np
import time

# Add Library to Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kalpana3d.utils.parser import parse_scene
from kalpana3d.core.mesher import generate_mesh, export_obj_mesh

def main():
    scenes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scenes'))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../gallery/models'))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    input_path = os.path.join(scenes_dir, '03_material_study.yaml')
    print(f"Loading Scene: {input_path}")
    scene = parse_scene(input_path)
    
    # Define bounds (Spheres are at -2.2, 0, 2.2)
    # Slightly offset bounds to avoid perfect alignment with grid/floor which causes artifacts
    min_bound = np.array([-3.51, -1.51, -1.51], dtype=np.float32)
    max_bound = np.array([3.51, 1.51, 1.51], dtype=np.float32)
    resolution = 0.12 # Fast resolution for iteration
    
    print("Generating Mesh (Marching Tetrahedra)...")
    start_time = time.time()
    
    # Add a Back Light for better 360 viewing (Not used for baking anymore, but kept for scene consistency)
    back_light = np.array([[0.0, 5.0, -5.0, 0.8, 0.8, 1.0, 5.0]], dtype=np.float32)
    scene['lights'] = np.vstack((scene['lights'], back_light))
    
    vertices, normals, tri_materials = generate_mesh(
        min_bound, max_bound, resolution,
        scene['num_objects'],
        scene['object_types'],
        scene['inv_matrices'],
        scene['scales'],
        scene['material_ids'],
        scene['operations'],
        scene['materials'],
        scene['lights']
    )
    
    print(f"Generated {len(vertices)//3} triangles in {time.time() - start_time:.2f}s")
    
    output_file = os.path.join(output_dir, '03_material_study.obj')
    print(f"Exporting to {output_file}...")
    export_obj_mesh(output_file, vertices, normals, tri_materials, scene['materials'])
    print("Done!")

if __name__ == "__main__":
    main()
