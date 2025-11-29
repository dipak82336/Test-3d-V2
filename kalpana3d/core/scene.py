import numpy as np
from .math import vec3, normalize

def make_translation(x, y, z):
    return np.array([[1,0,0,x], [0,1,0,y], [0,0,1,z], [0,0,0,1]], dtype=np.float32)

def make_scale(x, y, z):
    return np.array([[x,0,0,0], [0,y,0,0], [0,0,z,0], [0,0,0,1]], dtype=np.float32)

def make_rotation(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    Rx = np.array([[1,0,0,0], [0,cx,-sx,0], [0,sx,cx,0], [0,0,0,1]], dtype=np.float32)
    Ry = np.array([[cy,0,sy,0], [0,1,0,0], [-sy,0,cy,0], [0,0,0,1]], dtype=np.float32)
    Rz = np.array([[cz,-sz,0,0], [sz,cz,0,0], [0,0,1,0], [0,0,0,1]], dtype=np.float32)
    
    return Rz @ Ry @ Rx

class SceneNode:
    def __init__(self, name="Node", type_id=-1, parent=None):
        self.name = name
        self.type_id = type_id
        self.parent = parent
        self.children = []
        
        self.position = np.array([0,0,0], dtype=np.float32)
        self.rotation = np.array([0,0,0], dtype=np.float32)
        self.scale = np.array([1,1,1], dtype=np.float32)
        
        self.material_id = 0
        self.local_matrix = np.eye(4, dtype=np.float32)
        self.world_matrix = np.eye(4, dtype=np.float32)
        self.inv_world_matrix = np.eye(4, dtype=np.float32)
        
        if parent:
            parent.add_child(self)
            
    def add_child(self, child):
        self.children.append(child)
        child.parent = self
        
    def update(self):
        # 1. Local Matrix (T * R) - Scale is handled separately for SDFs
        T = make_translation(*self.position)
        R = make_rotation(*self.rotation)
        self.local_matrix = T @ R
        
        # 2. World Matrix
        if self.parent:
            self.world_matrix = self.parent.world_matrix @ self.local_matrix
        else:
            self.world_matrix = self.local_matrix
            
        # 3. Inverse World Matrix (for Raymarching)
        self.inv_world_matrix = np.linalg.inv(self.world_matrix).astype(np.float32)
        
        # 4. Propagate
        for child in self.children:
            child.update()
            
    def collect_render_data(self, object_types, inv_matrices, scales, material_ids):
        if self.type_id >= 0:
            object_types.append(self.type_id)
            inv_matrices.append(self.inv_world_matrix)
            # Global scale propagation is tricky in SDFs. 
            # We assume uniform scaling or handle it carefully.
            # For now, we pass local scale.
            scales.append(self.scale)
            material_ids.append(self.material_id)
            
        for child in self.children:
            child.collect_render_data(object_types, inv_matrices, scales, material_ids)

class Scene:
    def __init__(self):
        self.root = SceneNode("Root")
        self.materials = []
        self.lights = []
        self.camera = {'Pos': [8,5,8], 'LookAt': [0,2,0], 'FOV': 50}
        
    def add(self, node):
        self.root.add_child(node)
        
    def update(self):
        self.root.update()
        
    def get_render_data(self):
        object_types = []
        inv_matrices = []
        scales = []
        material_ids = []
        
        self.root.collect_render_data(object_types, inv_matrices, scales, material_ids)
        
        return {
            'num_objects': len(object_types),
            'object_types': np.array(object_types, dtype=np.int32),
            'inv_matrices': np.array(inv_matrices, dtype=np.float32),
            'scales': np.array(scales, dtype=np.float32),
            'material_ids': np.array(material_ids, dtype=np.float32),
            'materials': np.array(self.materials, dtype=np.float32) if self.materials else np.array([[0.8,0.8,0.8,0,0.5]], dtype=np.float32),
            'lights': np.array(self.lights, dtype=np.float32) if self.lights else np.array([[5,8,5,1,1,1,1]], dtype=np.float32),
            'camera': self.camera
        }
