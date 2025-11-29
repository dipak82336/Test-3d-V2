import numpy as np
from kalpana3d.physics import create_aabb

class BVHNode:
    def __init__(self):
        self.aabb = np.array([0,0,0,0,0,0], dtype=np.float32)
        self.left = None
        self.right = None
        self.object_id = -1 # -1 means internal node

def build_bvh(objects, aabbs):
    # objects: list of object IDs
    # aabbs: dict or list mapping object_id -> aabb
    
    node = BVHNode()
    
    # 1. Calculate Union AABB
    min_p = np.array([1e9, 1e9, 1e9], dtype=np.float32)
    max_p = np.array([-1e9, -1e9, -1e9], dtype=np.float32)
    
    centers = []
    
    for obj_id in objects:
        box = aabbs[obj_id]
        # Update Union
        min_p = np.minimum(min_p, box[0:3])
        max_p = np.maximum(max_p, box[3:6])
        # Center for splitting
        center = (box[0:3] + box[3:6]) * 0.5
        centers.append((obj_id, center))
        
    node.aabb = create_aabb(min_p, max_p)
    
    # Leaf condition
    if len(objects) == 1:
        node.object_id = objects[0]
        return node
        
    # Split
    # Find longest axis
    extent = max_p - min_p
    axis = np.argmax(extent)
    
    # Sort by center along axis
    centers.sort(key=lambda x: x[1][axis])
    
    mid = len(centers) // 2
    left_objs = [x[0] for x in centers[:mid]]
    right_objs = [x[0] for x in centers[mid:]]
    
    if len(left_objs) == 0 or len(right_objs) == 0:
        # Fallback if split failed (e.g. all centers same)
        node.object_id = objects[0] # Just take one? No, this shouldn't happen with unique objects
        return node
        
    node.left = build_bvh(left_objs, aabbs)
    node.right = build_bvh(right_objs, aabbs)
    
    return node

def flatten_bvh(root):
    # Convert tree to array for Numba
    # Format: [minx, miny, minz, maxx, maxy, maxz, left_idx, right_idx, obj_id]
    nodes = []
    
    def traverse(node):
        idx = len(nodes)
        # Placeholder
        data = np.zeros(9, dtype=np.float32)
        data[0:6] = node.aabb
        data[8] = node.object_id
        nodes.append(data)
        
        if node.left:
            l_idx = traverse(node.left)
            nodes[idx][6] = l_idx
        else:
            nodes[idx][6] = -1
            
        if node.right:
            r_idx = traverse(node.right)
            nodes[idx][7] = r_idx
        else:
            nodes[idx][7] = -1
            
        return idx
        
    traverse(root)
    return np.array(nodes, dtype=np.float32)
