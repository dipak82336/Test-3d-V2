import numpy as np
from PIL import Image
import math

def generate_studio_env_map(width=1024, height=512):
    # Create empty float buffer
    buffer = np.zeros((height, width, 3), dtype=np.float32)
    
    # 1. Background Gradient (Dark Blue-Grey to Black)
    for y in range(height):
        v = y / height
        # Top (Zenith) = Dark
        # Bottom (Nadir) = Slightly lighter (ground reflection)
        col = np.array([0.05, 0.05, 0.08]) * (1.0 - v) + np.array([0.1, 0.1, 0.12]) * v
        buffer[y, :] = col

    # 2. Add Soft Lights (Studio Setup)
    
    def add_light(u_center, v_center, u_radius, v_radius, intensity, color):
        # Simple rectangular soft light with falloff
        for y in range(height):
            v = y / height
            dv = (v - v_center) / v_radius
            if abs(dv) > 1.0: continue
            
            for x in range(width):
                u = x / width
                # Handle wrapping for u
                du = min(abs(u - u_center), abs(u - u_center + 1.0), abs(u - u_center - 1.0)) / u_radius
                
                if du > 1.0: continue
                
                # Soft falloff (cosine-like)
                falloff = (0.5 + 0.5 * math.cos(du * math.pi)) * (0.5 + 0.5 * math.cos(dv * math.pi))
                
                buffer[y, x] += color * intensity * falloff

    # Key Light (Warm, Right)
    add_light(0.1, 0.4, 0.15, 0.2, 5.0, np.array([1.0, 0.9, 0.8]))
    
    # Fill Light (Cool, Left)
    add_light(0.6, 0.4, 0.2, 0.3, 2.0, np.array([0.8, 0.9, 1.0]))
    
    # Rim Light (Bright, Back)
    add_light(0.85, 0.6, 0.1, 0.1, 8.0, np.array([1.0, 1.0, 1.0]))
    
    # Top Light (Overhead)
    add_light(0.5, 0.1, 0.3, 0.1, 3.0, np.array([0.9, 0.9, 0.9]))

    # Clip and Save
    buffer = np.clip(buffer, 0.0, 1.0)
    img_data = (buffer * 255).astype(np.uint8)
    img = Image.fromarray(img_data)
    
    import os
    os.makedirs('textures', exist_ok=True)
    img.save('textures/studio_small.jpg', quality=95)
    print("Generated textures/studio_small.jpg")

if __name__ == "__main__":
    generate_studio_env_map()
