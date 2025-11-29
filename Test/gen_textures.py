import numpy as np
from PIL import Image
import os

def generate_brick_textures():
    width = 512
    height = 512
    
    # Create textures directory if it doesn't exist
    if not os.path.exists('textures'):
        os.makedirs('textures')
        
    # 1. Albedo Map (Red Bricks with Grey Mortar)
    albedo = np.zeros((height, width, 3), dtype=np.uint8)
    # Brick color: Dark Red
    brick_col = np.array([180, 60, 40], dtype=np.float32)
    # Mortar color: Light Grey
    mortar_col = np.array([150, 150, 150], dtype=np.float32)
    
    brick_w = 60
    brick_h = 30
    mortar_size = 4
    
    for y in range(height):
        row = y // (brick_h + mortar_size)
        offset = 0 if row % 2 == 0 else brick_w // 2
        
        for x in range(width):
            # Check if mortar
            in_mortar_y = (y % (brick_h + mortar_size)) >= brick_h
            
            tx = (x + offset) % (brick_w + mortar_size)
            in_mortar_x = tx >= brick_w
            
            if in_mortar_y or in_mortar_x:
                col = mortar_col
            else:
                # Add some noise to brick
                noise = np.random.uniform(0.8, 1.0)
                col = brick_col * noise
                
            albedo[y, x] = col.astype(np.uint8)
            
    Image.fromarray(albedo).save('textures/brick_albedo.png')
    print("Saved textures/brick_albedo.png")
    
    # 2. Roughness Map (Bricks rough, Mortar very rough)
    roughness = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(height):
        row = y // (brick_h + mortar_size)
        offset = 0 if row % 2 == 0 else brick_w // 2
        
        for x in range(width):
            in_mortar_y = (y % (brick_h + mortar_size)) >= brick_h
            tx = (x + offset) % (brick_w + mortar_size)
            in_mortar_x = tx >= brick_w
            
            if in_mortar_y or in_mortar_x:
                val = 255 # Very rough
            else:
                val = 100 # Medium rough
                
            roughness[y, x] = val
            
    Image.fromarray(roughness).save('textures/brick_roughness.png')
    print("Saved textures/brick_roughness.png")
    
    # 3. Normal Map (Beveled edges)
    normal = np.zeros((height, width, 3), dtype=np.uint8)
    # Default normal (0, 0, 1) -> (128, 128, 255)
    normal[:, :] = [128, 128, 255]
    
    # Simple bevel logic: if near edge of brick, tilt normal
    for y in range(height):
        row = y // (brick_h + mortar_size)
        offset = 0 if row % 2 == 0 else brick_w // 2
        
        for x in range(width):
            in_mortar_y = (y % (brick_h + mortar_size)) >= brick_h
            tx = (x + offset) % (brick_w + mortar_size)
            in_mortar_x = tx >= brick_w
            
            if not (in_mortar_y or in_mortar_x):
                # Inside brick
                # Check distance to edge
                dy = y % (brick_h + mortar_size)
                dx = tx
                
                nx, ny, nz = 0.0, 0.0, 1.0
                
                bevel = 4
                if dx < bevel: nx = -0.5
                elif dx > brick_w - bevel: nx = 0.5
                
                if dy < bevel: ny = -0.5
                elif dy > brick_h - bevel: ny = 0.5
                
                # Normalize
                l = np.sqrt(nx*nx + ny*ny + nz*nz)
                nx, ny, nz = nx/l, ny/l, nz/l
                
                # Map -1..1 to 0..255
                normal[y, x, 0] = int((nx * 0.5 + 0.5) * 255)
                normal[y, x, 1] = int((ny * 0.5 + 0.5) * 255)
                normal[y, x, 2] = int((nz * 0.5 + 0.5) * 255)
                
    Image.fromarray(normal).save('textures/brick_normal.png')
    print("Saved textures/brick_normal.png")

if __name__ == "__main__":
    generate_brick_textures()
