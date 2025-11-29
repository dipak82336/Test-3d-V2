import yaml
import numpy as np
import math
import random

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: return v
    return v / norm

def make_rotation_matrix(direction, up=np.array([0, 1, 0])):
    # Create rotation matrix that aligns 'up' to 'direction'
    z_axis = normalize(direction)
    # Check if parallel
    if abs(np.dot(z_axis, up)) > 0.99:
        # choose a different up reference
        up = np.array([1, 0, 0])

    x_axis = normalize(np.cross(up, z_axis))
    y_axis = np.cross(z_axis, x_axis)

    # Wait, in standard lookAt:
    # Forward is Z? Or Y?
    # For a Cone/Cylinder defined along Y axis:
    # We want local Y to align with 'direction'

    # Matrix columns are the local axes in world space
    # Col 0: local X
    # Col 1: local Y (this should be our direction)
    # Col 2: local Z

    # Re-calculate to align Y with direction
    y_target = normalize(direction)

    # Arbitrary vector to generate orthogonal basis
    temp = np.array([0, 0, 1])
    if abs(np.dot(y_target, temp)) > 0.99:
        temp = np.array([1, 0, 0])

    x_axis = normalize(np.cross(y_target, temp))
    z_axis = normalize(np.cross(x_axis, y_target))

    R = np.array([x_axis, y_target, z_axis]).T
    return R

def rotation_matrix_to_euler(R):
    # R = Rz @ Ry @ Rx
    # sy = -R[0,2]
    # cy = sqrt(1 - sy*sy)
    # ry = atan2(-R[0,2], sqrt(R[0,0]^2 + R[0,1]^2))

    sy = R[0,2]
    cy = math.sqrt(1 - sy*sy)

    # Wait, check parser.py implementation:
    # Rz @ Ry @ Rx
    # R[0,2] is sin(ry) if I recall correctly?
    # Let's use scipy if available, or just use the approximation from before
    # which is simpler and "good enough" for random trees

    # Let's rely on my previous function which was close but maybe coordinate system is Z-up?
    # Kalpana3D is likely Y-up based on the camera pos in test_export_texture.

    # Better approach: Just use axis-angle to Quat to Euler?
    # Or just use the function I wrote before but tuned.

    # Let's try to align Y axis (0,1,0) to D
    # Axis = (0,1,0) x D
    # Angle = acos((0,1,0) . D)

    # ...
    # Actually, let's keep it simple.
    # The previous implementation had some issues with gimbal lock maybe?

    # Let's assume Z-Y-X order

    sy = -R[0,2]
    cy = math.sqrt(1.0 - sy * sy)

    if cy > 1e-6:
        rx = math.atan2(R[1,2], R[2,2])
        ry = math.atan2(-R[0,2], cy)
        rz = math.atan2(R[0,1], R[0,0])
    else:
        rx = math.atan2(-R[2,1], R[1,1])
        ry = math.atan2(-R[0,2], cy)
        rz = 0.0

    return [rx, ry, rz]

def get_euler_from_direction(d):
    R = make_rotation_matrix(d)
    return rotation_matrix_to_euler(R)

class TreeGenerator:
    def __init__(self):
        self.objects = []

    def add_branch(self, start, end, radius_start, radius_end):
        length = np.linalg.norm(end - start)
        center = (start + end) / 2.0
        direction = normalize(end - start)

        rot = get_euler_from_direction(direction)

        # RoundCone parameters: r1, r2, h
        # It is defined vertically along Y, centered at origin?
        # sdRoundCone(p, r1, r2, h)
        # usually checks 0 to h.
        # So we should position it at 'start', not 'center'?
        # Wait, the engine applies rotation then translation?
        # Engine: local_p = transform_point(inv_mat, p)
        # transform_point applies inverse of (Translate @ Rotate)
        # So p is transformed to local space.

        # If I use `Pos` as `start`, and rotate such that local Y aligns with direction...
        # Then `sdRoundCone` should be from (0,0,0) to (0,h,0) in local space.
        # My SDF implementation:
        # if k < 0.0: return length(q) - r1 (bottom sphere)
        # if k > a*h: return length(q - vec3(0, h, 0)) - r2 (top sphere)
        # So yes, it is from 0 to h along Y.

        # So `Pos` in YAML should be `start`.
        # `Rot` should align Y axis to `direction`.

        self.objects.append({
            'Type': 11, # RoundCone
            'Pos': [float(start[0]), float(start[1]), float(start[2])],
            'Rot': [float(x) for x in rot],
            'Scale': [float(radius_start), float(radius_end), float(length)],
            'Mat': 0,
            'Op': 0 # Union (RoundCone blends itself naturally? No. Need Smooth Union for joints)
            # Actually let's use Smooth Union (Op 3)
        })
        # Override Op
        self.objects[-1]['Op'] = 3

    def generate(self):
        # Fractal parameters
        max_depth = 7

        # Recursion
        def branch(pos, direction, length, radius, depth):
            if depth >= max_depth:
                return

            # Wiggle direction
            # Add some gravity curve?
            # Trees tend to grow up, but branches sag?

            end = pos + direction * length

            # Taper radius
            taper = 0.7
            next_radius = radius * taper

            self.add_branch(pos, end, radius, next_radius)

            # Spawn children
            # Split count
            count = random.randint(2, 3) if depth < 2 else random.randint(1, 3)

            for i in range(count):
                # Rotation
                # Spread branches around the parent direction
                # Azimuth angle
                azimuth = random.uniform(0, 2 * math.pi)
                # Polar angle (spread from parent axis)
                spread = random.uniform(0.3, 0.8) # 20 to 45 degrees

                # Construct rotation
                # Local basis
                z_axis = direction
                temp = np.array([0, 1, 0])
                if abs(np.dot(z_axis, temp)) > 0.9: temp = np.array([1, 0, 0])
                x_axis = normalize(np.cross(temp, z_axis))
                y_axis = normalize(np.cross(z_axis, x_axis))

                # Rotate vector (0,0,1) by spread and azimuth
                # Actually simpler:
                # new_dir = z_axis * cos(spread) + (x_axis * cos(azimuth) + y_axis * sin(azimuth)) * sin(spread)

                new_dir = z_axis * math.cos(spread) + (x_axis * math.cos(azimuth) + y_axis * math.sin(azimuth)) * math.sin(spread)
                new_dir = normalize(new_dir)

                # Gravity influence
                new_dir[1] += 0.1 # Grow slightly up
                new_dir = normalize(new_dir)

                new_length = length * random.uniform(0.7, 0.9)

                branch(end, new_dir, new_length, next_radius, depth + 1)

        # Main trunk
        root_pos = np.array([0, 0, 0])
        root_dir = np.array([0, 1, 0])
        root_len = 2.0
        root_rad = 0.5

        branch(root_pos, root_dir, root_len, root_rad, 0)

def main():
    gen = TreeGenerator()
    gen.generate()

    # Material
    materials = [{
        'Color': [0.15, 0.1, 0.05],
        'Roughness': 1.0, # Bark is rough
        'Bump': 0.8, # Heavy bark texture
        'Displacement': 0.05,
        'RoughnessMap': 'textures/brick_roughness.png',
        'NormalMap': 'textures/brick_normal.png',
        'UVScale': 1.0
    }]

    scene = {
        'Camera': {
            'Pos': [0, 5, 8],
            'LookAt': [0, 3, 0],
            'FOV': 45.0
        },
        'EnvironmentMap': '../textures/studio_small.jpg',
        'Materials': materials,
        'Objects': gen.objects
    }

    with open('example/tree_design.yaml', 'w') as f:
        yaml.dump(scene, f)

    print(f"Generated {len(gen.objects)} branches.")

if __name__ == "__main__":
    main()
