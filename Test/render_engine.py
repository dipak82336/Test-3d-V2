import sys
import os

# Add Library to Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kalpana3d import render_yaml

def main():
    # Configuration
    yaml_file = '../example/robot.yaml'
    output_file = '../gallery/robot_render.png'
    
    # Render
    render_yaml(yaml_file, output_file, width=800, height=600)

if __name__ == "__main__":
    main()
