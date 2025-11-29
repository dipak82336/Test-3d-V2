from kalpana3d import render_yaml
import os
import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: python render_preview.py <yaml_file> <output_image>")
        return

    yaml_file = sys.argv[1]
    output_image = sys.argv[2]

    # Very Low quality for speed as requested
    render_yaml(yaml_file, output_image, width=160, height=120)
    print(f"Rendered {output_image}")

if __name__ == "__main__":
    main()
