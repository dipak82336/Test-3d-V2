"""Quick single-scene render test"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kalpana3d.core.renderer import Renderer
from kalpana3d.utils.parser import parse_scene

def render_single_scene(scene_name):
    """Render only one scene for quick testing"""
    scene_path = Path(__file__).parent.parent / "scenes" / "Materials" / f"{scene_name}.yaml"
    output_dir = Path(__file__).parent.parent / "gallery" / "images" / "Materials"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Rendering: {scene_name}")
    scene_data = parse_scene(str(scene_path))
    
    renderer = Renderer(width=512, height=512)
    
    # Main view only
    img = renderer.render(scene_data)
    img.save(output_dir / f"{scene_name}_main.png")
    print(f"  ✓ Saved: {scene_name}_main.png")

if __name__ == "__main__":
    scene_name = sys.argv[1] if len(sys.argv) > 1 else "neon_tube"
    render_single_scene(scene_name)
