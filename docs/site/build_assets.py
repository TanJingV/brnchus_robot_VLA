"""Stage the reviewed public project figure used by the website."""
import shutil
from pathlib import Path

site = Path(__file__).resolve().parent
root = site.parents[1]
assets = site / 'assets'
assets.mkdir(exist_ok=True)
shutil.copy2(root / 'docs/figures/tdcr_mujoco_modeling/tdcr_mujoco_modeling_schematic.png', assets / 'model.png')
print(f'Updated website figure in {assets}')
