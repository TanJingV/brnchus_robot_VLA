"""Stage only reviewed public project figures and centerline exports."""
import csv
import json
import shutil
from pathlib import Path

site = Path(__file__).resolve().parent
root = site.parents[1]
assets = site / 'assets'
assets.mkdir(exist_ok=True)
source = root / 'exports/terminal_branches'
shutil.copy2(source / 'terminal_branch_trajectories_65.csv', assets / 'trajectories.csv')
shutil.copy2(source / 'terminal_branch_summary_65.csv', assets / 'summary.csv')
shutil.copy2(root / 'docs/figures/tdcr_mujoco_modeling/tdcr_mujoco_modeling_schematic.png', assets / 'model.png')
branches = {}
with (assets / 'trajectories.csv').open(encoding='utf-8-sig', newline='') as stream:
    for row in csv.DictReader(stream):
        branch = branches.setdefault(row['BranchID'], {
            'id': row['BranchID'], 'length': float(row['TotalPathLength_mm']), 'points': [],
        })
        branch['points'].append([float(row[f'{axis}_mm']) for axis in 'XYZ'])
assert len(branches) == 65
(assets / 'branches.json').write_text(json.dumps(list(branches.values()), separators=(',', ':')), encoding='utf-8')
print(f'Built {len(branches)} branches in {assets}')
