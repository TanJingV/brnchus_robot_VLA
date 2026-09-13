# Bronchus Robot

A research platform for a tendon-driven continuum bronchoscope. The repository
is organised as three standalone systems. Project code never crosses a system
boundary; recorded data is the only shared interface.

## Systems

| Directory | Purpose | Launch command |
| --- | --- | --- |
| `mujoco_desktop_system/` | MuJoCo model, desktop workbench, controllers and navigation | `python -m mujoco_desktop_system` |
| `d435_capture_system/` | Independent D435, side-camera, Trio and NDI acquisition | `python -m d435_capture_system` |
| `d435_seven_marker_system/` | Independent seven-marker reconstruction and training workstation | `python -m d435_seven_marker_system` |

Each system contains its own application entry point, source modules,
configuration, runtime resources, tests, dependency list and documentation.
The D435 capture system does not load MuJoCo, and the MuJoCo desktop does not
open or import the D435 application.

## Data interface

Generated and experimental data lives outside all three systems:

| Directory | Contents |
| --- | --- |
| `data/d435_sessions/` | Raw and synchronised D435 acquisition sessions |
| `data/seven_marker_outputs/` | Reconstructed seven-marker trajectories and videos |
| `data/mujoco_recordings/` | Desktop workbench and simulation recordings |
| `data/exports/` | Derived trajectory exports |

These directories are local experiment data and are ignored by Git. The
seven-marker system reads D435 sessions through the data directory rather than
importing acquisition-system code.

## Documentation and website

**Project Page:** [https://tanjingv.github.io/brnchus_robot/#simulation](https://tanjingv.github.io/brnchus_robot/#simulation)

System-specific instructions are stored in each system's `README.md`. The
static project website remains under `docs/site/` and can stage a browser model
from the standalone MuJoCo system:

```powershell
python docs/site/build_web_sim.py
python docs/site/build_assets.py
python -m http.server 8765 --directory docs/site
```

The continuum discretisation and spatial-tendon modelling approach is informed
by [OpenCR MuJoCo](https://github.com/ContinuumRoboticsLab/opencr-mujoco).
