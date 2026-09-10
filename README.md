# Bronchus Robot

A two-section tendon-driven continuum bronchoscope research platform with
MuJoCo simulation, manual control, centerline navigation and synchronized
multimodal recording.

**Project website:** https://tanjingv.github.io/brnchus_robot_VLA/

## Research status

This project supports an unpublished ICRA-oriented manuscript. The formal
paper title, authors, abstract and publication link are pending confirmation.
The website describes implemented capabilities and model geometry; it does
not claim validated clinical performance or a published navigation benchmark.

## Latest features

- Two active continuum sections, six tendons, a passive insertion carrier and
  a bronchial environment in MuJoCo.
- Compass/gamepad steering, insertion controls, robot initial-pose adjustment,
  passive-joint debug locking and a rigid base-guide constraint.
- Real/simulation environment selection, tip-camera and virtual-camera views.
- Synchronized camera, MuJoCo and control-panel recordings, with per-frame
  timing and tip, middle-platform and master/slave-connection poses.
- D435 capture and seven-marker 3D fusion modules in `Visual_information`.

## Run

The desktop application is developed on Windows. In the configured
`Bronchoscope` Conda environment:

```powershell
conda activate Bronchoscope
python -m UI
```

The launcher resolves the application under `window/`. Select simulation in
the startup dialog to use the MuJoCo environment. Hardware mode additionally
requires the device drivers, NDI tracker dependencies and Trio controller API.

The existing research environment is not yet packaged as a portable installer.
See [model documentation](two_segment_tdcr_opencr/README.md),
[visual processing](Visual_information/README.md), and
[capture dependencies](Visual_information/requirements-d435.txt) for module
requirements. Optional third-party checkpoints are obtained separately; see
[model weights](Visual_information/models/README.md).

## Repository map

| Directory | Purpose |
| --- | --- |
| `UI/`, `window/` | Application entry points and desktop workbench |
| `two_segment_tdcr_opencr/` | Parametric model generation and controllers |
| `meshes/`, `urdf/` | Robot and airway model assets |
| `agent_nav/` | Navigation and learning utilities |
| `Visual_information/` | Capture, segmentation and 3D tracking |
| `docs/site/` | Static project website and interactive MuJoCo simulation |

Local recordings, environments and Python caches are excluded from the source
release. Existing large legacy segmentation weights are retained at
`Visual_information/models/legacy_unet/1205weights_49.pth`.

## Website

```powershell
python docs/site/build_web_sim.py
python docs/site/build_assets.py
python -m http.server 8765 --directory docs/site
```

Open http://localhost:8765. The first visit downloads MuJoCo WebAssembly,
Three.js and the high-resolution lung STL. The simulation-only web build
contains the project two-section tendon model, dual compass controls, insertion
actuator, tip camera, lung visibility control, pause and reset. Hardware drivers
and real sensors remain in the desktop application. GitHub Pages deploys only
`docs/site` through `.github/workflows/project-pages.yml`.

## Acknowledgements

The continuum discretization and spatial-tendon modeling approach is informed
by [OpenCR MuJoCo](https://github.com/ContinuumRoboticsLab/opencr-mujoco).
Third-party models, libraries and hardware SDKs retain their respective terms.
