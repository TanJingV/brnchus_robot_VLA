# Project website

**Canonical Project Page:** [https://tanjingv.github.io/brnchus_robot_VLA/#simulation](https://tanjingv.github.io/brnchus_robot_VLA/#simulation)

Static GitHub Pages website with a browser-native MuJoCo simulation. Serve this
folder from the repository root with:

```powershell
python docs/site/build_assets.py
python docs/site/build_web_sim.py
python -m http.server 8765 --directory docs/site
```

Open http://localhost:8765. The first visit downloads MuJoCo WebAssembly,
Three.js and the high-resolution airway STL. The live demo contains the complete
dual-segment robot, six tendon actuators, insertion actuator, dual compass
controls, tip camera, visual airway toggle, pause, and reset. The non-convex
airway collision boundary remains active when the visual surface is hidden.

The academic sections and figures are derived from the current BronchoTwin ICRA
manuscript. Re-export the reviewed paper figures whenever the manuscript changes.

Regenerate the staged assets whenever the project models change.
