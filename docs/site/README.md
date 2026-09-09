# Project website

Static GitHub Pages website with a browser-native MuJoCo simulation. Serve this
folder from the repository root with:

```powershell
python docs/site/build_assets.py
python docs/site/build_web_sim.py
python -m http.server 8765 --directory docs/site
```

Open http://localhost:8765. The first visit downloads MuJoCo WebAssembly,
Three.js and the high-resolution lung STL. The live demo contains the project
two-section tendon model, six tendon actuators, insertion actuator, dual compass
controls, tip camera, lung visibility control, pause and reset.

Regenerate the staged assets whenever the project models change.
