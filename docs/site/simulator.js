import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import loadMujoco from "https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js";

const $ = (selector) => document.querySelector(selector);
const MODEL_URL = "sim/bronchoscope_web.xml";
const LUNG_URL = "sim/bronchus.stl";
const MAX_BEND_RAD = THREE.MathUtils.degToRad(80);
const TENDON_RADIUS = 0.00154;
const PROXIMAL_WIRE_ANGLES = [0, 120, 240].map(THREE.MathUtils.degToRad);
const DISTAL_WIRE_ANGLES = [60, 180, 300].map(THREE.MathUtils.degToRad);

const state = {
  ready: false,
  paused: false,
  lungVisible: true,
  followTip: false,
  proximal: { x: 0, y: 0 },
  distal: { x: 0, y: 0 },
  insertion: 0,
};

let mujoco;
let model;
let data;
let modelView;
let tendonView;
let lungGroup;
let scene;
let camera;
let tipCamera;
let renderer;
let tipRenderer;
let orbit;
let tipSiteId = -1;
let actuatorIds = [];
let actuatorBaselines = [];
let lastFrame = performance.now();
let telemetryDeadline = 0;

function setLoad(percent, title, detail) {
  $("#load-progress").style.width = `${percent}%`;
  $("#load-title").textContent = title;
  $("#load-detail").textContent = detail;
}

function setRuntime(label, type = "") {
  $("#runtime-state").textContent = label;
  $("#runtime-dot").className = type;
}

function setControlsEnabled(enabled) {
  ["#insertion", "#pause-sim", "#reset-sim", "#toggle-lung", "#focus-tip"].forEach((selector) => {
    $(selector).disabled = !enabled;
  });
  ["#proximal-pad", "#distal-pad"].forEach((selector) => {
    $(selector).setAttribute("aria-disabled", String(!enabled));
  });
}

function geometryFor(type, size) {
  let geometry;
  let rotateToZ = false;
  if (type === 2) {
    geometry = new THREE.SphereGeometry(size[0], 18, 12);
  } else if (type === 3) {
    geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2, 6, 12);
    rotateToZ = true;
  } else if (type === 4) {
    geometry = new THREE.SphereGeometry(1, 18, 12);
    geometry.scale(size[0], size[1], size[2]);
  } else if (type === 5) {
    geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2, 18);
    rotateToZ = true;
  } else if (type === 6) {
    geometry = new THREE.BoxGeometry(size[0] * 2, size[1] * 2, size[2] * 2);
  } else {
    return null;
  }
  if (rotateToZ) geometry.rotateX(Math.PI / 2);
  return geometry;
}

class MuJoCoGeometryView {
  constructor(targetScene) {
    this.items = [];
    for (let geomId = 0; geomId < model.ngeom; geomId += 1) {
      const rgba = Array.from(model.geom_rgba.slice(geomId * 4, geomId * 4 + 4));
      if (rgba[3] < 0.025 || model.geom_group[geomId] >= 3) continue;
      const size = Array.from(model.geom_size.slice(geomId * 3, geomId * 3 + 3));
      const geometry = geometryFor(model.geom_type[geomId], size);
      if (!geometry) continue;

      const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
        roughness: 0.42,
        metalness: 0.05,
        transparent: rgba[3] < 0.99,
        opacity: rgba[3],
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geometry, material);
      const holder = new THREE.Group();
      holder.matrixAutoUpdate = false;
      holder.add(mesh);
      targetScene.add(holder);
      this.items.push({ geomId, holder });
    }
  }

  sync() {
    const matrix = new THREE.Matrix4();
    this.items.forEach(({ geomId, holder }) => {
      const p = geomId * 3;
      const r = geomId * 9;
      matrix.set(
        data.geom_xmat[r], data.geom_xmat[r + 1], data.geom_xmat[r + 2], data.geom_xpos[p],
        data.geom_xmat[r + 3], data.geom_xmat[r + 4], data.geom_xmat[r + 5], data.geom_xpos[p + 1],
        data.geom_xmat[r + 6], data.geom_xmat[r + 7], data.geom_xmat[r + 8], data.geom_xpos[p + 2],
        0, 0, 0, 1,
      );
      holder.matrix.copy(matrix);
    });
  }
}

class TendonView {
  constructor(targetScene) {
    const siteType = mujoco.mjtObj.mjOBJ_SITE.value;
    this.wires = Array.from({ length: 6 }, (_, index) => {
      const prefix = `wire_${index + 1}_`;
      const siteIds = [];
      for (let siteId = 0; siteId < model.nsite; siteId += 1) {
        const name = mujoco.mj_id2name(model, siteType, siteId);
        if (name?.startsWith(prefix)) siteIds.push(siteId);
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(siteIds.length * 3), 3));
      const color = index < 3 ? 0xed3ab7 : 0xff8a25;
      const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.92 }));
      line.frustumCulled = false;
      targetScene.add(line);
      return { siteIds, geometry };
    });
  }

  sync() {
    this.wires.forEach(({ siteIds, geometry }) => {
      const positions = geometry.attributes.position.array;
      siteIds.forEach((siteId, index) => {
        const source = siteId * 3;
        const target = index * 3;
        positions[target] = data.site_xpos[source];
        positions[target + 1] = data.site_xpos[source + 1];
        positions[target + 2] = data.site_xpos[source + 2];
      });
      geometry.attributes.position.needsUpdate = true;
      geometry.computeBoundingSphere();
    });
  }
}

function setupScene() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x152336);
  scene.fog = new THREE.FogExp2(0x152336, 1.4);

  camera = new THREE.PerspectiveCamera(42, 1, 0.0005, 8);
  camera.up.set(0, 0, 1);
  camera.position.set(0.43, -0.26, 1.22);
  tipCamera = new THREE.PerspectiveCamera(72, 1, 0.00035, 1.5);
  tipCamera.up.set(0, 0, 1);

  renderer = new THREE.WebGLRenderer({ canvas: $("#mujoco-canvas"), antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;

  tipRenderer = new THREE.WebGLRenderer({ canvas: $("#tip-canvas"), antialias: true, alpha: false });
  tipRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  tipRenderer.outputColorSpace = THREE.SRGBColorSpace;
  tipRenderer.toneMapping = THREE.ACESFilmicToneMapping;
  tipRenderer.toneMappingExposure = 1.25;

  orbit = new OrbitControls(camera, renderer.domElement);
  orbit.target.set(0.25, 0.005, 1.075);
  orbit.enableDamping = true;
  orbit.dampingFactor = 0.08;
  orbit.minDistance = 0.035;
  orbit.maxDistance = 1.2;

  scene.add(new THREE.HemisphereLight(0xb7d8ff, 0x172033, 1.7));
  const keyLight = new THREE.DirectionalLight(0xffffff, 2.8);
  keyLight.position.set(0.2, -0.25, 1.45);
  scene.add(keyLight);
  const rimLight = new THREE.DirectionalLight(0x76baff, 1.6);
  rimLight.position.set(0.1, 0.3, 1.1);
  scene.add(rimLight);

  const grid = new THREE.GridHelper(0.8, 32, 0x36506c, 0x263a52);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = 0.985;
  grid.material.transparent = true;
  grid.material.opacity = 0.45;
  scene.add(grid);
}

async function loadLung() {
  setLoad(68, "加载肺部环境", "正在读取高分辨率支气管 STL");
  const geometry = await new STLLoader().loadAsync(LUNG_URL, (event) => {
    if (!event.total) return;
    const progress = 68 + Math.round((event.loaded / event.total) * 18);
    $("#load-progress").style.width = `${progress}%`;
  });
  geometry.computeVertexNormals();
  const lung = new THREE.Mesh(
    geometry,
    new THREE.MeshPhysicalMaterial({
      color: 0xd94a5b,
      roughness: 0.58,
      transmission: 0.05,
      transparent: true,
      opacity: 0.17,
      side: THREE.DoubleSide,
      depthWrite: false,
    }),
  );
  lung.scale.setScalar(0.001);
  lung.position.set(0.0072, -0.004, 0);
  lungGroup = new THREE.Group();
  lungGroup.position.set(0.3138, 0.015, 1.054);
  lungGroup.rotation.z = -Math.PI / 2;
  lungGroup.add(lung);
  scene.add(lungGroup);
}

class CompassControl {
  constructor(pad, stateKey, angleOutput, directionOutput) {
    this.pad = $(pad);
    this.knob = this.pad.querySelector(".compass-knob");
    this.vector = state[stateKey];
    this.angleOutput = $(angleOutput);
    this.directionOutput = $(directionOutput);
    this.pointerId = null;

    this.pad.addEventListener("pointerdown", (event) => {
      if (!state.ready) return;
      this.pointerId = event.pointerId;
      this.pad.setPointerCapture(event.pointerId);
      this.updateFromPointer(event);
    });
    this.pad.addEventListener("pointermove", (event) => {
      if (event.pointerId === this.pointerId) this.updateFromPointer(event);
    });
    const release = (event) => {
      if (event.pointerId === this.pointerId) this.pointerId = null;
    };
    this.pad.addEventListener("pointerup", release);
    this.pad.addEventListener("pointercancel", release);
    this.pad.addEventListener("dblclick", () => this.set(0, 0));
    this.pad.addEventListener("keydown", (event) => {
      if (!state.ready) return;
      const increments = {
        ArrowLeft: [-0.08, 0],
        ArrowRight: [0.08, 0],
        ArrowUp: [0, -0.08],
        ArrowDown: [0, 0.08],
      };
      if (event.code === "Space") {
        event.preventDefault();
        this.set(0, 0);
      } else if (increments[event.key]) {
        event.preventDefault();
        const [dx, dy] = increments[event.key];
        this.set(this.vector.x + dx, this.vector.y + dy);
      }
    });
  }

  updateFromPointer(event) {
    const box = this.pad.getBoundingClientRect();
    const radius = box.width / 2;
    this.set((event.clientX - box.left - radius) / (radius * 0.78), (event.clientY - box.top - radius) / (radius * 0.78));
  }

  set(x, y) {
    const magnitude = Math.hypot(x, y);
    const scale = magnitude > 1 ? 1 / magnitude : 1;
    this.vector.x = x * scale;
    this.vector.y = y * scale;
    const distance = Math.min(this.pad.clientWidth * 0.39, this.pad.clientWidth * 0.39 * Math.hypot(this.vector.x, this.vector.y));
    const direction = Math.atan2(this.vector.y, this.vector.x);
    this.knob.style.transform = `translate(calc(-50% + ${Math.cos(direction) * distance}px), calc(-50% + ${Math.sin(direction) * distance}px))`;

    const bend = THREE.MathUtils.radToDeg(MAX_BEND_RAD * Math.hypot(this.vector.x, this.vector.y) ** 2);
    this.angleOutput.textContent = `${bend.toFixed(1)}°`;
    this.directionOutput.textContent = bend < 0.1 ? "CENTER" : `${((THREE.MathUtils.radToDeg(-direction) + 360) % 360).toFixed(0)}°`;
  }
}

const proximalCompass = new CompassControl("#proximal-pad", "proximal", "#proximal-angle", "#proximal-direction");
const distalCompass = new CompassControl("#distal-pad", "distal", "#distal-angle", "#distal-direction");

function actuatorId(name) {
  const id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR.value, name);
  if (id < 0) throw new Error(`Actuator not found: ${name}`);
  return id;
}

function bendCommand(vector) {
  const magnitude = Math.min(1, Math.hypot(vector.x, vector.y));
  return {
    bend: MAX_BEND_RAD * magnitude * magnitude,
    direction: Math.atan2(-vector.y, vector.x),
  };
}

function applyControls() {
  if (!state.ready) return;
  const proximal = bendCommand(state.proximal);
  const distal = bendCommand(state.distal);

  PROXIMAL_WIRE_ANGLES.forEach((wireAngle, index) => {
    data.ctrl[actuatorIds[index]] = actuatorBaselines[index]
      - TENDON_RADIUS * proximal.bend * Math.cos(wireAngle - proximal.direction);
  });
  DISTAL_WIRE_ANGLES.forEach((wireAngle, index) => {
    const actuatorIndex = index + 3;
    const proximalContribution = proximal.bend * Math.cos(wireAngle - proximal.direction);
    const distalContribution = distal.bend * Math.cos(wireAngle - distal.direction);
    data.ctrl[actuatorIds[actuatorIndex]] = actuatorBaselines[actuatorIndex]
      - TENDON_RADIUS * (proximalContribution + distalContribution);
  });
  data.ctrl[actuatorIds[6]] = state.insertion;
}

function resetSimulation() {
  if (!state.ready) return;
  mujoco.mj_resetDataKeyframe(model, data, 0);
  state.insertion = 0;
  $("#insertion").value = "0";
  $("#insertion-value").textContent = "0.0 mm";
  proximalCompass.set(0, 0);
  distalCompass.set(0, 0);
  applyControls();
  mujoco.mj_forward(model, data);
}

function setupControls() {
  $("#insertion").addEventListener("input", (event) => {
    const millimetres = Number(event.target.value);
    state.insertion = millimetres / 1000;
    $("#insertion-value").textContent = `${millimetres.toFixed(1)} mm`;
  });
  $("#pause-sim").addEventListener("click", () => {
    state.paused = !state.paused;
    $("#pause-sim").textContent = state.paused ? "继续仿真" : "暂停仿真";
    setRuntime(state.paused ? "已暂停" : "实时运行", state.paused ? "" : "ready");
  });
  $("#reset-sim").addEventListener("click", resetSimulation);
  $("#toggle-lung").addEventListener("click", () => {
    state.lungVisible = !state.lungVisible;
    lungGroup.visible = state.lungVisible;
    $("#toggle-lung").textContent = `肺部：${state.lungVisible ? "显示" : "隐藏"}`;
    $("#toggle-lung").classList.toggle("active", state.lungVisible);
    $("#toggle-lung").setAttribute("aria-pressed", String(state.lungVisible));
  });
  $("#focus-tip").addEventListener("click", () => {
    state.followTip = !state.followTip;
    $("#focus-tip").textContent = state.followTip ? "停止跟随" : "跟随末端";
    $("#focus-tip").classList.toggle("active", state.followTip);
  });
  $("#retry-sim").addEventListener("click", () => location.reload());
}

setupControls();

function resizeRenderer(targetRenderer, targetCamera) {
  const canvas = targetRenderer.domElement;
  const width = Math.max(1, canvas.clientWidth);
  const height = Math.max(1, canvas.clientHeight);
  const dpr = targetRenderer.getPixelRatio();
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    targetRenderer.setSize(width, height, false);
    targetCamera.aspect = width / height;
    targetCamera.updateProjectionMatrix();
  }
}

function updateTipCamera() {
  if (tipSiteId < 0) return;
  const p = tipSiteId * 3;
  const r = tipSiteId * 9;
  const position = new THREE.Vector3(data.site_xpos[p], data.site_xpos[p + 1], data.site_xpos[p + 2]);
  const forward = new THREE.Vector3(data.site_xmat[r], data.site_xmat[r + 3], data.site_xmat[r + 6]).normalize();
  const up = new THREE.Vector3(data.site_xmat[r + 2], data.site_xmat[r + 5], data.site_xmat[r + 8]).normalize();
  tipCamera.position.copy(position).addScaledVector(forward, 0.00035);
  tipCamera.up.copy(up);
  tipCamera.lookAt(position.clone().addScaledVector(forward, 0.1));
  if (state.followTip) {
    orbit.target.lerp(position, 0.12);
  }
}

function updateTelemetry(now) {
  if (now < telemetryDeadline || tipSiteId < 0) return;
  telemetryDeadline = now + 90;
  const offset = tipSiteId * 3;
  $("#sim-time").textContent = `${data.time.toFixed(2)} s`;
  $("#tip-x").textContent = `${(data.site_xpos[offset] * 1000).toFixed(1)} mm`;
  $("#tip-y").textContent = `${(data.site_xpos[offset + 1] * 1000).toFixed(1)} mm`;
  $("#tip-z").textContent = `${(data.site_xpos[offset + 2] * 1000).toFixed(1)} mm`;
  $("#contacts").textContent = String(data.ncon);
}

function animate(now) {
  requestAnimationFrame(animate);
  if (!state.ready) return;
  const elapsed = Math.min(0.035, Math.max(0, (now - lastFrame) / 1000));
  lastFrame = now;
  if (!state.paused) {
    applyControls();
    const steps = Math.max(1, Math.min(36, Math.round(elapsed / model.opt.timestep)));
    for (let index = 0; index < steps; index += 1) mujoco.mj_step(model, data);
  }
  modelView.sync();
  tendonView.sync();
  updateTipCamera();
  orbit.update();
  resizeRenderer(renderer, camera);
  resizeRenderer(tipRenderer, tipCamera);
  renderer.render(scene, camera);
  tipRenderer.render(scene, tipCamera);
  updateTelemetry(now);
}

async function fetchModel() {
  const response = await fetch(MODEL_URL);
  if (!response.ok) throw new Error(`MJCF 下载失败（HTTP ${response.status}）`);
  return new Uint8Array(await response.arrayBuffer());
}

async function initialize() {
  try {
    setControlsEnabled(false);
    setupScene();
    setLoad(12, "加载物理引擎", "正在初始化 MuJoCo WebAssembly");
    mujoco = await loadMujoco();

    setLoad(43, "读取机器人模型", "正在载入双段腱驱连续体 MJCF");
    const modelBytes = await fetchModel();
    try { mujoco.FS.mkdir("/working"); } catch (error) { /* Directory already exists. */ }
    mujoco.FS.writeFile("/working/bronchoscope_web.xml", modelBytes);
    model = mujoco.MjModel.loadFromXML("/working/bronchoscope_web.xml");
    data = new mujoco.MjData(model);
    mujoco.mj_resetDataKeyframe(model, data, 0);
    mujoco.mj_forward(model, data);

    actuatorIds = ["wire_1", "wire_2", "wire_3", "wire_4", "wire_5", "wire_6", "web_insertion_actuator"].map(actuatorId);
    actuatorBaselines = actuatorIds.slice(0, 6).map((id) => (
      (model.actuator_ctrlrange[id * 2] + model.actuator_ctrlrange[id * 2 + 1]) / 2
    ));
    tipSiteId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, "tip_center");
    if (tipSiteId < 0) throw new Error("模型中缺少 tip_center 站点");

    modelView = new MuJoCoGeometryView(scene);
    tendonView = new TendonView(scene);
    await loadLung();

    setLoad(96, "准备控制器", "正在连接双罗盘、插入轴与 tip camera");
    applyControls();
    modelView.sync();
    tendonView.sync();
    updateTipCamera();

    state.ready = true;
    setControlsEnabled(true);
    $("#load-panel").hidden = true;
    $("#load-panel").style.display = "none";
    setRuntime("实时运行", "ready");
    lastFrame = performance.now();
  } catch (error) {
    console.error(error);
    $("#load-panel").hidden = true;
    $("#load-panel").style.display = "none";
    $("#scene-error").hidden = false;
    $("#error-message").textContent = error?.message || String(error);
    setRuntime("启动失败", "error");
  }
}

requestAnimationFrame(animate);
initialize();
