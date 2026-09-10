import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import loadMujoco from "https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js";

const $ = (selector) => document.querySelector(selector);
const ASSET_VERSION = "20";
const MODEL_URL = `sim/bronchoscope_web.xml?v=${ASSET_VERSION}`;
const LUNG_URL = `sim/part/bronchus.stl?v=${ASSET_VERSION}`;
const MODEL_ASSETS = [
  "base_link.STL",
  "slid_base.STL",
  "slid_M.STL",
  "lian.STL",
  "qudong1.STL",
  "qudong2.STL",
  "part/bronchus_collision_solid_nonconvex.stl",
];
const MAX_BEND_RAD = THREE.MathUtils.degToRad(160);
const TENDON_RADIUS = 0.00154;
const INSERTION_LIMIT_M = 0.577;
const INSERTION_RATE_MPS = 0.20;
const FREE_INSERTION_LEAD_M = 0.008;
const CONTACT_INSERTION_RATE_MPS = 0.20;
const CONTACT_INSERTION_LEAD_M = 0.0015;
const FREE_VELOCITY_DAMPING_PER_S = 12;
const CONTACT_VELOCITY_DAMPING_PER_S = 55;
const MAIN_RENDER_INTERVAL_MS = 1000 / 30;
const TIP_RENDER_INTERVAL_MS = 1000 / 15;
const CONTROL_SLEEP_DELAY_MS = 700;
const FREE_SLEEP_TIMEOUT_MS = 2500;
const CONTACT_STALL_SLEEP_MS = 450;
const INSERTION_PROGRESS_EPSILON_M = 0.00015;
const MOTOR_HISTORY_SECONDS = 4;
const MOTOR_SAMPLE_INTERVAL_MS = 40;
const MOTOR_COLORS = ["#e85bbd", "#ff8a55", "#f2c14e", "#5ed39a", "#4ecdc4", "#5c91ff", "#b07cff"];
const PASSIVE_GUIDE_FRONT_M = 0.5690000348619164;
const PASSIVE_GUIDE_OFFSETS_M = [
  -0.0795333, -0.0578666, -0.0361999, -0.0145332, 0.0071335,
  0.0288002, 0.0504669, 0.0721336, 0.0938003, 0.1154670,
  0.1371337, 0.1588004, 0.1804671, 0.2021338, 0.2238004,
  0.2454671, 0.2671338, 0.2888004, 0.3104671, 0.3321338,
  0.3538005, 0.3754671, 0.3971338, 0.4188004, 0.4404671,
  0.4621337, 0.4838003, 0.5054670, 0.5271336,
];
const PASSIVE_FOLLOW_WEIGHTS = [0.10, 0.13, 0.16, 0.18, 0.20, 0.23];
const PASSIVE_FOLLOW_RATIO = 0.55;
const PASSIVE_FOLLOW_STIFFNESS = 18.0;
const PASSIVE_FOLLOW_DAMPING = 0.35;
const PASSIVE_FOLLOW_MAX_TORQUE = 2.5;
const PASSIVE_FOLLOW_FILTER_S = 0.025;
const PROXIMAL_WIRE_ANGLES = [0, 120, 240].map(THREE.MathUtils.degToRad);
const DISTAL_WIRE_ANGLES = [60, 180, 300].map(THREE.MathUtils.degToRad);

const state = {
  ready: false,
  paused: false,
  lungVisible: true,
  followTip: false,
  proximal: { x: 0, y: 0 },
  distal: { x: 0, y: 0 },
  insertionTarget: 0,
  insertionCommand: 0,
  passiveDebugLocked: false,
  physicsSleeping: false,
  lastControlChange: 0,
  lastInsertionProgress: 0,
  lastInsertionPosition: 0,
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
let interfaceSiteId = -1;
let insertionQposAddress = -1;
let insertionDofAddress = -1;
let actuatorIds = [];
let actuatorBaselines = [];
let distalFeedback = new THREE.Vector2();
let passiveGuideJoints = [];
let passiveFollowerJoints = [];
let activeBaseBodyId = -1;
let proximalEndBodyId = -1;
let filteredActiveRotation = new THREE.Vector3();
let lungFlexId = -1;
let lungCollisionMasks = null;
let lastFrame = performance.now();
let telemetryDeadline = 0;
let motorSampleDeadline = 0;
let mainRenderDeadline = 0;
let tipRenderDeadline = 0;
let motorHistory = [];
let nonAirwayContactBaseline = 0;

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
  ["#insertion", "#pause-sim", "#reset-sim", "#toggle-lung", "#toggle-passive-lock", "#focus-tip"].forEach((selector) => {
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

function compiledMeshGeometry(meshId) {
  const vertexAddress = model.mesh_vertadr[meshId];
  const vertexCount = model.mesh_vertnum[meshId];
  const faceAddress = model.mesh_faceadr[meshId];
  const faceCount = model.mesh_facenum[meshId];
  const positions = new Float32Array(
    model.mesh_vert.slice(vertexAddress * 3, (vertexAddress + vertexCount) * 3),
  );
  const indices = new Uint32Array(
    model.mesh_face.slice(faceAddress * 3, (faceAddress + faceCount) * 3),
  );
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  return geometry;
}

class MuJoCoGeometryView {
  constructor(targetScene) {
    this.items = [];
    const meshCache = new Map();
    const meshType = mujoco.mjtObj.mjOBJ_MESH.value;
    for (let geomId = 0; geomId < model.ngeom; geomId += 1) {
      const rgba = Array.from(model.geom_rgba.slice(geomId * 4, geomId * 4 + 4));
      if (rgba[3] < 0.025 || model.geom_group[geomId] >= 3) continue;
      const size = Array.from(model.geom_size.slice(geomId * 3, geomId * 3 + 3));
      const geomType = model.geom_type[geomId];
      let geometry = geometryFor(geomType, size);
      if (geomType === 7) {
        const meshId = model.geom_dataid[geomId];
        const meshName = mujoco.mj_id2name(model, meshType, meshId);
        if (meshName === "visual_mesh") continue;
        if (!meshCache.has(meshId)) meshCache.set(meshId, compiledMeshGeometry(meshId));
        geometry = meshCache.get(meshId);
      }
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
  camera.position.set(-0.04, -0.72, 1.28);
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
  orbit.target.set(-0.05, 0.005, 1.075);
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
  setLoad(68, "Loading airway environment", "Reading the high-resolution bronchial STL");
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
    wakePhysics();
    const magnitude = Math.hypot(x, y);
    const scale = magnitude > 1 ? 1 / magnitude : 1;
    this.vector.x = x * scale;
    this.vector.y = y * scale;
    const distance = Math.min(this.pad.clientWidth * 0.39, this.pad.clientWidth * 0.39 * Math.hypot(this.vector.x, this.vector.y));
    const direction = Math.atan2(this.vector.y, this.vector.x);
    this.knob.style.transform = `translate(calc(-50% + ${Math.cos(direction) * distance}px), calc(-50% + ${Math.sin(direction) * distance}px))`;

    const bend = THREE.MathUtils.radToDeg(0.5 * MAX_BEND_RAD * Math.hypot(this.vector.x, this.vector.y) ** 2);
    this.angleOutput.textContent = `${bend.toFixed(1)} deg`;
    this.directionOutput.textContent = bend < 0.1 ? "CENTER" : `${((THREE.MathUtils.radToDeg(-direction) + 360) % 360).toFixed(0)} deg`;
  }
}

const proximalCompass = new CompassControl("#proximal-pad", "proximal", "#proximal-angle", "#proximal-direction");
const distalCompass = new CompassControl("#distal-pad", "distal", "#distal-angle", "#distal-direction");

function wakePhysics() {
  state.physicsSleeping = false;
  state.lastControlChange = performance.now();
  if (state.ready && !state.paused) setRuntime("Running live", "ready");
}

function sleepPhysics(actualInsertion) {
  state.physicsSleeping = true;
  state.insertionCommand = actualInsertion;
  data.qvel.fill(0);
  data.qacc.fill(0);
  data.qacc_warmstart.fill(0);
  applyControls();
  mujoco.mj_forward(model, data);
  setRuntime("Settled", "ready");
}

function actuatorId(name) {
  const id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR.value, name);
  if (id < 0) throw new Error(`Actuator not found: ${name}`);
  return id;
}

function motorVector(vector) {
  const result = new THREE.Vector2(vector.x, vector.y);
  const magnitude = Math.min(1, result.length());
  if (result.length() > 1) result.normalize();
  result.multiplyScalar(0.5 * magnitude);
  result.x *= -1;
  result.y *= -1;
  return result;
}

function setTendonTriplet(firstActuator, vector, wireAngles) {
  const magnitude = vector.length();
  const direction = magnitude > 1e-9 ? Math.atan2(vector.y, vector.x) : 0;
  const bend = MAX_BEND_RAD * magnitude;
  wireAngles.forEach((wireAngle, index) => {
    const actuatorIndex = firstActuator + index;
    const actuator = actuatorIds[actuatorIndex];
    const lower = model.actuator_ctrlrange[actuator * 2];
    const upper = model.actuator_ctrlrange[actuator * 2 + 1];
    const target = actuatorBaselines[actuatorIndex]
      - TENDON_RADIUS * bend * Math.cos(wireAngle - direction);
    data.ctrl[actuator] = THREE.MathUtils.clamp(target, lower, upper);
  });
}

function relativeRotationVector(firstSiteId, secondSiteId) {
  const firstOffset = firstSiteId * 9;
  const secondOffset = secondSiteId * 9;
  const first = data.site_xmat.slice(firstOffset, firstOffset + 9);
  const second = data.site_xmat.slice(secondOffset, secondOffset + 9);
  const relative = new Float64Array(9);
  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      for (let index = 0; index < 3; index += 1) {
        relative[row * 3 + column] += first[index * 3 + row]
          * second[index * 3 + column];
      }
    }
  }
  const cosine = THREE.MathUtils.clamp(
    (relative[0] + relative[4] + relative[8] - 1) / 2,
    -1,
    1,
  );
  const angle = Math.acos(cosine);
  if (angle < 1e-8) return new THREE.Vector3();
  const scale = angle / Math.max(2 * Math.sin(angle), 1e-8);
  return new THREE.Vector3(
    (relative[7] - relative[5]) * scale,
    (relative[2] - relative[6]) * scale,
    (relative[3] - relative[1]) * scale,
  );
}

function bodyRelativeRotationVector(firstBodyId, secondBodyId) {
  const firstOffset = firstBodyId * 9;
  const secondOffset = secondBodyId * 9;
  const first = data.xmat.slice(firstOffset, firstOffset + 9);
  const second = data.xmat.slice(secondOffset, secondOffset + 9);
  const relative = new Float64Array(9);
  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      for (let index = 0; index < 3; index += 1) {
        relative[row * 3 + column] += first[index * 3 + row]
          * second[index * 3 + column];
      }
    }
  }
  const cosine = THREE.MathUtils.clamp(
    (relative[0] + relative[4] + relative[8] - 1) / 2,
    -1,
    1,
  );
  const angle = Math.acos(cosine);
  if (angle < 1e-8) return new THREE.Vector3();
  const scale = angle / Math.max(2 * Math.sin(angle), 1e-8);
  return new THREE.Vector3(
    (relative[7] - relative[5]) * scale,
    (relative[2] - relative[6]) * scale,
    (relative[3] - relative[1]) * scale,
  );
}

function quaternionRotationVector(qposAddress) {
  let w = data.qpos[qposAddress];
  let x = data.qpos[qposAddress + 1];
  let y = data.qpos[qposAddress + 2];
  let z = data.qpos[qposAddress + 3];
  if (w < 0) {
    w = -w;
    x = -x;
    y = -y;
    z = -z;
  }
  const vectorNorm = Math.hypot(x, y, z);
  if (vectorNorm < 1e-10) return new THREE.Vector3();
  const angle = 2 * Math.atan2(vectorNorm, THREE.MathUtils.clamp(w, -1, 1));
  return new THREE.Vector3(x, y, z).multiplyScalar(angle / vectorNorm);
}

function applyControls() {
  if (!state.ready) return;
  const proximalMotor = motorVector(state.proximal);
  const distalDesired = motorVector(state.distal);
  const distalRotation = relativeRotationVector(interfaceSiteId, tipSiteId);
  const measuredDistal = new THREE.Vector2(
    0.00838 * distalRotation.y + 0.40178 * distalRotation.z,
    -0.40419 * distalRotation.y,
  );
  distalFeedback.addScaledVector(distalDesired.clone().sub(measuredDistal), 0.03);
  if (distalFeedback.length() > 0.3) distalFeedback.setLength(0.3);
  const distalMotor = new THREE.Vector2(
    0.94 * proximalMotor.x - 0.026 * proximalMotor.y,
    0.92 * proximalMotor.y,
  ).add(distalDesired).add(distalFeedback);
  if (distalMotor.length() > 2) distalMotor.setLength(2);

  setTendonTriplet(0, proximalMotor, PROXIMAL_WIRE_ANGLES);
  setTendonTriplet(3, distalMotor, DISTAL_WIRE_ANGLES);
  data.ctrl[actuatorIds[6]] = state.insertionCommand;
}

function resetSimulation() {
  if (!state.ready) return;
  mujoco.mj_resetDataKeyframe(model, data, 0);
  state.insertionTarget = 0;
  state.insertionCommand = 0;
  state.physicsSleeping = false;
  state.lastInsertionPosition = 0;
  state.lastInsertionProgress = performance.now();
  distalFeedback.set(0, 0);
  filteredActiveRotation.set(0, 0, 0);
  motorHistory = [];
  $("#insertion").value = "0";
  $("#insertion-value").textContent = "0.0 mm";
  proximalCompass.set(0, 0);
  distalCompass.set(0, 0);
  applyControls();
  enforcePassiveBaseGuide();
  mujoco.mj_forward(model, data);
}

function initializePassiveGuide() {
  const names = [
    ...Array.from({ length: 28 }, (_, index) => `cable_stiffJ_${index + 1}`),
    "cable_stiffJ_last",
  ];
  passiveGuideJoints = names.map((name, index) => {
    const jointId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, name);
    if (jointId < 0) throw new Error(`Passive joint missing from model: ${name}`);
    return {
      qpos: model.jnt_qposadr[jointId],
      dof: model.jnt_dofadr[jointId],
      offset: PASSIVE_GUIDE_OFFSETS_M[index],
    };
  });
}

function initializePassiveFollower() {
  const names = [
    ...Array.from({ length: 5 }, (_, index) => `cable_stiffJ_${index + 24}`),
    "cable_stiffJ_last",
  ];
  passiveFollowerJoints = names.map((name, index) => {
    const jointId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, name);
    if (jointId < 0) throw new Error(`Passive follower joint missing from model: ${name}`);
    return {
      qpos: model.jnt_qposadr[jointId],
      dof: model.jnt_dofadr[jointId],
      maximumAngle: model.jnt_range[jointId * 2 + 1],
      weight: PASSIVE_FOLLOW_WEIGHTS[index],
    };
  });
  activeBaseBodyId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, "active_tdcr_base");
  proximalEndBodyId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, "seg2_body");
  if (activeBaseBodyId < 0 || proximalEndBodyId < 0) {
    throw new Error("Active-segment reference bodies are missing from the model.");
  }
}

function applyPassiveFollower() {
  if (state.passiveDebugLocked) return;
  const activeRotation = bodyRelativeRotationVector(activeBaseBodyId, proximalEndBodyId);
  activeRotation.x = 0;
  const alpha = THREE.MathUtils.clamp(
    model.opt.timestep / (PASSIVE_FOLLOW_FILTER_S + model.opt.timestep),
    0,
    1,
  );
  filteredActiveRotation.lerp(activeRotation, alpha);

  passiveFollowerJoints.forEach(({ qpos, dof, maximumAngle, weight }) => {
    const desired = filteredActiveRotation.clone().multiplyScalar(PASSIVE_FOLLOW_RATIO * weight);
    if (maximumAngle > 0 && desired.length() > maximumAngle) desired.setLength(maximumAngle);
    const current = quaternionRotationVector(qpos);
    const torque = desired.sub(current).multiplyScalar(PASSIVE_FOLLOW_STIFFNESS);
    torque.x -= PASSIVE_FOLLOW_DAMPING * data.qvel[dof];
    torque.y -= PASSIVE_FOLLOW_DAMPING * data.qvel[dof + 1];
    torque.z -= PASSIVE_FOLLOW_DAMPING * data.qvel[dof + 2];
    if (torque.length() > PASSIVE_FOLLOW_MAX_TORQUE) torque.setLength(PASSIVE_FOLLOW_MAX_TORQUE);
    data.qfrc_applied[dof] += torque.x;
    data.qfrc_applied[dof + 1] += torque.y;
    data.qfrc_applied[dof + 2] += torque.z;
  });
}

function enforcePassiveBaseGuide() {
  if (insertionQposAddress < 0) return;
  const insertion = data.qpos[insertionQposAddress];
  passiveGuideJoints.forEach(({ qpos, dof, offset }) => {
    if (!state.passiveDebugLocked && offset + insertion > PASSIVE_GUIDE_FRONT_M) return;
    data.qpos[qpos] = 1;
    data.qpos[qpos + 1] = 0;
    data.qpos[qpos + 2] = 0;
    data.qpos[qpos + 3] = 0;
    for (let axis = 0; axis < 3; axis += 1) {
      data.qvel[dof + axis] = 0;
      data.qacc[dof + axis] = 0;
      data.qacc_warmstart[dof + axis] = 0;
      data.qfrc_applied[dof + axis] = 0;
    }
  });
}

function setLungEnabled(enabled) {
  state.lungVisible = enabled;
  lungGroup.visible = enabled;
  if (lungFlexId >= 0 && lungCollisionMasks) {
    model.flex_contype[lungFlexId] = enabled ? lungCollisionMasks.contype : 0;
    model.flex_conaffinity[lungFlexId] = enabled ? lungCollisionMasks.conaffinity : 0;
  }
  mujoco.mj_forward(model, data);
  $("#toggle-lung").textContent = `Airway model: ${enabled ? "on" : "off"}`;
  $("#toggle-lung").classList.toggle("active", enabled);
  $("#toggle-lung").setAttribute("aria-pressed", String(enabled));
  $("#boundary-status").classList.toggle("disabled", !enabled);
  $("#boundary-label").textContent = enabled
    ? "AIRWAY WALL CONSTRAINT ACTIVE"
    : "AIRWAY MODEL DISABLED";
  $("#airway-mode-note").innerHTML = enabled
    ? "<b>Airway boundary enforced</b><br>The visible airway and its reinforced collision boundary are active. Turning the airway off removes both."
    : "<b>Airway model disabled</b><br>Both the red surface and its collision boundary are removed, leaving the robot unobstructed for inspection.";
}

function setupControls() {
  $("#insertion").addEventListener("input", (event) => {
    wakePhysics();
    const millimetres = Number(event.target.value);
    state.insertionTarget = millimetres / 1000;
    $("#insertion-value").textContent = `${millimetres.toFixed(1)} mm`;
  });
  $("#pause-sim").addEventListener("click", () => {
    state.paused = !state.paused;
    $("#pause-sim").textContent = state.paused ? "Resume simulation" : "Pause simulation";
    setRuntime(state.paused ? "Paused" : "Running live", state.paused ? "" : "ready");
  });
  $("#reset-sim").addEventListener("click", resetSimulation);
  $("#toggle-lung").addEventListener("click", () => {
    wakePhysics();
    setLungEnabled(!state.lungVisible);
  });
  $("#toggle-passive-lock").addEventListener("click", () => {
    wakePhysics();
    state.passiveDebugLocked = !state.passiveDebugLocked;
    filteredActiveRotation.set(0, 0, 0);
    enforcePassiveBaseGuide();
    mujoco.mj_forward(model, data);
    $("#toggle-passive-lock").textContent = `Passive section: ${state.passiveDebugLocked ? "locked" : "follow"}`;
    $("#toggle-passive-lock").classList.toggle("active", state.passiveDebugLocked);
    $("#toggle-passive-lock").setAttribute("aria-pressed", String(state.passiveDebugLocked));
  });
  $("#focus-tip").addEventListener("click", () => {
    state.followTip = !state.followTip;
    $("#focus-tip").textContent = state.followTip ? "Stop following" : "Follow tip";
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

function airwayContactCount() {
  if (!state.lungVisible || lungFlexId < 0) return 0;
  return Math.max(0, data.ncon - nonAirwayContactBaseline);
}

function normalizedMotorSignals() {
  return actuatorIds.map((actuator, index) => {
    const lower = model.actuator_ctrlrange[actuator * 2];
    const upper = model.actuator_ctrlrange[actuator * 2 + 1];
    if (index < 6) {
      const baseline = actuatorBaselines[index];
      const span = Math.max(Math.abs(upper - baseline), Math.abs(lower - baseline), 1e-9);
      return THREE.MathUtils.clamp((data.ctrl[actuator] - baseline) / span, -1, 1);
    }
    return THREE.MathUtils.clamp((data.ctrl[actuator] - lower) / Math.max(upper - lower, 1e-9), 0, 1);
  });
}

function drawMotorChart(now) {
  const canvas = $("#motor-chart");
  const context = canvas.getContext("2d");
  const width = Math.max(1, canvas.clientWidth);
  const height = Math.max(1, canvas.clientHeight);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const pixelWidth = Math.round(width * dpr);
  const pixelHeight = Math.round(height * dpr);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);

  const left = 24;
  const right = width - 7;
  const top = 8;
  const bottom = height - 16;
  context.strokeStyle = "rgba(116, 139, 170, 0.18)";
  context.lineWidth = 1;
  [-1, -0.5, 0, 0.5, 1].forEach((value) => {
    const y = top + ((1 - value) / 2) * (bottom - top);
    context.beginPath();
    context.moveTo(left, y);
    context.lineTo(right, y);
    context.stroke();
  });
  context.fillStyle = "#687a93";
  context.font = '7px "SFMono-Regular", Consolas, monospace';
  context.textAlign = "right";
  context.fillText("+1", left - 4, top + 3);
  context.fillText("0", left - 4, (top + bottom) / 2 + 3);
  context.fillText("-1", left - 4, bottom + 3);

  if (now >= motorSampleDeadline) {
    motorSampleDeadline = now + MOTOR_SAMPLE_INTERVAL_MS;
    motorHistory.push({ timestamp: now / 1000, values: normalizedMotorSignals() });
  }
  const latestTime = now / 1000;
  const windowStart = latestTime - MOTOR_HISTORY_SECONDS;
  while (motorHistory.length && motorHistory[0].timestamp < windowStart) motorHistory.shift();

  MOTOR_COLORS.forEach((color, motorIndex) => {
    context.beginPath();
    context.strokeStyle = color;
    context.lineWidth = motorIndex === 6 ? 1.7 : 1.35;
    let started = false;
    motorHistory.forEach((sample) => {
      const x = left + ((sample.timestamp - windowStart) / MOTOR_HISTORY_SECONDS) * (right - left);
      const y = top + ((1 - sample.values[motorIndex]) / 2) * (bottom - top);
      if (!started) {
        context.moveTo(x, y);
        started = true;
      } else {
        context.lineTo(x, y);
      }
    });
    if (started) context.stroke();
  });
  context.fillStyle = "#586b84";
  context.textAlign = "left";
  context.fillText("-4 s", left, height - 4);
  context.textAlign = "right";
  context.fillText("now", right, height - 4);
}

function animate(now) {
  requestAnimationFrame(animate);
  if (!state.ready) return;
  const elapsed = Math.min(0.035, Math.max(0, (now - lastFrame) / 1000));
  lastFrame = now;
  if (!state.paused && !state.physicsSleeping) {
    const actualInsertion = insertionQposAddress >= 0
      ? data.qpos[insertionQposAddress]
      : state.insertionCommand;
    const inAirwayContact = airwayContactCount() > 0;
    const insertionRate = inAirwayContact ? CONTACT_INSERTION_RATE_MPS : INSERTION_RATE_MPS;
    const insertionLead = inAirwayContact ? CONTACT_INSERTION_LEAD_M : FREE_INSERTION_LEAD_M;
    const maximumStep = insertionRate * elapsed;
    state.insertionCommand += THREE.MathUtils.clamp(
      state.insertionTarget - state.insertionCommand,
      -maximumStep,
      maximumStep,
    );
    state.insertionCommand = THREE.MathUtils.clamp(
      state.insertionCommand,
      Math.max(0, actualInsertion - insertionLead),
      Math.min(INSERTION_LIMIT_M, actualInsertion + insertionLead),
    );
    applyControls();
    const steps = Math.max(1, Math.min(24, Math.round(elapsed / model.opt.timestep)));
    for (let index = 0; index < steps; index += 1) {
      data.qfrc_applied.fill(0);
      enforcePassiveBaseGuide();
      applyPassiveFollower();
      mujoco.mj_step(model, data);
      enforcePassiveBaseGuide();
    }
    const dampingRate = inAirwayContact
      ? CONTACT_VELOCITY_DAMPING_PER_S
      : FREE_VELOCITY_DAMPING_PER_S;
    const damping = Math.exp(-dampingRate * elapsed);
    for (let index = 0; index < data.qvel.length; index += 1) {
      const insertionRebound = index === insertionDofAddress
        && data.qvel[index] < 0
        && state.insertionTarget > actualInsertion;
      if (index !== insertionDofAddress || insertionRebound) {
        data.qvel[index] *= damping;
        if (Math.abs(data.qvel[index]) < 1e-6) data.qvel[index] = 0;
      }
    }

    const actualAfterStep = insertionQposAddress >= 0
      ? data.qpos[insertionQposAddress]
      : state.insertionCommand;
    if (Math.abs(actualAfterStep - state.lastInsertionPosition) >= INSERTION_PROGRESS_EPSILON_M) {
      state.lastInsertionPosition = actualAfterStep;
      state.lastInsertionProgress = now;
    }
    const controlQuietFor = now - state.lastControlChange;
    const insertionSettled = Math.abs(state.insertionTarget - actualAfterStep) < 0.0005;
    const contactStalled = inAirwayContact
      && now - state.lastInsertionProgress >= CONTACT_STALL_SLEEP_MS;
    const freeTimedOut = !inAirwayContact
      && insertionSettled
      && controlQuietFor >= FREE_SLEEP_TIMEOUT_MS;
    if (controlQuietFor >= CONTROL_SLEEP_DELAY_MS && (contactStalled || freeTimedOut)) {
      sleepPhysics(actualAfterStep);
    }
  }
  const renderMain = now >= mainRenderDeadline;
  const renderTip = now >= tipRenderDeadline;
  if (renderMain || renderTip) {
    modelView.sync();
    tendonView.sync();
    updateTipCamera();
  }
  if (renderMain) {
    mainRenderDeadline = now + MAIN_RENDER_INTERVAL_MS;
    orbit.update();
    resizeRenderer(renderer, camera);
    renderer.render(scene, camera);
  }
  if (renderTip) {
    tipRenderDeadline = now + TIP_RENDER_INTERVAL_MS;
    resizeRenderer(tipRenderer, tipCamera);
    tipRenderer.render(scene, tipCamera);
  }
  updateTelemetry(now);
  drawMotorChart(now);
}

async function stageModelFiles() {
  try { mujoco.FS.mkdir("/working"); } catch (error) { /* Directory already exists. */ }
  try { mujoco.FS.mkdir("/working/part"); } catch (error) { /* Directory already exists. */ }
  const files = [
    { path: "bronchoscope_web.xml", url: MODEL_URL },
    ...MODEL_ASSETS.map((path) => ({ path, url: `sim/${path}?v=${ASSET_VERSION}` })),
  ];
  await Promise.all(files.map(async ({ path, url }) => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to download ${path} (HTTP ${response.status})`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    mujoco.FS.writeFile(`/working/${path}`, bytes);
  }));
}

async function initialize() {
  try {
    setControlsEnabled(false);
    setupScene();
    setLoad(12, "Loading physics engine", "Initializing MuJoCo WebAssembly");
    mujoco = await loadMujoco();

    setLoad(43, "Loading the complete robot", "Staging the base, insertion mechanism, passive conduit, active segments, and airway collision model");
    await stageModelFiles();
    model = mujoco.MjModel.loadFromXML("/working/bronchoscope_web.xml");
    data = new mujoco.MjData(model);
    mujoco.mj_resetDataKeyframe(model, data, 0);
    mujoco.mj_forward(model, data);
    state.lastControlChange = performance.now();
    state.lastInsertionProgress = state.lastControlChange;
    state.lastInsertionPosition = insertionQposAddress >= 0 ? data.qpos[insertionQposAddress] : 0;

    actuatorIds = ["act_t1", "act_t2", "act_t3", "act_t4", "act_t5", "act_t6", "act_slid_M"].map(actuatorId);
    actuatorBaselines = actuatorIds.slice(0, 6).map((id) => data.ctrl[id]);
    tipSiteId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, "tip_center");
    interfaceSiteId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, "interface_center");
    const insertionJointId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, "slid_M");
    insertionQposAddress = insertionJointId >= 0 ? model.jnt_qposadr[insertionJointId] : -1;
    insertionDofAddress = insertionJointId >= 0 ? model.jnt_dofadr[insertionJointId] : -1;
    initializePassiveGuide();
    initializePassiveFollower();
    lungFlexId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX.value, "bronchial_wall_nonconvex");
    if (lungFlexId < 0) throw new Error("The non-convex airway collision boundary is missing from the model.");
    lungCollisionMasks = {
      contype: model.flex_contype[lungFlexId],
      conaffinity: model.flex_conaffinity[lungFlexId],
    };
    model.flex_contype[lungFlexId] = 0;
    model.flex_conaffinity[lungFlexId] = 0;
    mujoco.mj_forward(model, data);
    nonAirwayContactBaseline = data.ncon;
    model.flex_contype[lungFlexId] = lungCollisionMasks.contype;
    model.flex_conaffinity[lungFlexId] = lungCollisionMasks.conaffinity;
    mujoco.mj_forward(model, data);
    if (tipSiteId < 0) throw new Error("The tip_center site is missing from the model.");
    if (interfaceSiteId < 0) throw new Error("The interface_center site is missing from the model.");

    modelView = new MuJoCoGeometryView(scene);
    tendonView = new TendonView(scene);
    await loadLung();

    setLoad(96, "Preparing controls", "Connecting the dual compasses, insertion axis, and tip camera");
    state.ready = true;
    applyControls();
    modelView.sync();
    tendonView.sync();
    updateTipCamera();

    setControlsEnabled(true);
    $("#load-panel").hidden = true;
    $("#load-panel").style.display = "none";
    setRuntime("Running live", "ready");
    lastFrame = performance.now();
  } catch (error) {
    console.error(error);
    $("#load-panel").hidden = true;
    $("#load-panel").style.display = "none";
    $("#scene-error").hidden = false;
    $("#error-message").textContent = error?.message || String(error);
    setRuntime("Startup failed", "error");
  }
}

requestAnimationFrame(animate);
initialize();
