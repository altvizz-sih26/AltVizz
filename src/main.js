import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import "./style.css";

// ===============================================================
// MODEL REGISTRY
// ===============================================================
// These three files were inspected directly (pygltflib + manual
// bounding-box / attribute analysis). Findings that drove every
// decision below:
//
//  - All three GLBs are authored Z-UP (elevation is stored on Z,
//    footprint is X/Y). Three.js is Y-up, so every model needs a
//    -90 deg rotation on X after load, BEFORE we measure it.
//  - None of the three meshes ship a NORMAL attribute (only
//    POSITION + TEXCOORD_0), so without an explicit
//    computeVertexNormals() call the surface lights unevenly or,
//    depending on renderer defaults, can end up fully unlit/black.
//  - Materials are alphaMode OPAQUE / doubleSided:false. Combined
//    with the wrong up-axis this is the main reason the page could
//    render nothing but the sky-blue clear color: the heightfield
//    was effectively standing on its edge, back-face culled away
//    from camera.
//  - Every mesh is a single-material 512x512 heightfield
//    (262,144 verts / 522,242 tris) with one baked 512x512 satellite
//    texture — no cameras, lights, animations or skins inside any
//    of the files, so nothing needs to be stripped out.
//
// Filename -> button mapping:
//   depthwizard_bare_terrain_3d.glb  -> Bare Terrain (name says so,
//         and it has the largest elevation range: -42.5..172.1)
//   depthwizard_urban_3d.glb         -> Terrain (its baked texture
//         shows open ground + built areas, elevation -10.4..43.9)
//   urban_depthwizard_3d.glb         -> Vegetation (its baked
//         texture is dominated by tree canopy, elevation -4.2..18.6,
//         the flattest/lowest-relief of the three)
// The "urban" pair can only be told apart by the baked texture
// content, not the filename, so if these two ever look swapped in
// the viewer, just swap the two paths below — nothing else needs
// to change.
// ===============================================================

const MODEL_CONFIG = [
  {
    key: "bare",
    label: "Bare Terrain",
    url: "/depthwizard_bare_terrain_3d.glb",
    buttonId: "btn-bare",
  },
  {
    key: "terrain",
    label: "Terrain",
    url: "/depthwizard_urban_3d.glb",
    buttonId: "btn-terrain",
  },
  {
    key: "vegetation",
    label: "Vegetation",
    url: "/urban_depthwizard_3d.glb",
    buttonId: "btn-vegetation",
  },
];

// Every one of these GLBs was authored Z-up. If a future model
// export is already Y-up, set its `upAxis` to "y" instead and it
// will be skipped during the correction rotation.
const SOURCE_UP_AXIS = "z";

// ===============================================================
// SCENE
// ===============================================================

const scene = new THREE.Scene();
// A slightly lighter sky than pure "sky blue" keeps the terrain reading
// with good contrast against the background without the background
// itself glaring or washing out at the horizon.
const SKY_COLOR = 0x9fd9f2;
scene.background = new THREE.Color(SKY_COLOR);
scene.fog = new THREE.Fog(SKY_COLOR, 1, 8000);

// ===============================================================
// CAMERA
// ===============================================================

const camera = new THREE.PerspectiveCamera(
  55,
  window.innerWidth / window.innerHeight,
  0.1,
  10000
);

camera.position.set(600, 500, 600);

// ===============================================================
// RENDERER
// ===============================================================

const canvasContainer = document.getElementById("viewer");

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  powerPreference: "high-performance",
});

renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
// A touch brighter than a "neutral" 1.0, but ACES rolls off highlights
// naturally so this brightens midtones/shadows without blowing out
// the bright parts of the baked satellite textures.
renderer.toneMappingExposure = 1.2;

renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

(canvasContainer || document.body).appendChild(renderer.domElement);

// ===============================================================
// LIGHTS
// ===============================================================
// Four-light "studio" rig instead of a single strong key light:
//   1. Ambient       - flat base brightness, lifts pure-black shadow
//                       cores so nothing on the terrain is ever unreadable.
//   2. Hemisphere    - sky/ground tint so slopes facing up vs. down
//                       aren't lit identically (keeps it looking natural,
//                       not flat).
//   3. Key (sun)      - the only shadow-casting light; toned down from
//                       before so cast shadows are soft, not black pits.
//   4. Fill           - dim, opposite side, no shadows. Its whole job is
//                       to knock the darkness out of the side of the
//                       terrain facing away from the key light.
// Total intensity is balanced so it reads as "bright and clear" without
// any single light being strong enough to overexpose the texture.

const ambientLight = new THREE.AmbientLight(0xffffff, 1.15);
scene.add(ambientLight);

const hemisphereLight = new THREE.HemisphereLight(0xeaf4ff, 0x8a7a63, 1.35);
scene.add(hemisphereLight);

const directionalLight = new THREE.DirectionalLight(0xfff6e6, 1.7);
directionalLight.position.set(500, 900, 350);
directionalLight.castShadow = true;

// Softer penumbra so shadow edges aren't a hard, dark line (works with
// PCFSoftShadowMap). Bias tuned to avoid both shadow acne and peter-panning
// on this heightfield's triangle density.
directionalLight.shadow.radius = 6;
directionalLight.shadow.blurSamples = 16;

// All three files share the same ~511x511 footprint, so a single
// generous, fixed shadow frustum comfortably covers every model.
const SHADOW_FRUSTUM = 420;
directionalLight.shadow.mapSize.set(2048, 2048);
directionalLight.shadow.camera.left = -SHADOW_FRUSTUM;
directionalLight.shadow.camera.right = SHADOW_FRUSTUM;
directionalLight.shadow.camera.top = SHADOW_FRUSTUM;
directionalLight.shadow.camera.bottom = -SHADOW_FRUSTUM;
directionalLight.shadow.camera.near = 10;
directionalLight.shadow.camera.far = 3000;
directionalLight.shadow.bias = -0.0004;

scene.add(directionalLight);
scene.add(directionalLight.target);

// Fill light: comes from roughly the opposite side of the key light,
// casts no shadows, and is deliberately dim. This is what removes the
// "very dark side" of the terrain without flattening the shading from
// the key light or casting a second set of shadows.
const fillLight = new THREE.DirectionalLight(0xdcebff, 0.6);
fillLight.position.set(-450, 500, -380);
fillLight.castShadow = false;
scene.add(fillLight);
scene.add(fillLight.target);

// ===============================================================
// GROUND-PLANE FOR SHADOWS
// (helps depth perception; disabled once a model is centered on
// it since the terrain itself IS the ground)
// ===============================================================

// ===============================================================
// CONTROLS
// ===============================================================

const controls = new OrbitControls(camera, renderer.domElement);

controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.screenSpacePanning = false;

controls.minDistance = 5;
controls.maxDistance = 5000;
controls.maxPolarAngle = Math.PI * 0.495; // stop just short of going underground

// ===============================================================
// GLTF LOADER
// ===============================================================

const loader = new GLTFLoader();

// key -> { root, size: Vector3, center: Vector3 }
const models = {};
let currentKey = null;

// ===============================================================
// UI ELEMENTS
// ===============================================================

const statusEl = document.getElementById("status-text");
const overlayEl = document.getElementById("loading-overlay");

function setStatus(text) {
  if (statusEl) statusEl.textContent = text;
  console.log("[STATUS]", text);
}

function setButtonState(key, state) {
  const cfg = MODEL_CONFIG.find((c) => c.key === key);
  if (!cfg) return;
  const btn = document.getElementById(cfg.buttonId);
  if (!btn) return;

  btn.classList.remove("is-loading", "is-ready", "is-error", "is-active");

  if (state === "loading") {
    btn.classList.add("is-loading");
    btn.disabled = true;
  } else if (state === "ready") {
    btn.classList.add("is-ready");
    btn.disabled = false;
  } else if (state === "error") {
    btn.classList.add("is-error");
    btn.disabled = true;
    btn.title = "This model failed to load — see console for details";
  }

  if (currentKey === key && state !== "error") {
    btn.classList.add("is-active");
  }
}

function markActiveButton(activeKey) {
  MODEL_CONFIG.forEach((cfg) => {
    const btn = document.getElementById(cfg.buttonId);
    if (!btn) return;
    btn.classList.toggle("is-active", cfg.key === activeKey);
  });
}

// ===============================================================
// PREPARE MODEL
// Fixes: axis correction, missing normals, backface culling,
// texture color space / anisotropy, and finally centers the model
// horizontally and rests it on the y=0 ground plane.
// ===============================================================

function prepareModel(root, label) {
  const maxAniso = renderer.capabilities.getMaxAnisotropy();

  root.traverse((object) => {
    if (!object.isMesh) return;

    object.castShadow = true;
    object.receiveShadow = true;

    // These GLBs ship POSITION + TEXCOORD_0 only. Without normals
    // the surface either shades flat/incorrectly or, on some
    // material/renderer combinations, doesn't receive directional
    // light at all. Compute them if missing rather than assuming.
    const geom = object.geometry;
    if (geom && !geom.attributes.normal) {
      geom.computeVertexNormals();
      console.log(`[${label}] normals were missing — computed them`);
    }

    const materials = Array.isArray(object.material)
      ? object.material
      : [object.material];

    materials.forEach((mat) => {
      if (!mat) return;

      // Source materials are doubleSided:false. After the axis
      // correction this is usually fine, but heightfields can still
      // show gaps at masked/degenerate triangles from the depth
      // source data, and the flythrough camera can end up briefly
      // below the surface. DoubleSide is the safe, cheap fix for a
      // single-material terrain mesh like this.
      mat.side = THREE.DoubleSide;

      // --- This is the real cause of the "too dark" terrain ---
      // All three GLBs were exported (by trimesh) with
      // metallicFactor: 1.0 and a baseColorFactor of [0.4,0.4,0.4,1].
      // A metalness-1 surface has essentially no diffuse response and
      // only shows up via specular reflection of an environment map —
      // which this scene doesn't have — so it reads as dark/flat
      // regardless of how many lights are added. The 0.4 factor then
      // multiplies the baked satellite texture down to 40% brightness
      // on top of that. Neither of these touches the actual texture
      // pixels or the mesh, so correcting them still fully preserves
      // the original GLB textures/colors — it just stops suppressing
      // them.
      if (mat.isMeshStandardMaterial || mat.metalness !== undefined) {
        mat.metalness = 0;
        // Keep roughness close to the authored value (matte satellite
        // imagery) rather than making it glossy.
        if (mat.roughness === undefined || mat.roughness > 0.95) {
          mat.roughness = 0.9;
        }
      }

      // Only neutralize a flat, uncolored gray dimming factor (this is
      // an export-time exposure knob, not deliberate color grading —
      // r === g === b and well below 1). If a future export ever ships
      // a genuine tint, this check leaves it alone.
      if (mat.color) {
        const { r, g, b } = mat.color;
        const isFlatGrayDimmer =
          Math.abs(r - g) < 0.01 && Math.abs(g - b) < 0.01 && r < 0.9;
        if (isFlatGrayDimmer) {
          mat.color.setRGB(1, 1, 1);
        }
      }

      if (mat.map) {
        mat.map.colorSpace = THREE.SRGBColorSpace;
        mat.map.anisotropy = maxAniso;
        mat.map.needsUpdate = true;
      }

      mat.needsUpdate = true;
    });
  });

  // --- Axis correction (Z-up authored -> Y-up three.js world) ---
  if (SOURCE_UP_AXIS === "z") {
    root.rotation.x = -Math.PI / 2;
  }

  root.updateMatrixWorld(true);

  // --- Measure in world space, after rotation ---
  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());

  // --- Center horizontally, rest vertically on the ground (y=0) ---
  root.position.x -= center.x;
  root.position.z -= center.z;
  root.position.y -= box.min.y;

  root.updateMatrixWorld(true);

  // Re-measure once more post-placement so downstream code (camera
  // framing, flythrough path) works off exact final coordinates.
  const finalBox = new THREE.Box3().setFromObject(root);
  const finalSize = finalBox.getSize(new THREE.Vector3());
  const finalCenter = finalBox.getCenter(new THREE.Vector3());

  console.log(`[${label}] size:`, finalSize, "center:", finalCenter);

  return { size: finalSize, center: finalCenter };
}

// ===============================================================
// FRAME CAMERA TO FIT A MODEL'S BOUNDING BOX
// ===============================================================

function frameModel(entry) {
  const { size, center } = entry;

  const maxDim = Math.max(size.x, size.y, size.z, 0.001);

  // Fit the model's bounding SPHERE inside the tighter of the
  // vertical/horizontal FOV. Unlike fitting the box's raw axes,
  // this stays correct regardless of which diagonal angle the
  // camera views it from, so the model is guaranteed to be fully
  // in frame without being needlessly zoomed out.
  const radius = 0.5 * Math.sqrt(size.x ** 2 + size.y ** 2 + size.z ** 2);

  const fovV = (camera.fov * Math.PI) / 180;
  const fovH = 2 * Math.atan(Math.tan(fovV / 2) * camera.aspect);
  const tightestFov = Math.min(fovV, fovH);

  const padding = 1.15;
  const distance = (radius / Math.sin(tightestFov / 2)) * padding;

  const dir = new THREE.Vector3(1, 0.65, 1).normalize();

  camera.position.copy(center).addScaledVector(dir, distance);
  camera.near = Math.max(distance / 1000, 0.05);
  camera.far = distance * 8 + maxDim * 4;
  camera.updateProjectionMatrix();

  controls.target.copy(center);
  controls.minDistance = Math.max(maxDim * 0.02, 0.5);
  controls.maxDistance = distance * 6;
  controls.update();

  directionalLight.target.position.copy(center);
  directionalLight.target.updateMatrixWorld();

  fillLight.target.position.copy(center);
  fillLight.target.updateMatrixWorld();
}

// ===============================================================
// SHOW MODEL
// ===============================================================

function showModel(key) {
  const entry = models[key];

  if (!entry) {
    console.error("MODEL NOT LOADED:", key);
    setStatus(
      `${labelFor(key)} isn't loaded yet${loadErrors[key] ? " (failed to load)" : ""
      }.`
    );
    return;
  }

  MODEL_CONFIG.forEach((cfg) => {
    if (models[cfg.key]) {
      models[cfg.key].root.visible = cfg.key === key;
    }
  });

  currentKey = key;
  markActiveButton(key);

  stopFlythrough();
  frameModel(entry);

  setStatus(`Showing ${labelFor(key)}`);
  console.log("SHOWING MODEL:", key);
}

function labelFor(key) {
  return MODEL_CONFIG.find((c) => c.key === key)?.label ?? key;
}

// ===============================================================
// LOAD ALL MODELS
// ===============================================================

const loadErrors = {};
let loadedCount = 0;

function loadModel(cfg) {
  setButtonState(cfg.key, "loading");

  return new Promise((resolve) => {
    loader.load(
      cfg.url,

      (gltf) => {
        const root = gltf.scene;
        scene.add(root);
        root.visible = false;

        const measurements = prepareModel(root, cfg.label);
        models[cfg.key] = { root, ...measurements };

        setButtonState(cfg.key, "ready");
        console.log(`✅ ${cfg.label} loaded`);

        loadedCount += 1;
        updateOverlayProgress();
        resolve();
      },

      (progress) => {
        if (progress.total > 0) {
          const percent = ((progress.loaded / progress.total) * 100).toFixed(
            0
          );
          setStatus(`Loading ${cfg.label}... ${percent}%`);
        }
      },

      (error) => {
        loadErrors[cfg.key] = error;
        setButtonState(cfg.key, "error");
        console.error(`❌ ${cfg.label} failed to load:`, error);

        loadedCount += 1;
        updateOverlayProgress();
        resolve(); // resolve (not reject) so Promise.all still settles
      }
    );
  });
}

function updateOverlayProgress() {
  const total = MODEL_CONFIG.length;
  if (overlayEl) {
    const progressEl = overlayEl.querySelector(".loading-progress");
    if (progressEl) progressEl.textContent = `${loadedCount} / ${total}`;
  }
}

async function loadAllModels() {
  setStatus("Loading models...");

  await Promise.all(MODEL_CONFIG.map(loadModel));

  const failed = MODEL_CONFIG.filter((c) => loadErrors[c.key]);
  const succeeded = MODEL_CONFIG.filter((c) => !loadErrors[c.key]);

  if (overlayEl) overlayEl.classList.add("is-hidden");

  if (failed.length === 0) {
    setStatus("Models ready");
  } else if (succeeded.length === 0) {
    setStatus("All models failed to load — check the console for details.");
  } else {
    setStatus(
      `Models ready (${failed.length} failed: ${failed
        .map((c) => c.label)
        .join(", ")})`
    );
  }

  // Auto-show the first model that loaded successfully.
  const first = succeeded[0];
  if (first) {
    showModel(first.key);
  }
}

loadAllModels();

// ===============================================================
// FLYTHROUGH
// ===============================================================

let flying = false;
let flyClock = new THREE.Clock(false);
let flyPath = null;
const FLYTHROUGH_DURATION = 18; // seconds per full loop

function createFlyPath(entry) {
  const { size, center } = entry;

  const radius = Math.max(size.x, size.z) * 0.55;
  const baseHeight = center.y + size.y * 0.65 + Math.max(size.y, 20) * 0.5;

  const points = [];
  const loops = 8;

  for (let i = 0; i < loops; i += 1) {
    const angle = (i / loops) * Math.PI * 2;
    const heightWobble = Math.sin(angle * 2) * size.y * 0.15;

    points.push(
      new THREE.Vector3(
        center.x + Math.cos(angle) * radius,
        baseHeight + heightWobble,
        center.z + Math.sin(angle) * radius
      )
    );
  }

  const curve = new THREE.CatmullRomCurve3(points, true, "catmullrom", 0.5);
  return curve;
}

function startFlythrough() {
  const entry = models[currentKey];

  if (!entry) {
    setStatus("Select a model before starting the flythrough.");
    return;
  }

  flyPath = createFlyPath(entry);
  flying = true;
  flyClock.start();
  controls.enabled = false;

  setActiveCameraButton("start");
  setStatus(`Flythrough: ${labelFor(currentKey)}`);
  console.log("FLYTHROUGH STARTED");
}

function stopFlythrough() {
  flying = false;
  flyClock.stop();
  controls.enabled = true;
  setActiveCameraButton("manual");
}

function updateFlythrough() {
  if (!flyPath || !models[currentKey]) return;

  const t = (flyClock.getElapsedTime() % FLYTHROUGH_DURATION) / FLYTHROUGH_DURATION;

  const position = flyPath.getPointAt(t);
  camera.position.copy(position);

  camera.lookAt(models[currentKey].center);
}

function resetCamera() {
  flying = false;
  flyClock.stop();
  controls.enabled = true;
  setActiveCameraButton("reset");

  const entry = models[currentKey];
  if (entry) frameModel(entry);
}

function setActiveCameraButton(which) {
  const map = {
    start: "btn-flythrough-start",
    manual: "btn-flythrough-stop",
    reset: "btn-camera-reset",
  };
  Object.values(map).forEach((id) => {
    document.getElementById(id)?.classList.remove("is-active");
  });
  const activeId = map[which];
  if (activeId) document.getElementById(activeId)?.classList.add("is-active");
}

// ===============================================================
// ANIMATION LOOP
// ===============================================================

function animate() {
  requestAnimationFrame(animate);

  if (flying) {
    updateFlythrough();
  } else {
    controls.update();
  }

  renderer.render(scene, camera);
}

animate();

// ===============================================================
// RESIZE
// ===============================================================

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ===============================================================
// WIRE UP UI
// ===============================================================

MODEL_CONFIG.forEach((cfg) => {
  document.getElementById(cfg.buttonId)?.addEventListener("click", () => {
    showModel(cfg.key);
  });
});

document
  .getElementById("btn-flythrough-start")
  ?.addEventListener("click", startFlythrough);

document
  .getElementById("btn-flythrough-stop")
  ?.addEventListener("click", stopFlythrough);

document
  .getElementById("btn-camera-reset")
  ?.addEventListener("click", resetCamera);

// Exposed for console debugging / inline HTML fallback.
window.showModel = showModel;
window.startFlythrough = startFlythrough;
window.stopFlythrough = stopFlythrough;
window.resetCamera = resetCamera;