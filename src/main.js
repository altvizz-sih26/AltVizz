import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { loadNPZ, getMetrics, getString, getScalar } from "./npz-loader.js";
import viewerInputConfig, { resolveAnalysisSources, resolveMeshMode } from "./viewer-input-config.js";
import { fromUrl as geotiffFromUrl } from "geotiff";
import "./style.css";


// ============================================================================
// HELPER
// ============================================================================

const $ = (id) => document.getElementById(id);

import "./viewer.css";

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

const params = new URLSearchParams(window.location.search);
const dynamicGlbUrl = params.get("glb");
const dynamicLabel = params.get("label") || "Generated Terrain";

const MODEL_CONFIG = dynamicGlbUrl
  ? [
      { key: "dynamic", label: dynamicLabel, url: dynamicGlbUrl, buttonId: "btn-bare" },
    ]
  : [
      { key: "bare", label: "Bare Terrain", url: "/depthwizard_bare_terrain_3d.glb", buttonId: "btn-bare" },
      { key: "terrain", label: "Terrain", url: "/depthwizard_urban_3d.glb", buttonId: "btn-terrain" },
      { key: "vegetation", label: "Vegetation", url: "/urban_depthwizard_3d.glb", buttonId: "btn-vegetation" },
    ];

// ===============================================================
// DYNAMIC MODE UI ADJUSTMENTS
// ===============================================================
// When a real backend result is loaded via ?glb=&label=, the page only
// ever shows ONE model — so the static 3-layer HTML (button text,
// loading copy, "Layers" group label) needs to reflect that instead of
// leftover demo copy ("Bare Terrain", "Reading bare terrain, terrain
// and vegetation meshes", etc.), which would otherwise be misleading
// for a user looking at their own reconstruction.
if (dynamicGlbUrl) {
  // Remove the two demo-only layer buttons; #btn-bare is repurposed
  // below as the single "current model" button instead of removing it.
  document.getElementById("btn-terrain")?.remove();
  document.getElementById("btn-vegetation")?.remove();

  // Relabel the remaining layer button with the real model's label
  // instead of leaving the hardcoded "Bare Terrain" text in place.
  const dynamicBtn = document.getElementById("btn-bare");
  if (dynamicBtn) {
    const textEl = dynamicBtn.querySelector(".ctrl-text");
    if (textEl) textEl.textContent = dynamicLabel;

    const iconEl = dynamicBtn.querySelector(".ctrl-icon");
    if (iconEl) iconEl.textContent = "📍";
  }

  // "Layers" no longer makes sense when there's only one model to show.
  const layersGroup = document.querySelector('.control-group[aria-label="Model layers"]');
  if (layersGroup) {
    const groupLabel = layersGroup.querySelector(".control-group-label");
    if (groupLabel) groupLabel.textContent = "Model";
  }

  // Loading overlay copy assumed the 3 static demo files; rewrite it to
  // describe loading a single generated result instead.
  const loadingTitleEl = document.querySelector(".loading-title");
  if (loadingTitleEl) loadingTitleEl.textContent = "Preparing your reconstruction";

  const loadingNoteEl = document.querySelector(".loading-note");
  if (loadingNoteEl) loadingNoteEl.textContent = `Loading ${dynamicLabel}`;
}

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
// CONTROLS
// ===============================================================

const controls = new OrbitControls(camera, renderer.domElement);

controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.screenSpacePanning = true; // Google Maps jaisa flat pan

controls.mouseButtons = {
  LEFT: THREE.MOUSE.PAN,     // left-drag = pan (left/right/up/down)
  MIDDLE: THREE.MOUSE.DOLLY, // scroll = zoom
  RIGHT: THREE.MOUSE.ROTATE, // right-drag = rotate/tilt
};

controls.touches = {
  ONE: THREE.TOUCH.PAN,
  TWO: THREE.TOUCH.DOLLY_ROTATE, // 2-finger = zoom+rotate (mobile)
};

controls.panSpeed = 1.2;
controls.rotateSpeed = 0.8;

controls.minDistance = 5;
controls.maxDistance = 5000;
controls.maxPolarAngle = Math.PI * 0.495; // stop just short of going underground

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

            i.removeAttribute(
              "aria-current"
            );

          }
        );

        item.classList.add(
          "is-active"
        );

        item.setAttribute(
          "aria-current",
          "true"
        );
      }
    );

  }
);


// ============================================================================
// MODEL PREPARATION
// ============================================================================

function prepareModel(root) {

  const maxAnisotropy =
    renderer.capabilities
      .getMaxAnisotropy();


  root.traverse(
    (object) => {

      if (!object.isMesh) {
        return;
      }

      object.castShadow = true;
      object.receiveShadow = true;


      // ------------------------------------------------------------
      // Normals
      // ------------------------------------------------------------

      if (
        object.geometry &&
        !object.geometry.attributes.normal
      ) {

        object.geometry.computeVertexNormals();

      }


      // ------------------------------------------------------------
      // Materials
      // ------------------------------------------------------------

      const materials =
        Array.isArray(object.material)
          ? object.material
          : [object.material];


      materials.forEach(
        (material) => {

          if (!material) {
            return;
          }

          material.side =
            THREE.DoubleSide;


          if (material.map) {

            material.map.colorSpace =
              THREE.SRGBColorSpace;

            material.map.anisotropy =
              maxAnisotropy;

            material.map.needsUpdate =
              true;
          }


          // Keep original GLB material.
          if (
            material.isMeshStandardMaterial ||
            material.isMeshPhysicalMaterial
          ) {

            material.metalness =
              Math.min(
                material.metalness ?? 0,
                0.25
              );

            material.roughness =
              Math.max(
                material.roughness ?? 0.8,
                0.5
              );
          }

          material.needsUpdate =
            true;
        }
      );

    }
  );


  // ------------------------------------------------------------
  // Person 3 output is already Z-UP.
  // X/Y = terrain plane, Z = elevation.
  // No axis rotation is applied here.
  // ------------------------------------------------------------

  root.position.set(0, 0, 0);
  root.rotation.set(0, 0, 0);
  root.updateMatrixWorld(true);

  // Measure the mesh in its own coordinates and detect which axis is "up"
  // (the thinnest one, for a terrain). glTF is normally Y-up; the viewer is Z-up.
  const mapping = buildMeshMapping(root);
  if (mapping.up === "y") {
    root.rotation.x = Math.PI / 2; // Y-up -> Z-up
  }
  root.updateMatrixWorld(true);

  // ------------------------------------------------------------
  // Center only the horizontal X/Y position.
  // Keep Z as the actual elevation axis.
  // ------------------------------------------------------------

  const box =
    new THREE.Box3()
      .setFromObject(root);

  const center =
    box.getCenter(
      new THREE.Vector3()
    );

  root.position.x -= center.x;
  root.position.y -= center.y;

  root.updateMatrixWorld(true);


  // ------------------------------------------------------------
  // Final dimensions
  // ------------------------------------------------------------

  const finalBox =
    new THREE.Box3()
      .setFromObject(root);

  const size =
    finalBox.getSize(
      new THREE.Vector3()
    );

  const finalCenter =
    finalBox.getCenter(
      new THREE.Vector3()
    );


  // ------------------------------------------------------------
  // Vertex count
  // ------------------------------------------------------------

  let vertexCount = 0;

  root.traverse(
    (object) => {

      if (
        object.isMesh &&
        object.geometry?.attributes?.position
      ) {

        vertexCount +=
          object.geometry
            .attributes
            .position
            .count;
      }

    }
  );


  return { size, center: finalCenter, vertexCount, mapping, upAxis: mapping.up };
}


// ============================================================================
// FRAME MODEL
// ============================================================================

function frameModel(model) {

  if (!model) {
    return;
  }

  const {
    size,
    center,
  } = model;


  const radius =
    Math.max(
      size.length() / 2,
      10
    );


  const fov =
    THREE.MathUtils.degToRad(
      camera.fov
    );


  const distance =
    (radius /
      Math.sin(fov / 2)) *
    1.25;


  const direction =
    new THREE.Vector3(
      1,
      0.8,
      0.85
    ).normalize();


  camera.position.copy(
    center
  );

  camera.position.addScaledVector(
    direction,
    distance
  );


  camera.near =
    Math.max(
      distance / 1000,
      0.05
    );

  camera.far =
    Math.max(
      distance * 10,
      10000
    );

  camera.updateProjectionMatrix();


  controls.target.copy(
    center
  );

  controls.minDistance =
    Math.max(
      size.length() * 0.005,
      1
    );

  controls.maxDistance =
    Math.max(
      distance * 8,
      5000
    );

  controls.update();


  sun.target.position.copy(
    center
  );

  sun.target.updateMatrixWorld();

  fill.target.position.copy(
    center
  );

  fill.target.updateMatrixWorld();


  applySolarPosition();

  clearCameraPresetActive();
}


// ============================================================================
// CAMERA PRESET ACTIVE
// ============================================================================

function clearCameraPresetActive() {

  [
    btnCameraTop,
    btnCameraIso,
    btnCameraProfile,
    btnCameraResetPreset,
  ].forEach(
    (btn) =>
      btn?.classList.remove(
        "is-active"
      )
  );
}


// ============================================================================
// CAMERA PRESETS
// ============================================================================

function applyCameraPreset(
  kind,
  button
) {

  if (!activeModel) {

    setStatus(
      "Load a terrain before selecting a camera preset.",
      true
    );

    return;
  }


  stopFlythrough(false);
  setAutoRotate(false);

  controls.enabled = true;


  const {
    size,
    center,
  } = activeModel;


  const radius =
    Math.max(
      size.length() / 2,
      10
    );


  const fov =
    THREE.MathUtils.degToRad(
      camera.fov
    );


  const distance =
    (radius /
      Math.sin(fov / 2)) *
    1.3;


  let direction;


  if (kind === "top") {

    // Look straight down the Z elevation axis.
    direction =
      new THREE.Vector3(
        0,
        0,
        1
      ).normalize();

    // Avoid camera roll when looking straight down.
    camera.up.set(0, 1, 0);

  }

  else if (kind === "iso") {

    // X/Y = terrain plane, Z = elevation.
    direction =
      new THREE.Vector3(
        1,
        0.9,
        0.8
      ).normalize();

    camera.up.set(0, 0, 1);

  }

  else if (kind === "profile") {

    // Look across the terrain in Y,
    // with Z remaining vertical.
    direction =
      new THREE.Vector3(
        0,
        1,
        0
      ).normalize();

    camera.up.set(0, 0, 1);

  }

  else {

    direction =
      new THREE.Vector3(
        1,
        0.7,
        0.8
      ).normalize();

    camera.up.set(0, 0, 1);

  }


  camera.position.copy(
    center
  );

  camera.position.addScaledVector(
    direction,
    distance
  );


  camera.near =
    Math.max(
      distance / 1000,
      0.05
    );

  camera.far =
    Math.max(
      distance * 10,
      10000
    );

  camera.updateProjectionMatrix();


  controls.target.copy(
    center
  );

  controls.update();


  clearCameraPresetActive();

  button?.classList.add(
    "is-active"
  );


  setStatus(
    `Camera preset: ${kind}`
  );
}


btnCameraTop?.addEventListener(
  "click",
  () =>
    applyCameraPreset(
      "top",
      btnCameraTop
    )
);


btnCameraIso?.addEventListener(
  "click",
  () =>
    applyCameraPreset(
      "iso",
      btnCameraIso
    )
);


btnCameraProfile?.addEventListener(
  "click",
  () =>
    applyCameraPreset(
      "profile",
      btnCameraProfile
    )
);


btnCameraResetPreset?.addEventListener(
  "click",
  resetCamera
);


// ============================================================================
// DISPOSE MODEL
// ============================================================================

function disposeModel(model) {

  if (!model?.root) {
    return;
  }


  model.root.traverse(
    (object) => {

      if (!object.isMesh) {
        return;
      }


      if (object.geometry) {
        object.geometry.dispose();
      }


      const materials =
        Array.isArray(object.material)
          ? object.material
          : [object.material];


      materials.forEach(
        (material) => {

          if (!material) {
            return;
          }


          // Dispose textures.
          [
            "map",
            "normalMap",
            "roughnessMap",
            "metalnessMap",
            "aoMap",
            "emissiveMap",
            "alphaMap",
            "bumpMap",
          ].forEach(
            (key) => {

              if (
                material[key] &&
                material[key].isTexture
              ) {

                material[key].dispose();
              }

            }
          );


          material.dispose();
        }
      );

    }
  );


  scene.remove(
    model.root
  );
}


// ============================================================================
// READ REAL GEOREFERENCING FROM THE SRTM FILE
// This is what makes Coordinates / Spatial Resolution / Datum show
// actual values instead of "Not provided by pipeline yet" — it reads
// the GeoTIFF's own embedded tags, nothing is typed in by hand.
// ============================================================================

async function loadSRTMMetadata(url) {

  if (!url) {
    return;
  }

  try {

    const tiff = await geotiffFromUrl(url);
    const image = await tiff.getImage();

    // [minX, minY, maxX, maxY] in the file's native CRS units.
    const bbox = image.getBoundingBox();

    // Pixel size in native units. SRTM tiles are almost always
    // delivered in geographic degrees (EPSG:4326), not meters —
    // so we convert approximately for a human-readable value and
    // say so, rather than silently mislabeling degrees as meters.
    const [resX] = image.getResolution();

    const geoKeys = image.getGeoKeys ? image.getGeoKeys() : {};
    const epsg =
      (geoKeys && (geoKeys.ProjectedCSTypeGeoKey || geoKeys.GeographicTypeGeoKey)) ||
      null;

    const isLikelyDegrees = Math.abs(resX) < 1;
    const approxMeters = Math.abs(resX) * 111320; // rough, equator-based

    const resolutionText = isLikelyDegrees
      ? `~${approxMeters.toFixed(1)} m/px (from ${Math.abs(resX).toFixed(6)}°/px, approx.)`
      : `${Math.abs(resX).toFixed(2)} m/px`;

    const coordinatesText = isLikelyDegrees
      ? `Lat/Lon: ${bbox[1].toFixed(4)}, ${bbox[0].toFixed(4)} — ${bbox[3].toFixed(4)}, ${bbox[2].toFixed(4)}`
      : `Easting/Northing (m): ${bbox[0].toFixed(1)}, ${bbox[1].toFixed(1)} — ${bbox[2].toFixed(1)}, ${bbox[3].toFixed(1)}`;

    const horizontalDatumText = epsg
      ? `EPSG:${epsg}`
      : "WGS84 (assumed — SRTM default, EPSG code not found in file)";

    // SRTM elevation values are referenced to the EGM96 geoid, not
    // the ellipsoid — this is the standard vertical datum for SRTM
    // and isn't something we can read from the GeoTIFF's own tags,
    // so it's stated as the known SRTM convention rather than parsed.
    const verticalDatumText = "EGM96 geoid (SRTM standard)";

    if (viewportGeorefEl) {
      viewportGeorefEl.textContent = "Georeferenced (SRTM)";
    }

    if (metadataCoordinatesEl) {
      metadataCoordinatesEl.textContent = coordinatesText;
    }

    if (metadataResolutionEl) {
      metadataResolutionEl.textContent = resolutionText;
    }

    if (metadataDatumEl) {
      metadataDatumEl.textContent =
        `Horizontal: ${horizontalDatumText} · Vertical: ${verticalDatumText}`;
    }

    // Keep the config object itself in sync, so anything else reading
    // viewerInputConfig.metadata later in this session sees real
    // values instead of the original nulls.
    if (viewerInputConfig && viewerInputConfig.metadata) {
      viewerInputConfig.metadata.spatialResolution = resolutionText;
      viewerInputConfig.metadata.coordinates = coordinatesText;
      viewerInputConfig.metadata.horizontalDatum = horizontalDatumText;
      viewerInputConfig.metadata.verticalDatum = verticalDatumText;
    }

    console.log(
      "[SASA MINE] SRTM metadata loaded:",
      { bbox, resX, epsg }
    );

  } catch (error) {

    console.error(
      "[SASA MINE] Could not read SRTM GeoTIFF metadata:",
      error
    );

    setStatus(
      "Could not read SRTM file metadata — check console for details.",
      true
    );
  }
}


// ============================================================================
// DERIVE LABEL FROM FILENAME
// This is what makes the model name reflect the actual file the
// pipeline handed us, instead of a hardcoded string like
// "Configured terrain" or a leftover test label.
// ============================================================================

function deriveLabelFromUrl(url) {

  if (!url) {
    return "Terrain Model";
  }

  try {

    const clean = url.split("?")[0].split("#")[0];
    const filename = clean.substring(clean.lastIndexOf("/") + 1);
    const withoutExt = filename.replace(/\.(glb|gltf)$/i, "");

    if (!withoutExt) {
      return "Terrain Model";
    }

    return withoutExt
      .replace(/[_-]+/g, " ")
      .replace(/\s+/g, " ")
      .trim()
      .split(" ")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");

  } catch (error) {

    console.warn("[SASA MINE] Could not derive label from URL:", url, error);
    return "Terrain Model";
  }
}


// ============================================================================
// LOAD GLB FROM URL
// ============================================================================

function loadGLBFromURL(
  url,
  label = "Terrain Model",
  revokeURL = false
) {

  return new Promise(
    (resolve, reject) => {

      if (!url) {

        setStatus(
          "No GLB file selected.",
          true
        );

        reject(
          new Error("No GLB URL")
        );

        return;
      }


      showLoading(
        "SASA MINE",
        `Loading ${label}...`
      );


      setStatusBusy(
        "Loading terrain model..."
      );


      setRowStatus(
        statusIndicatorModelEl,
        statusValueModelEl,
        "Loading...",
        "busy"
      );


      stopFlythrough(false);


      loader.load(

        url,


        // ======================================================
        // SUCCESS
        // ======================================================

        (gltf) => {

          try {

            const root =
              gltf.scene;


            if (!root) {

              throw new Error(
                "GLB scene is empty."
              );
            }


            // Remove previous model FIRST.
            const oldModel =
              activeModel;


            if (oldModel) {

              disposeModel(
                oldModel
              );

              activeModel =
                null;
            }


            // A fresh model load invalidates any before/after pairing
            // that was built against the previous After mesh.
            clearComparisonState();


            // Add new model.
            scene.add(root);


            const measurements =
              prepareModel(
                root
              );


            activeModel = {

              root,

              size:
                measurements.size,

              center:
                measurements.center,

              vertexCount: measurements.vertexCount,
              mapping: measurements.mapping,
              upAxis: measurements.upAxis,

              meshRows:
                DEFAULT_MESH_ROWS,

              meshCols:
                DEFAULT_MESH_COLS,

              gridRows:
                terrainData?.rows ??
                GRID_ROWS,

              gridCols:
                terrainData?.cols ??
                GRID_COLS,

              samplingStep:
                DEFAULT_GRID_STEP,
            };


            // ------------------------------------------------
            // Apply existing display states
            // ------------------------------------------------

            applyWireframeState();

            calibrateMeshMapping();
            applySlopeHeatmapState();
            applyZExaggeration();


            // ------------------------------------------------
            // Camera
            // ------------------------------------------------

            frameModel(
              activeModel
            );


            clearInspection();

            hideLoading();


            // ------------------------------------------------
            // Status
            // ------------------------------------------------

            setRowStatus(
              statusIndicatorModelEl,
              statusValueModelEl,
              label,
              "ok"
            );


            setRowStatus(
              statusIndicatorRenderEl,
              statusValueRenderEl,
              "READY",
              "ok"
            );


            setRowStatus(
              statusIndicatorCameraEl,
              statusValueCameraEl,
              "READY",
              "ok"
            );


            if (
              telemetryModelValueEl
            ) {

              telemetryModelValueEl.textContent =
                label;
            }


            updateModelMetadataPanels(
              label
            );


            setStatus(
              `Model loaded: ${label}`
            );


            if (revokeURL) {
              URL.revokeObjectURL(
                url
              );
            }


            resolve(
              activeModel
            );

          }

          catch (error) {

            console.error(
              "[SASA MINE] Model preparation error:",
              error
            );


            hideLoading();


            setRowStatus(
              statusIndicatorModelEl,
              statusValueModelEl,
              "ERROR",
              "error"
            );


            setStatus(
              "GLB loaded but could not be prepared.",
              true
            );


            if (revokeURL) {
              URL.revokeObjectURL(
                url
              );
            }


            reject(error);
          }
        },


        // ======================================================
        // PROGRESS
        // ======================================================

        (progress) => {

          if (
            progress &&
            progress.total > 0
          ) {

            updateLoadingProgress(
              (
                progress.loaded /
                progress.total
              ) * 100
            );
          }
        },


        // ======================================================
        // ERROR
        // ======================================================

        (error) => {

          console.error(
            "[SASA MINE] GLB ERROR:",
            error
          );


          hideLoading();


          setRowStatus(
            statusIndicatorModelEl,
            statusValueModelEl,
            "ERROR",
            "error"
          );


          setStatus(
            "Could not load this GLB file. Check the browser console.",
            true
          );


          if (revokeURL) {
            URL.revokeObjectURL(
              url
            );
          }


          reject(error);
        }
      );

    }
  );
}


// ============================================================================
// LOAD GLB FROM FILE
// ============================================================================

function loadGLBFromFile(file) {

  if (!file) {
    return;
  }


  const filename =
    file.name.toLowerCase();


  if (
    !filename.endsWith(".glb") &&
    !filename.endsWith(".gltf")
  ) {

    setStatus(
      "Please select a .glb or .gltf file.",
      true
    );

    return;
  }


  setStatus(
    `Selected: ${file.name}`
  );


  const objectURL =
    URL.createObjectURL(
      file
    );


  loadGLBFromURL(
    objectURL,
    file.name,
    true
  ).catch(
    () => { }
  );
}


// ============================================================================
// MODEL METADATA
// ============================================================================

function updateModelMetadataPanels(
  label
) {

  if (metadataDatasetEl) {

    metadataDatasetEl.textContent =
      label;
  }


  if (
    metadataMeshEl &&
    activeModel
  ) {

    metadataMeshEl.textContent =
      `${activeModel.vertexCount.toLocaleString()} vertices`;
  }


  if (
    viewportMeshEl &&
    activeModel
  ) {

    viewportMeshEl.textContent =
      `${activeModel.vertexCount.toLocaleString()} verts`;
  }


  // ------------------------------------------------------------
  // These read from the REAL viewer-input-config.js schema:
  // viewerInputConfig.metadata = { datasetName, spatialResolution,
  // coordinates, horizontalDatum, verticalDatum, calibration }.
  // All start out null in local testing — loadSRTMMetadata() fills
  // in real values from the actual .tif where it can. Anything still
  // null falls back to an honest label, never a fabricated value.
  // ------------------------------------------------------------

  const configMeta = (viewerInputConfig && viewerInputConfig.metadata) || {};

  if (viewportGeorefEl) {

    viewportGeorefEl.textContent =
      configMeta.coordinates || configMeta.spatialResolution
        ? "Georeferenced"
        : "Not provided by pipeline yet";
  }


  if (metadataResolutionEl) {

    metadataResolutionEl.textContent =
      configMeta.spatialResolution ||
      "Not provided by pipeline yet";
  }


  if (metadataCoordinatesEl) {

    metadataCoordinatesEl.textContent =
      configMeta.coordinates ||
      "Not provided by pipeline yet";
  }


  if (metadataDatumEl) {

    const horiz = configMeta.horizontalDatum;
    const vert = configMeta.verticalDatum;

    metadataDatumEl.textContent =
      (horiz || vert)
        ? [
            horiz ? `Horizontal: ${horiz}` : null,
            vert ? `Vertical: ${vert}` : null,
          ].filter(Boolean).join(" · ")
        : "Not provided by pipeline yet";
  }


  if (metadataCalibrationEl) {

    metadataCalibrationEl.textContent =
      configMeta.calibration ||
      "Relative depth (uncalibrated)";
  }


  updateElevationPanels();
}


// ============================================================================
// ELEVATION PANELS
// ============================================================================

function updateElevationPanels() {

  if (!activeModel) {
    return;
  }


  let peak;
  let base;


  if (
    gridLoaded &&
    terrainData?.elevation
  ) {

    const range =
      scanGridRange(
        terrainData.elevation
      );


    if (range) {

      peak = range.max;
      base = range.min;
    }
  }


  // If grid isn't loaded, use model dimensions.
  if (
    peak === undefined ||
    base === undefined
  ) {

    const elevationBox = new THREE.Box3().setFromObject(
    activeModel.root
);

peak = elevationBox.max.z;
base = elevationBox.min.z;
  }


  const relief =
    peak - base;


  // Legend labels — only meaningful against real per-point data,
  // not the GLB-mesh fallback range used above for peak/base display.
  const realRange =
    gridLoaded && terrainData?.elevation
      ? scanGridRange(terrainData.elevation)
      : null;

  if (elevationLegendMinEl && elevationLegendMidEl && elevationLegendMaxEl) {

    if (realRange) {

      const mid = (realRange.min + realRange.max) / 2;

      elevationLegendMinEl.textContent = formatValue(realRange.min, " m");
      elevationLegendMidEl.textContent = formatValue(mid, " m");
      elevationLegendMaxEl.textContent = formatValue(realRange.max, " m");

    } else {

      elevationLegendMinEl.textContent = "---";
      elevationLegendMidEl.textContent = "---";
      elevationLegendMaxEl.textContent = "---";
    }
  }


  if (elevationPeakEl) {

    elevationPeakEl.textContent =
      formatValue(
        peak,
        " m"
      );
  }


  if (elevationBaseEl) {

    elevationBaseEl.textContent =
      formatValue(
        base,
        " m"
      );
  }


  if (elevationReliefEl) {

    elevationReliefEl.textContent =
      formatValue(
        relief,
        " m"
      );
  }


  if (metadataElevationRangeEl) {

    metadataElevationRangeEl.textContent =
      `${formatValue(base, " m")} – ${formatValue(peak, " m")}`;
  }


  drawElevationProfile();
}


// ============================================================================
// SCAN GRID RANGE
// ============================================================================

function scanGridRange(grid) {

  if (!grid) {
    return null;
  }


  let min = Infinity;
  let max = -Infinity;

  let found = false;


  const values =
    Array.isArray(grid) &&
      Array.isArray(grid[0])
      ? grid.flat()
      : grid;


  if (
    !values ||
    typeof values.length !==
    "number"
  ) {

    return null;
  }


  for (
    let i = 0;
    i < values.length;
    i++
  ) {

    const value =
      Number(values[i]);


    if (
      !Number.isFinite(value)
    ) {

      continue;
    }


    found = true;

    min =
      Math.min(
        min,
        value
      );

    max =
      Math.max(
        max,
        value
      );
  }


  if (!found) {
    return null;
  }


  return {
    min,
    max,
  };
}


// ============================================================================
// COMPUTE SLOPE FROM ELEVATION
//
// Central-difference gradient over the grid, converted to a slope
// angle in degrees. Cell spacing is treated as 1 grid unit rather
// than a real-world distance, since the analysis grid's physical
// resolution isn't reliably known (the SRTM file's resolution is a
// separate raster, not guaranteed to match this grid 1:1) — so this
// is a relative, uncalibrated slope, consistent with how the app
// already labels elevation itself ("Relative depth (uncalibrated)").
// Good enough to drive a heatmap and a rough terrain-type estimate;
// not a substitute for a real slope field from the pipeline.
// ============================================================================

function computeSlopeGrid(elevation, rows, cols) {

  if (!elevation || !rows || !cols) {
    return null;
  }

  const result = new Float32Array(rows * cols);

  for (let row = 0; row < rows; row++) {

    for (let col = 0; col < cols; col++) {

      const center = getGridValue(elevation, row, col, cols);

      if (typeof center !== "number") {

        result[row * cols + col] = NaN;
        continue;
      }

      const left = getGridValue(elevation, row, Math.max(col - 1, 0), cols);
      const right = getGridValue(elevation, row, Math.min(col + 1, cols - 1), cols);
      const up = getGridValue(elevation, Math.max(row - 1, 0), col, cols);
      const down = getGridValue(elevation, Math.min(row + 1, rows - 1), col, cols);

      const dzdx =
        (typeof left === "number" && typeof right === "number")
          ? (right - left) / 2
          : 0;

      const dzdy =
        (typeof up === "number" && typeof down === "number")
          ? (down - up) / 2
          : 0;

      const slopeRad = Math.atan(Math.sqrt(dzdx * dzdx + dzdy * dzdy));

      result[row * cols + col] = THREE.MathUtils.radToDeg(slopeRad);
    }
  }

  return result;
}


// Rough, clearly-labeled terrain classification from slope alone —
// used only as a fallback when the analysis file has no categorical
// terrain field of its own (this NPZ schema doesn't, per the comment
// on normalizeGrid). Always suffixed "(est.)" by the caller so it's
// never confused with a real pipeline-provided classification.
function deriveTerrainTypeLabel(slopeDegrees) {

  if (typeof slopeDegrees !== "number" || Number.isNaN(slopeDegrees)) {
    return null;
  }

  if (slopeDegrees < 5) return "Flat";
  if (slopeDegrees < 15) return "Gentle Slope";
  if (slopeDegrees < 30) return "Moderate Slope";
  if (slopeDegrees < 45) return "Steep Slope";

  return "Very Steep";
}


// ============================================================================
// ELEVATION PROFILE
// ============================================================================

function drawElevationProfile() {

  if (!elevationChartEl) {
    return;
  }


  if (
    !gridLoaded ||
    !terrainData?.elevation
  ) {

    elevationChartEl.innerHTML =
      "";

    return;
  }


  const rows =
    terrainData.rows;

  const cols =
    terrainData.cols;


  const midRow =
    Math.floor(
      rows / 2
    );


  const rowValues = [];


  for (
    let col = 0;
    col < cols;
    col++
  ) {

    const value =
      getGridValue(
        terrainData.elevation,
        midRow,
        col,
        cols
      );


    if (
      typeof value ===
      "number"
    ) {

      rowValues.push(
        value
      );
    }
  }


  if (
    rowValues.length < 2
  ) {

    elevationChartEl.innerHTML =
      "";

    return;
  }


  const width = 300;
  const height = 150;


  const min =
    Math.min(
      ...rowValues
    );

  const max =
    Math.max(
      ...rowValues
    );


  const span =
    max - min || 1;


  const step =
    Math.max(
      1,
      Math.floor(
        rowValues.length /
        120
      )
    );


  const points = [];


  for (
    let i = 0;
    i < rowValues.length;
    i += step
  ) {

    const x =
      (
        i /
        (rowValues.length - 1)
      ) *
      width;


    const y =
      height -
      (
        (
          rowValues[i] -
          min
        ) /
        span
      ) *
      (
        height - 10
      ) -
      5;


    points.push(
      `${x.toFixed(1)},${y.toFixed(1)}`
    );
  }


  elevationChartEl.innerHTML = `
    <svg
      viewBox="0 0 ${width} ${height}"
      width="100%"
      height="100%"
      preserveAspectRatio="none"
    >
      <polyline
        points="${points.join(" ")}"
        fill="none"
        stroke="#5dd8e8"
        stroke-width="1.5"
      />
    </svg>
  `;
}


// ============================================================================
// FORMAT VALUE
// ============================================================================

function formatValue(
  value,
  suffix = ""
) {

  if (
    value === null ||
    value === undefined ||
    Number.isNaN(
      Number(value)
    )
  ) {

    return "---";
  }


  if (
    typeof value ===
    "number" ||
    !Number.isNaN(
      Number(value)
    )
  ) {

    return `${Number(value).toFixed(2)}${suffix}`;
  }


  return `${value}${suffix}`;
}


// ============================================================================
// INFER GRID SHAPE FROM ACTUAL DATA
// NPZ files don't necessarily carry an explicit "this is 1024 wide"
// tag — they're just raw arrays. Trusting a hardcoded default here
// silently breaks every indexed (row, col) lookup if the real shape
// differs, even though whole-array scans (min/max) still work fine
// regardless of shape. Always prefer the real shape when we can see it.
// ============================================================================

function inferGridShape(grid) {

  if (!grid) {
    return null;
  }

  // Properly nested 2D array — shape is unambiguous.
  if (Array.isArray(grid) && Array.isArray(grid[0])) {

    return {
      rows: grid.length,
      cols: grid[0].length,
    };
  }

  // Flat array/typed array — only safe to guess if it's a perfect
  // square. A non-square flat array needs explicit dimensions from
  // the source (add rows/cols to the NPZ export), we can't guess it.
  if (Array.isArray(grid) || ArrayBuffer.isView(grid)) {

    const n = grid.length;
    const side = Math.round(Math.sqrt(n));

    if (side * side === n) {

      return {
        rows: side,
        cols: side,
      };
    }
  }

  return null;
}


// ============================================================================
// GRID NORMALIZATION
// ============================================================================

// ============================================================================
// EXTRACT FIELD (handles both raw arrays and {data, shape} wrappers,
// which is the standard way most .npy/.npz JS parsers preserve the
// shape info from the file header alongside the flat data buffer)
// ============================================================================

function extractField(raw, keys) {

  for (const key of keys) {

    const value = raw[key];

    if (value === undefined || value === null) {
      continue;
    }

    if (
      typeof value === "object" &&
      !Array.isArray(value) &&
      !ArrayBuffer.isView(value) &&
      value.data !== undefined
    ) {

      return {
        data: value.data,
        shape: Array.isArray(value.shape) ? value.shape : null,
      };
    }

    return { data: value, shape: null };
  }

  return { data: null, shape: null };
}


// ============================================================================
// GRID NORMALIZATION
//
// Confirmed real field names in Person 2's before/during NPZ export
// (verified directly against the actual file, not guessed):
//   elevation_before, elevation_after, confidence_before,
//   confidence_after, elevation_change, class_changed,
//   combined_confidence, significant_change
//
// This file has NO separate slope array and NO categorical terrain
// class array — only a boolean "did the class change" flag. Slope
// and Terrain Type will correctly stay unavailable for this file;
// that's real, not a bug. A single-snapshot analysis file (if one
// exists) would need those fields added separately.
// ============================================================================

function normalizeGrid(raw) {

  if (!raw || typeof raw !== "object") {

    throw new Error(
      "Grid JSON must contain an object."
    );
  }


  // "elevation" (canonical/current) defaults to the before-state,
  // since this file has no separate absolute baseline outside the
  // before/during comparison itself.
  const elevationField = extractField(raw, [
    "elevation_before", "elevation", "elevation_grid", "elevationGrid",
  ]);

  const beforeField = extractField(raw, [
    "elevation_before", "before", "before_elevation", "before_grid",
  ]);

  const duringField = extractField(raw, [
    "elevation_after", "during", "during_elevation", "during_grid",
  ]);

  const changeField = extractField(raw, [
    "elevation_change", "change", "difference", "difference_grid", "differenceGrid",
  ]);

  const confidenceField = extractField(raw, [
    "combined_confidence", "confidence", "confidence_grid", "confidenceGrid",
  ]);

  // Genuinely absent from this file type — kept as generic aliases
  // in case a different NPZ (a single-snapshot analysis file) does
  // carry these.
  const slopeField = extractField(raw, [
    "slope", "slope_grid", "slopeGrid",
  ]);

  const terrainField = extractField(raw, [
    "terrain", "terrain_type", "terrain_grid", "terrainGrid",
  ]);


  // Prefer real shape metadata over a hardcoded guess, in this
  // priority order: explicit raw.rows/cols/width/height, then a
  // {data, shape} wrapper's own shape, then inference from the
  // actual array, then finally the hardcoded constant as a last
  // resort.
  const explicitShape =
    elevationField.shape ||
    beforeField.shape ||
    duringField.shape ||
    changeField.shape ||
    confidenceField.shape ||
    null;

  // A shape tuple from the NPY header is only usable here if it's
  // genuinely 2D. A 1D export (e.g. the array was flattened with
  // .ravel() before saving) has explicitShape.length === 1, which
  // previously fell through to `explicitShape[1]` -> undefined ->
  // Number(undefined) -> NaN for cols. Every getGridValue() lookup
  // then does `row * NaN + col`, which is NaN for every cell, so the
  // Analysis Map silently paints its "no data" color for the whole
  // canvas — a black box with no error anywhere. Treat a non-2D
  // explicit shape as absent and fall through to inference instead.
  const usableExplicitShape =
    Array.isArray(explicitShape) && explicitShape.length >= 2
      ? explicitShape
      : null;

  if (explicitShape && !usableExplicitShape) {
    console.warn(
      "[SASA MINE] Elevation/grid array has a non-2D shape",
      explicitShape,
      "— ignoring it and inferring dimensions from the flat data instead."
    );
  }

  const inferredShape =
    usableExplicitShape
      ? { rows: usableExplicitShape[0], cols: usableExplicitShape[1] }
      : (
        inferGridShape(elevationField.data) ||
        inferGridShape(beforeField.data) ||
        inferGridShape(duringField.data)
      );

  if (
    !raw.rows && !raw.height &&
    !raw.cols && !raw.width &&
    inferredShape
  ) {

    console.log(
      "[SASA MINE] Grid dimensions not provided explicitly — inferred:",
      inferredShape
    );
  }


  return {

    rows: Number(
      raw.rows ??
      raw.height ??
      (inferredShape && inferredShape.rows) ??
      GRID_ROWS
    ),

    cols: Number(
      raw.cols ??
      raw.width ??
      (inferredShape && inferredShape.cols) ??
      GRID_COLS
    ),

    elevation: elevationField.data,
    slope: slopeField.data,
    terrain: terrainField.data,
    confidence: confidenceField.data,
    before: beforeField.data,
    during: duringField.data,
    change: changeField.data,

    // Kept for reference even though the current UI schema doesn't
    // have dedicated rows for these yet — visible via
    // window.getTerrainData() for debugging, and available to wire
    // into the UI later if useful.
    classChanged: extractField(raw, ["class_changed"]).data,
    significantChange: extractField(raw, ["significant_change"]).data,
  };
}


// ============================================================================
// LOAD GRID OBJECT
// ============================================================================

function loadGridObject(raw) {

  try {

    terrainData =
      normalizeGrid(
        raw
      );


    // Belt-and-suspenders check: even a "valid-looking" 2D shape can
    // disagree with the actual flat array length (e.g. the shape
    // tuple describes a different array than the one that got
    // exported). If rows*cols doesn't match elevation.length, every
    // getGridValue() lookup past the true end of the data silently
    // returns null, which is exactly what makes the Analysis Map
    // render as a plain black box with no thrown error to notice.
    // Log it loudly so a mismatch is obvious in the console instead
    // of only showing up as "the map looks empty."
    if (
      terrainData.elevation &&
      terrainData.rows * terrainData.cols !== terrainData.elevation.length
    ) {

      console.warn(
        "[SASA MINE] Grid dimension mismatch:",
        `rows(${terrainData.rows}) x cols(${terrainData.cols}) =`,
        terrainData.rows * terrainData.cols,
        `but elevation array has ${terrainData.elevation.length} values.`,
        "Analysis Map / Terrain Inspection lookups will be wrong or blank for this file."
      );

      const squareGuess = inferGridShape(terrainData.elevation);

      if (squareGuess) {

        console.warn(
          "[SASA MINE] Falling back to inferred square dimensions:",
          squareGuess
        );

        terrainData.rows = squareGuess.rows;
        terrainData.cols = squareGuess.cols;
      }
    }


    // The analysis NPZ this pipeline currently exports has no slope
    // field at all (see the comment on normalizeGrid) — that's not
    // something loading a GLB can fix, since the GLB only carries
    // 3D shape, not per-point analysis values. Rather than leaving
    // Slope Heatmap permanently disabled and Terrain Inspection's
    // Slope/Terrain Type rows permanently blank, derive slope
    // directly from the elevation grid itself whenever the file
    // doesn't supply it.
    if (!terrainData.slope && terrainData.elevation) {

      terrainData.slope = computeSlopeGrid(
        terrainData.elevation,
        terrainData.rows,
        terrainData.cols
      );

      terrainData.slopeIsEstimated = true;

      console.log(
        "[SASA MINE] No slope field in analysis file — computed slope from elevation grid instead."
      );
    }


    gridLoaded = true;

    slopeRange = null;

    if (activeModel) calibrateMeshMapping();


    if (activeModel) {

      activeModel.gridRows =
        terrainData.rows;

      activeModel.gridCols =
        terrainData.cols;
    }


    setRowStatus(
      statusIndicatorGridEl,
      statusValueGridEl,
      `${terrainData.rows} × ${terrainData.cols}`,
      "ok"
    );


    if (metadataGridEl) {

      metadataGridEl.textContent =
        `${terrainData.rows} × ${terrainData.cols}`;
    }


    if (viewportGridEl) {

      viewportGridEl.textContent =
        `${terrainData.rows} × ${terrainData.cols}`;
    }


    updateElevationPanels();


    // Paint the 2D top-down elevation map now that real data exists.
    // (This call was missing entirely before — the canvas element
    // existed but nothing ever drew to it, which is why it only ever
    // showed as a blank black box.)
    drawAnalysisMap();


    // Recolor the terrain now that real elevation data is available
    // (covers the case where the GLB finished loading before the
    // grid did).
    applySlopeHeatmapState();


    // Visibly grey out Slope Heatmap when this file has no slope
    // data, rather than letting it silently do nothing on click.
    const hasSlope = !!terrainData.slope;

    [btnSlopeHeatmap, btnToolbarSlopeHeatmap].forEach((button) => {
      button?.classList.toggle("is-disabled", !hasSlope);
    });


    setStatus(
      `Grid loaded: ${terrainData.rows} × ${terrainData.cols}`
    );

  }

  catch (error) {

    console.error(
      "[SASA MINE] Grid error:",
      error
    );


    gridLoaded = false;

    terrainData = null;


    setRowStatus(
      statusIndicatorGridEl,
      statusValueGridEl,
      "ERROR",
      "error"
    );


    setStatus(
      "Invalid grid data.",
      true
    );
  }
}


// ============================================================================
// LOAD GRID FILE
// ============================================================================

function loadGridFile(file) {

  if (!file) {
    return;
  }


  if (
    !file.name
      .toLowerCase()
      .endsWith(".json")
  ) {

    setStatus(
      "Please select a JSON grid file.",
      true
    );

    return;
  }


  const reader =
    new FileReader();


  reader.onload = () => {

    try {

      const data =
        JSON.parse(
          reader.result
        );

      loadGridObject(
        data
      );

    }

    catch (error) {

      console.error(
        error
      );

      setStatus(
        "Grid JSON could not be read.",
        true
      );
    }
  };


  reader.onerror = () => {

    setStatus(
      "Could not read the grid file.",
      true
    );
  };


  reader.readAsText(
    file
  );
}


// ============================================================================
// GET GRID VALUE
// ============================================================================

function getGridValue(
  grid,
  row,
  col,
  cols
) {

  if (
    grid === null ||
    grid === undefined
  ) {

    return null;
  }


  // 2D array
  if (
    Array.isArray(grid) &&
    Array.isArray(grid[row])
  ) {

    const value =
      grid[row][col];

    return value ??
      null;
  }


  // Flat array
  if (
    Array.isArray(grid) ||
    ArrayBuffer.isView(grid)
  ) {

    const index =
      row * cols +
      col;

    const value =
      grid[index];

    return value ??
      null;
  }


  return null;
}


// ============================================================================
// GET DATA AT GRID LOCATION
// ============================================================================

function getTerrainDataAt(
  row,
  col
) {

  if (
    !terrainData ||
    !gridLoaded
  ) {

    return {};
  }


  const cols =
    terrainData.cols;


  const slopeValue =
    getGridValue(
      terrainData.slope,
      row,
      col,
      cols
    );


  const rawTerrain =
    getGridValue(
      terrainData.terrain,
      row,
      col,
      cols
    );


  // Real pipeline-provided terrain classification always wins. Only
  // fall back to a slope-derived estimate when the file has none —
  // and label it clearly as an estimate, never presented as if it
  // came from the pipeline.
  const terrainLabel =
    (rawTerrain !== null && rawTerrain !== undefined)
      ? rawTerrain
      : (
        typeof slopeValue === "number"
          ? (() => {
              const label = deriveTerrainTypeLabel(slopeValue);
              return label ? `${label} (est.)` : null;
            })()
          : null
      );


  return {

    elevation:
      getGridValue(
        terrainData.elevation,
        row,
        col,
        cols
      ),

    slope:
      slopeValue,

    terrain:
      terrainLabel,

    confidence:
      getGridValue(
        terrainData.confidence,
        row,
        col,
        cols
      ),

    before:
      getGridValue(
        terrainData.before,
        row,
        col,
        cols
      ),

    during:
      getGridValue(
        terrainData.during,
        row,
        col,
        cols
      ),

    change:
      getGridValue(
        terrainData.change,
        row,
        col,
        cols
      ),
  };
}


// ============================================================================
// WORLD -> PERSON 2 GRID
// ============================================================================

// Mesh <-> grid mapping, derived from the REAL mesh bounding box.
// No assumed vertex count, grid step or axis convention.

function buildMeshMapping(root) {
  root.updateMatrixWorld(true);
  const inv = new THREE.Matrix4().copy(root.matrixWorld).invert();
  const toRoot = new THREE.Matrix4();
  const v = new THREE.Vector3();
  const min = new THREE.Vector3(Infinity, Infinity, Infinity);
  const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);

  root.traverse((o) => {
    if (!o.isMesh || !o.geometry?.attributes?.position) return;
    toRoot.multiplyMatrices(inv, o.matrixWorld);
    const p = o.geometry.attributes.position;
    for (let i = 0; i < p.count; i++) {
      v.fromBufferAttribute(p, i).applyMatrix4(toRoot);
      min.min(v);
      max.max(v);
    }
  });

  const size = max.clone().sub(min);
  const forced = String(viewerInputConfig?.meshUpAxis || "auto").toLowerCase();
  const up = forced === "y" || forced === "z" ? forced : (size.y < size.z ? "y" : "z");
  const gB = up === "y" ? "z" : "y";

  const mapping = {
    up, gA: "x", gB,
    minA: min.x, rangeA: (max.x - min.x) || 1,
    minB: min[gB], rangeB: (max[gB] - min[gB]) || 1,
    rowGrowsWithB: up === "y",
    swap: false, flipU: false, flipV: false,
    calibrated: false,
  };

  console.log("[SASA MINE] Mesh extent (local):", { x: size.x, y: size.y, z: size.z }, "-> up axis:", up, forced !== "auto" ? "(forced by config)" : "(auto)");
  return mapping;
}

function orientUV(a, b, m, o) {
  let u = a;
  let v = m.rowGrowsWithB ? b : 1 - b;
  if (o.swap) [u, v] = [v, u];
  if (o.flipU) u = 1 - u;
  if (o.flipV) v = 1 - v;
  return { u, v };
}

function localToCell(local, m, rows, cols) {
  const a = (local[m.gA] - m.minA) / m.rangeA;
  const b = (local[m.gB] - m.minB) / m.rangeB;
  const { u, v } = orientUV(a, b, m, m);
  if (u < -0.01 || u > 1.01 || v < -0.01 || v > 1.01) return null;
  const cu = Math.min(Math.max(u, 0), 1);
  const cv = Math.min(Math.max(v, 0), 1);
  return { row: Math.round(cv * (rows - 1)), col: Math.round(cu * (cols - 1)) };
}

// Picks the orientation (swap / mirror) whose grid elevation best correlates
// with the mesh's own height. Keeps the previous orientation if nothing fits.
function calibrateMeshMapping() {
  if (!activeModel?.mapping || !gridLoaded || !terrainData?.elevation) return;
  const m = activeModel.mapping;
  const root = activeModel.root;
  root.updateMatrixWorld(true);
  const inv = new THREE.Matrix4().copy(root.matrixWorld).invert();
  const toRoot = new THREE.Matrix4();
  const v = new THREE.Vector3();

  let total = 0;
  root.traverse((o) => { if (o.isMesh && o.geometry?.attributes?.position) total += o.geometry.attributes.position.count; });
  const stride = Math.max(1, Math.floor(total / 4000));

  const samples = [];
  root.traverse((o) => {
    if (!o.isMesh || !o.geometry?.attributes?.position) return;
    toRoot.multiplyMatrices(inv, o.matrixWorld);
    const p = o.geometry.attributes.position;
    for (let i = 0; i < p.count; i += stride) {
      v.fromBufferAttribute(p, i).applyMatrix4(toRoot);
      samples.push({ a: (v[m.gA] - m.minA) / m.rangeA, b: (v[m.gB] - m.minB) / m.rangeB, h: v[m.up] });
    }
  });

  const { rows, cols, elevation } = terrainData;
  let best = null;

  for (let s = 0; s < 2; s++) for (let fu = 0; fu < 2; fu++) for (let fv = 0; fv < 2; fv++) {
    const o = { swap: !!s, flipU: !!fu, flipV: !!fv };
    let n = 0, sh = 0, se = 0, shh = 0, see = 0, she = 0;
    for (const p of samples) {
      const { u, v: vv } = orientUV(p.a, p.b, m, o);
      const row = Math.round(Math.min(Math.max(vv, 0), 1) * (rows - 1));
      const col = Math.round(Math.min(Math.max(u, 0), 1) * (cols - 1));
      const e = getGridValue(elevation, row, col, cols);
      if (typeof e !== "number" || !Number.isFinite(e)) continue;
      n++; sh += p.h; se += e; shh += p.h * p.h; see += e * e; she += p.h * e;
    }
    if (n < 10) continue;
    const den = Math.sqrt((n * shh - sh * sh) * (n * see - se * se));
    const r = den > 0 ? (n * she - sh * se) / den : 0;
    if (!best || r > best.r) best = { ...o, r };
  }

  if (best && best.r >= 0.2) {
    Object.assign(m, { swap: best.swap, flipU: best.flipU, flipV: best.flipV, calibrated: true });
    console.log("[SASA MINE] Mesh/grid orientation calibrated:", best);
  } else {
    console.warn("[SASA MINE] Could not confirm mesh/grid orientation (best r =", best?.r, "). Keeping previous orientation.");
  }
}

function worldToGrid(worldPoint) {
  if (!activeModel?.mapping) return null;
  const local = worldPoint.clone();
  activeModel.root.worldToLocal(local);
  const rows = activeModel.gridRows || GRID_ROWS;
  const cols = activeModel.gridCols || GRID_COLS;
  const cell = localToCell(local, activeModel.mapping, rows, cols);
  return cell ? { ...cell, meshRow: cell.row, meshCol: cell.col } : null;
}

// ============================================================================
// RAYCASTER
// ============================================================================

const raycaster =
  new THREE.Raycaster();

const pointer =
  new THREE.Vector2();

let pointerDown = null;


// ============================================================================
// POINTER -> NDC
// ============================================================================

function pointerToNDC(
  event
) {

  const rect =
    renderer.domElement
      .getBoundingClientRect();


  pointer.x =
    (
      (
        event.clientX -
        rect.left
      ) /
      rect.width
    ) *
    2 -
    1;


  pointer.y =
    -(
      (
        event.clientY -
        rect.top
      ) /
      rect.height
    ) *
    2 +
    1;
}


// ============================================================================
// POINTER DOWN
// ============================================================================

renderer.domElement.addEventListener(
  "pointerdown",
  (event) => {

    pointerDown = {

      x:
        event.clientX,

      y:
        event.clientY,
    };
  }
);


// ============================================================================
// POINTER UP = CLICK
// ============================================================================

renderer.domElement.addEventListener(
  "pointerup",
  (event) => {

    if (!pointerDown) {
      return;
    }


    const dx =
      event.clientX -
      pointerDown.x;


    const dy =
      event.clientY -
      pointerDown.y;


    const distance =
      Math.sqrt(
        dx * dx +
        dy * dy
      );


    pointerDown = null;


    // Ignore orbit dragging. Threshold raised from 8px — trackpads
    // commonly jitter past a tight threshold and get misread as a
    // drag, which silently swallows the click with zero feedback.
    if (distance > 14) {
      return;
    }


    inspectTerrain(
      event
    );
  }
);


// ============================================================================
// POINTER MOVE
// ============================================================================

renderer.domElement.addEventListener(
  "pointermove",
  (event) => {

    if (
      !activeModel ||
      !viewportCursorAltEl
    ) {

      return;
    }


    pointerToNDC(
      event
    );


    raycaster.setFromCamera(
      pointer,
      camera
    );


    const hits =
      raycaster.intersectObject(
        activeModel.root,
        true
      );


    if (!hits.length) {

      viewportCursorAltEl.textContent =
        "---";

      return;
    }


    const worldPoint =
      hits[0].point;


    const index =
      worldToGrid(
        worldPoint
      );


    if (index) {

      const data =
        getTerrainDataAt(
          index.row,
          index.col
        );


      if (
        typeof data.elevation ===
        "number"
      ) {

        viewportCursorAltEl.textContent =
          formatValue(
            data.elevation,
            " m"
          );

      }

      else {

        viewportCursorAltEl.textContent =
          formatValue(
            worldPoint.z,
            " rel"
          );
      }

    }

    else {

      viewportCursorAltEl.textContent =
        formatValue(
          worldPoint.z,
          " rel"
        );
    }
  }
);


// ============================================================================
// TERRAIN INSPECTION
// ============================================================================

function inspectTerrain(
  event
) {

  if (!activeModel) {

    setStatus(
      "Load a GLB before inspecting terrain.",
      true
    );

    return;
  }


  pointerToNDC(
    event
  );


  raycaster.setFromCamera(
    pointer,
    camera
  );


  const hits =
    raycaster.intersectObject(
      activeModel.root,
      true
    );


  if (!hits.length) {

    setStatus(
      "No terrain surface at that point — click directly on the visible mesh.",
      true
    );

    return;
  }


  const worldPoint =
    hits[0].point.clone();


  const index =
    worldToGrid(
      worldPoint
    );


  placeMarker(
    worldPoint
  );


  if (!index) {

    showInspection({

      message:
        "3D point detected, but grid mapping was unavailable.",
    });

    return;
  }


  const data =
    getTerrainDataAt(
      index.row,
      index.col
    );


  showInspection({

    ...index,

    ...data,
  });


  console.log(
    "[SASA MINE] TERRAIN INSPECTION",
    {
      index,
      data,
    }
  );
}


// ============================================================================
// SHOW INSPECTION
// ============================================================================

function showInspection(
  data
) {

  if (!inspectionPanelEl) {
    return;
  }


  if (
    typeof data.row === "number" &&
    typeof data.col === "number"
  ) {

    updateAnalysisMapMarker(data.row, data.col);
  }


  inspectionPanelEl.classList.add(
    "has-selection"
  );


  const set =
    (
      element,
      text
    ) => {

      if (element) {
        element.textContent =
          text;
      }
    };


  set(
    inspectionRowEl,
    data.row !== undefined
      ? String(data.row)
      : "---"
  );


  set(
    inspectionColEl,
    data.col !== undefined
      ? String(data.col)
      : "---"
  );


  set(
    inspectionElevationEl,
    formatValue(
      data.elevation,
      " m"
    )
  );


  set(
    inspectionSlopeEl,
    formatValue(
      data.slope,
      "°"
    )
  );


  set(
    inspectionTerrainTypeEl,
    data.terrain !== null &&
      data.terrain !== undefined
      ? String(data.terrain)
      : "---"
  );


  set(
    inspectionConfidenceEl,
    formatValue(
      data.confidence
    )
  );


  set(
    inspectionChangeEl,
    formatValue(
      data.change,
      " m"
    )
  );


  set(
    inspectionBeforeEl,
    formatValue(
      data.before,
      " m"
    )
  );


  set(
    inspectionDuringEl,
    formatValue(
      data.during,
      " m"
    )
  );


  if (data.message) {

    setStatus(
      data.message
    );

  }

  else if (!gridLoaded) {

    setStatus(
      "3D point detected. Load Person 2 grid JSON for terrain analytics."
    );

  }

  else {

    setStatus(
      `Grid point selected: row ${data.row}, col ${data.col}`
    );
  }
}


// ============================================================================
// CLEAR INSPECTION
// ============================================================================

function clearInspection() {

  inspectionPanelEl?.classList.remove(
    "has-selection"
  );

  markerWorldPoint =
    null;

  terrainMarkerEl?.classList.remove(
    "is-visible"
  );
}


// ============================================================================
// ANALYSIS MAP — 2D top-down elevation raster
// Rebuilt per request: a flat colored view of the same elevation grid
// the 3D terrain uses, with a marker for the last 3D click, and a
// draggable rectangle for region selection (see Region Analysis below).
// Reuses the same rainbow ramp as the Elevation Color 3D toggle for
// visual consistency between the two views.
// ============================================================================

function drawAnalysisMap() {

  if (!analysisMapCanvasEl) {
    return;
  }

  const hasData = gridLoaded && terrainData?.elevation;

  analysisMapCardEl?.classList.toggle("is-empty", !hasData);

  if (!hasData) {
    return;
  }

  const rows = terrainData.rows;
  const cols = terrainData.cols;

  const range = scanGridRange(terrainData.elevation);

  if (!range) {
    analysisMapCardEl?.classList.add("is-empty");
    return;
  }

  // Diagnostic: if this ever prints a low "valid" count relative to
  // total, the canvas will look like a black/near-black box even
  // though hasData was true — that means row/col math or the grid's
  // dimensions don't line up with the actual data, not that the map
  // "isn't working" in general. Cheap to compute once per draw.
  {
    let validCount = 0;

    for (let i = 0; i < terrainData.elevation.length; i++) {
      if (Number.isFinite(terrainData.elevation[i])) validCount++;
    }

    console.log(
      "[SASA MINE] Analysis Map draw:",
      `${rows} x ${cols} grid,`,
      `${terrainData.elevation.length} elevation values,`,
      `${validCount} finite,`,
      `range ${range.min.toFixed(2)}..${range.max.toFixed(2)}`
    );
  }

  analysisMapCanvasEl.width = cols;
  analysisMapCanvasEl.height = rows;

  const ctx = analysisMapCanvasEl.getContext("2d");
  const imageData = ctx.createImageData(cols, rows);
  const span = range.max - range.min || 1;
  const color = new THREE.Color();

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {

      const value = getGridValue(terrainData.elevation, row, col, cols);
      const pixelIndex = (row * cols + col) * 4;

      if (typeof value === "number") {

        const t = Math.min(Math.max((value - range.min) / span, 0), 1);
        // Same rainbow ramp as the 3D Elevation Color toggle.
        color.setHSL(0.66 - t * 0.66, 0.75, 0.5);

        imageData.data[pixelIndex] = Math.round(color.r * 255);
        imageData.data[pixelIndex + 1] = Math.round(color.g * 255);
        imageData.data[pixelIndex + 2] = Math.round(color.b * 255);
        imageData.data[pixelIndex + 3] = 255;

      } else {

        imageData.data[pixelIndex] = 20;
        imageData.data[pixelIndex + 1] = 22;
        imageData.data[pixelIndex + 2] = 20;
        imageData.data[pixelIndex + 3] = 255;
      }
    }
  }

  ctx.putImageData(imageData, 0, 0);
}


function updateAnalysisMapMarker(row, col) {

  if (!analysisMapMarkerEl || !gridLoaded || !terrainData) {
    return;
  }

  const rows = terrainData.rows;
  const cols = terrainData.cols;

  const leftPct = (col / (cols - 1)) * 100;
  const topPct = (row / (rows - 1)) * 100;

  analysisMapMarkerEl.style.left = `${leftPct}%`;
  analysisMapMarkerEl.style.top = `${topPct}%`;
  analysisMapMarkerEl.classList.add("is-visible");
}


// ---- Region select: map a mouse event on the map to a grid cell ----

function mapEventToGridCell(event) {

  if (!analysisMapEl || !gridLoaded || !terrainData) {
    return null;
  }

  const rect = analysisMapEl.getBoundingClientRect();

  const xPct = Math.min(Math.max((event.clientX - rect.left) / rect.width, 0), 1);
  const yPct = Math.min(Math.max((event.clientY - rect.top) / rect.height, 0), 1);

  const col = Math.round(xPct * (terrainData.cols - 1));
  const row = Math.round(yPct * (terrainData.rows - 1));

  return { row, col, xPct, yPct };
}


let regionDragStart = null;

analysisMapEl?.addEventListener("pointerdown", (event) => {

  // Belt-and-suspenders: CSS already sets touch-action:none on
  // #analysis-map so a drag here shouldn't be read as a page
  // gesture, but calling preventDefault directly stops any browser
  // default (text selection, image-drag ghost, etc.) from starting
  // on the very first pointerdown, before CSS touch-action even has
  // a chance to apply on some browsers.
  event.preventDefault();

  const cell = mapEventToGridCell(event);

  if (!cell) {
    return;
  }

  regionDragStart = cell;

  if (analysisMapSelectionEl) {
    analysisMapSelectionEl.classList.add("is-visible");
    analysisMapSelectionEl.style.left = `${cell.xPct * 100}%`;
    analysisMapSelectionEl.style.top = `${cell.yPct * 100}%`;
    analysisMapSelectionEl.style.width = "0%";
    analysisMapSelectionEl.style.height = "0%";
  }

  analysisMapEl.setPointerCapture?.(event.pointerId);
});


analysisMapEl?.addEventListener("pointermove", (event) => {

  if (!regionDragStart || !analysisMapSelectionEl) {
    return;
  }

  const cell = mapEventToGridCell(event);

  if (!cell) {
    return;
  }

  const left = Math.min(regionDragStart.xPct, cell.xPct) * 100;
  const top = Math.min(regionDragStart.yPct, cell.yPct) * 100;
  const width = Math.abs(cell.xPct - regionDragStart.xPct) * 100;
  const height = Math.abs(cell.yPct - regionDragStart.yPct) * 100;

  analysisMapSelectionEl.style.left = `${left}%`;
  analysisMapSelectionEl.style.top = `${top}%`;
  analysisMapSelectionEl.style.width = `${width}%`;
  analysisMapSelectionEl.style.height = `${height}%`;
});


analysisMapEl?.addEventListener("pointerup", (event) => {

  if (!regionDragStart) {
    return;
  }

  const cell = mapEventToGridCell(event);
  const start = regionDragStart;
  regionDragStart = null;

  if (!cell) {
    return;
  }

  const rowMin = Math.min(start.row, cell.row);
  const rowMax = Math.max(start.row, cell.row);
  const colMin = Math.min(start.col, cell.col);
  const colMax = Math.max(start.col, cell.col);

  // Treat a near-zero drag as a click, not a region — clears the
  // selection rather than reporting a meaningless 1-point region.
  if (rowMax - rowMin < 2 && colMax - colMin < 2) {

    clearRegionSelection();
    return;
  }

  computeRegionStats(rowMin, rowMax, colMin, colMax);
});


function clearRegionSelection() {

  regionPanelEl?.classList.remove("has-selection");
  analysisMapSelectionEl?.classList.remove("is-visible");
}


// ---- Region Analysis: aggregate stats over a rectangular selection ----

function computeRegionStats(rowMin, rowMax, colMin, colMax) {

  if (!gridLoaded || !terrainData || !regionPanelEl) {
    return;
  }

  const cols = terrainData.cols;

  let count = 0;
  let changeSum = 0;
  let changeMin = Infinity;
  let changeMax = -Infinity;
  let elevationSum = 0;
  let elevationCount = 0;
  let significantCount = 0;

  for (let row = rowMin; row <= rowMax; row++) {
    for (let col = colMin; col <= colMax; col++) {

      const changeValue = getGridValue(terrainData.change, row, col, cols);
      const elevationValue = getGridValue(terrainData.elevation, row, col, cols);
      const significantValue = getGridValue(terrainData.significantChange, row, col, cols);

      if (typeof changeValue === "number") {

        count++;
        changeSum += changeValue;
        changeMin = Math.min(changeMin, changeValue);
        changeMax = Math.max(changeMax, changeValue);
      }

      if (typeof elevationValue === "number") {

        elevationSum += elevationValue;
        elevationCount++;
      }

      if (significantValue === true || significantValue === 1) {
        significantCount++;
      }
    }
  }

  regionPanelEl.classList.add("has-selection");

  if (regionCountEl) {
    regionCountEl.textContent = String(count);
  }

  if (regionAvgChangeEl) {
    regionAvgChangeEl.textContent = count > 0
      ? formatValue(changeSum / count, " m")
      : "---";
  }

  if (regionMinMaxChangeEl) {
    regionMinMaxChangeEl.textContent = count > 0
      ? `${formatValue(changeMin, " m")} / ${formatValue(changeMax, " m")}`
      : "---";
  }

  if (regionAvgElevationEl) {
    regionAvgElevationEl.textContent = elevationCount > 0
      ? formatValue(elevationSum / elevationCount, " m")
      : "---";
  }

  if (regionSignificantCountEl) {
    regionSignificantCountEl.textContent =
      typeof terrainData.significantChange !== "undefined" && terrainData.significantChange !== null
        ? String(significantCount)
        : "Not available";
  }

  setStatus(
    `Region selected: ${count} points (rows ${rowMin}-${rowMax}, cols ${colMin}-${colMax})`
  );

  console.log(
    "[SASA MINE] REGION ANALYSIS",
    { rowMin, rowMax, colMin, colMax, count, avgChange: count > 0 ? changeSum / count : null }
  );
}


// ============================================================================
// SCREEN MARKER
// ============================================================================

function placeMarker(
  worldPoint
) {

  markerWorldPoint =
    worldPoint.clone();

  updateMarkerScreenPosition();
}


function updateMarkerScreenPosition() {

  if (
    !terrainMarkerEl ||
    !markerWorldPoint
  ) {

    return;
  }


  const projected =
    markerWorldPoint
      .clone()
      .project(camera);


  if (
    projected.z > 1 ||
    projected.z < -1
  ) {

    terrainMarkerEl.classList.remove(
      "is-visible"
    );

    return;
  }


  const rect =
    renderer.domElement
      .getBoundingClientRect();


  const x =
    (
      projected.x *
      0.5 +
      0.5
    ) *
    rect.width +
    rect.left;


  const y =
    (
      -projected.y *
      0.5 +
      0.5
    ) *
    rect.height +
    rect.top;


  terrainMarkerEl.style.left =
    `${x}px`;

  terrainMarkerEl.style.top =
    `${y}px`;


  terrainMarkerEl.classList.add(
    "is-visible"
  );
}


// ============================================================================
// MATERIAL HELPERS
// ============================================================================

function forEachMeshMaterial(
  callback
) {

  if (!activeModel) {
    return;
  }


  activeModel.root.traverse(
    (object) => {

      if (!object.isMesh) {
        return;
      }


      const materials =
        Array.isArray(
          object.material
        )
          ? object.material
          : [object.material];


      materials.forEach(
        (material) => {

          if (material) {

            callback(
              material,
              object
            );
          }

        }
      );
    }
  );
}


// ============================================================================
// WIREFRAME
// ============================================================================

function applyWireframeState() {

  forEachMeshMaterial(
    (material) => {

      material.wireframe =
        wireframeOn;

      material.needsUpdate =
        true;
    }
  );
}


function setWireframe(
  on
) {

  wireframeOn = on;

  applyWireframeState();

  setStatus(
    on
      ? "Wireframe enabled"
      : "Wireframe disabled"
  );
}


// ============================================================================
// HILLSHADE
// ============================================================================

function setHillshade(
  on
) {

  hillshadeOn = on;


  const values =
    on
      ? LIGHT_HILLSHADE
      : LIGHT_BASE;


  ambient.intensity =
    values.ambient;

  hemisphere.intensity =
    values.hemisphere;

  sun.intensity =
    values.sun;


  setStatus(
    on
      ? "Hillshade enabled"
      : "Hillshade disabled"
  );
}


// ============================================================================
// SHARED: COLOR MESH VERTICES FROM A GRID FIELD
// Both slope heatmap and elevation coloring use this same proven
// mesh-vertex-to-grid-index mapping.
// ============================================================================

function colorMeshFromGrid(grid, range, colorFn) {

  if (!activeModel || !grid || !range) {
    return false;
  }

  const span = range.max - range.min || 1;

  activeModel.root.traverse((object) => {

    if (!object.isMesh || !object.geometry?.attributes?.position) {
      return;
    }

    const geometry = object.geometry;
    const positions = geometry.attributes.position;
    const colors = new Float32Array(positions.count * 3);
    const color = new THREE.Color();

    activeModel.root.updateMatrixWorld(true);
    const toRoot = new THREE.Matrix4().multiplyMatrices(
      new THREE.Matrix4().copy(activeModel.root.matrixWorld).invert(),
      object.matrixWorld
    );
    const vtx = new THREE.Vector3();

    for (let i = 0; i < positions.count; i++) {

      vtx.fromBufferAttribute(positions, i).applyMatrix4(toRoot);
      const cell = localToCell(vtx, activeModel.mapping, terrainData.rows, terrainData.cols);
      const value = cell ? getGridValue(grid, cell.row, cell.col, terrainData.cols) : null;

      if (typeof value === "number") {

        const t = Math.min(Math.max((value - range.min) / span, 0), 1);
        colorFn(color, t);

      } else {

        color.setRGB(0.35, 0.37, 0.4);
      }

      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }

    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const materials = Array.isArray(object.material) ? object.material : [object.material];

    materials.forEach((material) => {
      if (material) {
        material.vertexColors = true;
        material.needsUpdate = true;
      }
    });
  });

  return true;
}


// ============================================================================
// ELEVATION COLORING — now an explicit opt-in toggle (Elevation Color
// button), not the default view. Full rainbow ramp — blue (low) ->
// cyan -> green -> yellow -> red (high) — matching the reference
// point-cloud coloring the person asked for, not a muted green-red tint.
// ============================================================================

function applyElevationColorState() {

  if (!activeModel || !gridLoaded || !terrainData?.elevation) {
    return false;
  }

  const range = scanGridRange(terrainData.elevation);

  if (!range) {
    return false;
  }

  return colorMeshFromGrid(terrainData.elevation, range, (color, t) => {
    // hue 0.66 (blue) down to 0 (red) — matches the CSS legend ramp.
    color.setHSL(0.66 - t * 0.66, 0.75, 0.5);
  });
}


// ============================================================================
// SHADING DISPATCHER
// Only one coloring mode is visually active at a time. Default (both
// off) is the plain textured look — coloring is opt-in per the
// person's explicit request, not applied automatically on load.
// ============================================================================

function applyShadingMode() {

  if (!activeModel) {
    return;
  }

  let colored = false;

  if (slopeHeatmapOn) {

    colored = applySlopeColorState();

  } else if (elevationColorOn) {

    colored = applyElevationColorState();
  }

  if (!colored) {

    forEachMeshMaterial((material) => {
      material.vertexColors = false;
      material.needsUpdate = true;
    });
  }
}


// ============================================================================
// SLOPE RANGE
// ============================================================================

function computeSlopeRange() {

  if (slopeRange) {
    return slopeRange;
  }


  if (!terrainData?.slope) {
    return null;
  }


  slopeRange =
    scanGridRange(
      terrainData.slope
    );


  return slopeRange;
}


// ============================================================================
// SLOPE HEATMAP
// ============================================================================

function applySlopeColorState() {

  const range = computeSlopeRange();

  if (!range) {

    slopeHeatmapOn = false;

    [btnSlopeHeatmap, btnToolbarSlopeHeatmap].forEach((button) =>
      button?.classList.remove("is-active")
    );

    setStatus("Slope data unavailable.", true);
    return false;
  }

  return colorMeshFromGrid(terrainData.slope, range, (color, t) => {
    // Blue -> cyan -> yellow -> red
    color.setHSL(0.62 - t * 0.62, 0.75, 0.5);
  });
}


function applySlopeHeatmapState() {

  applyShadingMode();

  if (slopeHeatmapOn) {
    setStatus("Slope heatmap enabled");
  }
}


function setSlopeHeatmap(
  on
) {

  slopeHeatmapOn =
    on;

  if (on && elevationColorOn) {

    elevationColorOn = false;

    [btnElevationColor].forEach((button) =>
      button?.classList.remove("is-active")
    );
  }

  applySlopeHeatmapState();
}


function setElevationColor(on) {

  if (!activeModel || !gridLoaded || !terrainData?.elevation) {

    setStatus(
      "Load grid data containing elevation values first.",
      true
    );

    [btnElevationColor].forEach((button) =>
      button?.classList.remove("is-active")
    );

    elevationColorOn = false;
    return;
  }

  elevationColorOn = on;

  if (on && slopeHeatmapOn) {

    slopeHeatmapOn = false;

    [btnSlopeHeatmap, btnToolbarSlopeHeatmap].forEach((button) =>
      button?.classList.remove("is-active")
    );
  }

  applyShadingMode();

  setStatus(on ? "Elevation color enabled" : "Elevation color disabled");
}


// ============================================================================
// LINKED TOGGLES
// ============================================================================

function wireLinkedToggle(
  buttons,
  callback,
  guard
) {

  const valid =
    buttons.filter(Boolean);


  if (!valid.length) {
    return;
  }


  valid.forEach(
    (button) => {

      button.addEventListener(
        "click",
        () => {

          const currentlyOn =
            valid[0].classList.contains(
              "is-active"
            );


          const nextState =
            !currentlyOn;


          if (
            guard &&
            guard(nextState) === false
          ) {

            return;
          }


          valid.forEach(
            (b) =>
              b.classList.toggle(
                "is-active",
                nextState
              )
          );


          callback(
            nextState
          );
        }
      );
    }
  );
}


// ============================================================================
// SHADING BUTTONS
// ============================================================================

wireLinkedToggle(
  [
    btnWireframe,
    btnToolbarWireframe,
  ],
  setWireframe
);


wireLinkedToggle(
  [
    btnHillshade,
    btnToolbarHillshade,
  ],
  setHillshade
);


wireLinkedToggle(
  [
    btnSlopeHeatmap,
    btnToolbarSlopeHeatmap,
  ],
  setSlopeHeatmap,
  (turningOn) => {

    if (
      turningOn &&
      (
        !gridLoaded ||
        !terrainData?.slope
      )
    ) {

      setStatus(
        "Load grid data containing slope values first.",
        true
      );

      return false;
    }


    return true;
  }
);


btnElevationColor?.addEventListener("click", () => {

  const nextState = !elevationColorOn;
  setElevationColor(nextState);
  btnElevationColor.classList.toggle("is-active", elevationColorOn);
});


// ============================================================================
// TERRAIN TEXTURE / RESET SHADING
// ============================================================================

function resetShading() {

  wireframeOn = false;
  hillshadeOn = false;
  slopeHeatmapOn = false;
  elevationColorOn = false;


  applyWireframeState();

  setHillshade(false);

  applyShadingMode();


  [
    btnWireframe,
    btnToolbarWireframe,
    btnHillshade,
    btnToolbarHillshade,
    btnSlopeHeatmap,
    btnToolbarSlopeHeatmap,
    btnElevationColor,
  ].forEach(
    (button) =>
      button?.classList.remove(
        "is-active"
      )
  );


  setStatus(
    "Terrain appearance reset"
  );
}


btnTerrainTexture?.addEventListener(
  "click",
  resetShading
);


// ============================================================================
// SLIDER FILL
// ============================================================================

function updateSliderFill(
  element
) {

  if (!element) {
    return;
  }


  const min =
    Number(element.min) || 0;

  const max =
    Number(element.max) || 100;


  const value =
    Number(element.value);


  const percentage =
    (
      (
        value - min
      ) /
      (
        max - min
      )
    ) *
    100;


  element.style.setProperty(
    "--range-progress",
    `${percentage}%`
  );
}


// ============================================================================
// Z EXAGGERATION
// ============================================================================

function applyZExaggeration() {

  if (!activeModel) {
    return;
  }


  // Person 3 elevation axis is Z.
  // Scale only Z. X/Y ground coordinates remain unchanged.
  activeModel.root.scale[activeModel.upAxis === "y" ? "y" : "z"] = zExaggeration;


  activeModel.root.updateMatrixWorld(
    true
  );


  // Recalculate model measurements after scaling.
  const box =
    new THREE.Box3()
      .setFromObject(
        activeModel.root
      );


  activeModel.size =
    box.getSize(
      new THREE.Vector3()
    );

  activeModel.center =
    box.getCenter(
      new THREE.Vector3()
    );


  sun.target.position.copy(
    activeModel.center
  );

  sun.target.updateMatrixWorld();

  applySolarPosition();
}


// ============================================================================
// SOLAR AZIMUTH
// ============================================================================

solarAzimuthEl?.addEventListener(
  "input",
  () => {

    solarAzimuthDeg =
      Number(
        solarAzimuthEl.value
      );


    if (
      solarAzimuthValueEl
    ) {

      solarAzimuthValueEl.textContent =
        `${Math.round(
          solarAzimuthDeg
        )}°`;
    }


    updateSliderFill(
      solarAzimuthEl
    );


    applySolarPosition();
  }
);


// ============================================================================
// SOLAR ELEVATION
// ============================================================================

solarElevationEl?.addEventListener(
  "input",
  () => {

    solarElevationDeg =
      Number(
        solarElevationEl.value
      );


    if (
      solarElevationValueEl
    ) {

      solarElevationValueEl.textContent =
        `${Math.round(
          solarElevationDeg
        )}°`;
    }


    updateSliderFill(
      solarElevationEl
    );


    applySolarPosition();
  }
);


// ============================================================================
// Z EXAGGERATION SLIDER
// ============================================================================

zExaggerationEl?.addEventListener(
  "input",
  () => {

    zExaggeration =
      Number(
        zExaggerationEl.value
      );


    if (
      zExaggerationValueEl
    ) {

      zExaggerationValueEl.textContent =
        `${zExaggeration.toFixed(
          1
        )}x`;
    }


    updateSliderFill(
      zExaggerationEl
    );


    applyZExaggeration();
  }
);


updateSliderFill(
  solarAzimuthEl
);

updateSliderFill(
  solarElevationEl
);

updateSliderFill(
  zExaggerationEl
);


// ============================================================================
// BEFORE / AFTER TERRAIN COMPARISON
//
// Person 3 now provides two GLBs of the same site — before mining
// activity and after. Both are exported from the same pipeline, so
// their meshes are expected to share identical vertex counts/order;
// that lets us morph vertex positions directly instead of merely
// crossfading two overlapping models (which would look like a double
// exposure rather than the ground actually changing shape).
//
// The After model is what loadGLBFromURL() already loads as
// activeModel via the normal path (camera framing, shading, click
// inspection — all untouched). This section loads the Before model
// separately, off-scene, purely to read its vertex positions.
// ============================================================================

function clearComparisonState() {

  comparisonEntries = [];
  comparisonReady = false;
  comparisonT = comparisonSliderEl
    ? Number(comparisonSliderEl.value) / 100
    : 1;

  comparisonDockEl?.classList.add("is-hidden");
}


// Loads a GLB purely to read its geometry, without touching the scene,
// activeModel, camera, or any of the existing single-model UI state.
function loadGLBGeometryOnly(url) {

  return new Promise((resolve, reject) => {

    if (!url) {
      reject(new Error("No GLB URL"));
      return;
    }

    loader.load(
      url,
      (gltf) => resolve(gltf.scene),
      undefined,
      (error) => reject(error)
    );
  });
}


// Walks a root in traversal order and returns one entry per mesh with
// a position attribute — the same order prepareModel()/frameModel()
// rely on implicitly, and the same order Person 3's exporter uses
// consistently across a before/after pair from the same pipeline.
function collectMeshPositionArrays(root) {

  const entries = [];

  root.traverse((object) => {

    if (!object.isMesh || !object.geometry?.attributes?.position) {
      return;
    }

    entries.push({
      mesh: object,
      positions: object.geometry.attributes.position,
    });
  });

  return entries;
}


async function loadComparisonPair(afterRoot, beforeUrl) {

  if (!beforeUrl || !afterRoot) {
    return;
  }

  try {

    setStatus("Loading before/after comparison model...");

    const beforeRoot = await loadGLBGeometryOnly(beforeUrl);

    const afterMeshes = collectMeshPositionArrays(afterRoot);
    const beforeMeshes = collectMeshPositionArrays(beforeRoot);

    if (!afterMeshes.length || !beforeMeshes.length) {

      throw new Error(
        "Before/After comparison: one of the models has no mesh geometry."
      );
    }

    if (afterMeshes.length !== beforeMeshes.length) {

      throw new Error(
        `Before/After comparison: mesh count differs (After has ${afterMeshes.length}, Before has ${beforeMeshes.length}).`
      );
    }

    const entries = [];

    for (let i = 0; i < afterMeshes.length; i++) {

      const afterEntry = afterMeshes[i];
      const beforeEntry = beforeMeshes[i];

      if (afterEntry.positions.count !== beforeEntry.positions.count) {

        throw new Error(
          `Before/After comparison: vertex count differs on mesh ${i} (After has ${afterEntry.positions.count}, Before has ${beforeEntry.positions.count}). The two GLBs likely came from different mesh resolutions.`
        );
      }

      // Snapshot both buffers now, independent of the live geometry —
      // the live "after" attribute gets overwritten as the slider
      // moves, so we can't re-read it from the mesh once dragging
      // starts.
      entries.push({
        mesh: afterEntry.mesh,
        afterPositions: Float32Array.from(afterEntry.positions.array),
        beforePositions: Float32Array.from(beforeEntry.positions.array),
      });
    }

    // The before-only scene graph is never added to the renderer and
    // isn't needed once its position data has been copied out.
    beforeRoot.traverse((object) => {
      if (object.isMesh) {
        object.geometry?.dispose();
      }
    });

    comparisonEntries = entries;
    comparisonReady = true;

    comparisonDockEl?.classList.remove("is-hidden");

    if (comparisonSliderEl) {
      comparisonT = Number(comparisonSliderEl.value) / 100;
    }

    applyComparisonT(comparisonT, { recomputeFraming: false });

    setStatus("Before/After comparison ready — drag the slider over the viewport.");

  } catch (error) {

    console.error("[SASA MINE] Before/After comparison unavailable:", error);

    comparisonEntries = [];
    comparisonReady = false;

    comparisonDockEl?.classList.add("is-hidden");

    setStatus(
      `Before/After comparison unavailable: ${error.message}`,
      true
    );
  }
}


// Re-derives every paired mesh's position attribute from the original
// Before/After snapshots (never lerping into an already-lerped
// buffer), so scrubbing the slider back and forth stays exact instead
// of drifting.
function applyComparisonT(t, { recomputeFraming = true } = {}) {

  if (!comparisonReady || !comparisonEntries.length) {
    return;
  }

  const clampedT = Math.min(Math.max(t, 0), 1);

  comparisonEntries.forEach(({ mesh, beforePositions, afterPositions }) => {

    const attribute = mesh.geometry.attributes.position;
    const array = attribute.array;

    for (let i = 0; i < array.length; i++) {

      array[i] =
        beforePositions[i] +
        (afterPositions[i] - beforePositions[i]) * clampedT;
    }

    attribute.needsUpdate = true;

    mesh.geometry.computeVertexNormals();
    mesh.geometry.computeBoundingBox();
    mesh.geometry.computeBoundingSphere();
  });

  comparisonT = clampedT;

  if (comparisonValueEl) {

    comparisonValueEl.textContent =
      `${Math.round(clampedT * 100)}%`;
  }

  // Recompute the overall model bounds/center so lighting and (if the
  // person resets the camera) framing follow the currently displayed
  // shape rather than staying pinned to wherever the After model's
  // bounds originally were. Skipped during the very first application
  // (right after load) since frameModel() already ran once for the
  // After model and re-framing here would fight the person's camera.
  if (recomputeFraming && activeModel) {

    const box = new THREE.Box3().setFromObject(activeModel.root);

    activeModel.size = box.getSize(new THREE.Vector3());
    activeModel.center = box.getCenter(new THREE.Vector3());

    sun.target.position.copy(activeModel.center);
    sun.target.updateMatrixWorld();

    applySolarPosition();

    updateElevationPanels();
  }
}


// Slider drags fire many "input" events per second — recomputing
// normals for a 512x512-ish mesh on every single one is wasted work
// the browser can't keep up with. Collapse to at most one update per
// animation frame.
comparisonSliderEl?.addEventListener("input", () => {

  const t = Number(comparisonSliderEl.value) / 100;

  comparisonT = t;

  if (comparisonUpdatePending) {
    return;
  }

  comparisonUpdatePending = true;

  requestAnimationFrame(() => {

    comparisonUpdatePending = false;
    applyComparisonT(comparisonT);
  });
});


// ============================================================================
// FLYTHROUGH PATH
// ============================================================================

function createFlyPath(
  model
) {

  const { size, center } = model;

  const width = Math.max(size.x, size.y);
  const height = Math.max(size.z, width * 0.12, 20);

  const points = [
    new THREE.Vector3(
      center.x + width * 0.95,
      center.y + width * 0.75,
      center.z + height * 1.8
    ),
    new THREE.Vector3(
      center.x + width * 0.6,
      center.y + width * 0.45,
      center.z + height * 1.15
    ),
    new THREE.Vector3(
      center.x + width * 0.3,
      center.y + width * 0.25,
      center.z + height * 0.7
    ),
    new THREE.Vector3(
      center.x,
      center.y + width * 0.2,
      center.z + height * 0.45
    ),
    new THREE.Vector3(
      center.x - width * 0.45,
      center.y - width * 0.15,
      center.z + height * 0.3
    ),
    new THREE.Vector3(
      center.x - width * 0.25,
      center.y - width * 0.05,
      center.z + height * 0.45
    ),
  ];

  return new THREE.CatmullRomCurve3(
    points,
    false,
    "catmullrom",
    0.5
  );
}

// ============================================================================
// START FLYTHROUGH
// ============================================================================

function startFlythrough() {

  if (!activeModel) {

    setStatus(
      "Load a GLB before starting flythrough.",
      true
    );

    return;
  }


  setAutoRotate(false);


  flyPath =
    createFlyPath(
      activeModel
    );


  flying = true;


  flyClock.start();


  controls.enabled =
    false;


  btnFlythroughStart?.classList.add(
    "is-active"
  );


  btnFlythroughStop?.classList.remove(
    "is-active"
  );


  setRowStatus(
    statusIndicatorCameraEl,
    statusValueCameraEl,
    "FLYTHROUGH",
    "busy"
  );


  setStatus(
    "Flythrough running..."
  );
}


// ============================================================================
// STOP FLYTHROUGH
// ============================================================================

function stopFlythrough(
  showMessage = true
) {

  flying = false;

  flyClock.stop();

  controls.enabled = true;


  btnFlythroughStart?.classList.remove(
    "is-active"
  );


  if (showMessage) {

    btnFlythroughStop?.classList.add(
      "is-active"
    );


    setRowStatus(
      statusIndicatorCameraEl,
      statusValueCameraEl,
      "READY",
      "ok"
    );


    setStatus(
      "Manual navigation"
    );
  }
}


// ============================================================================
// EASING
// ============================================================================

function easeInOut(t) {

  return t < 0.5

    ? 2 * t * t

    : 1 -
    Math.pow(
      -2 * t + 2,
      2
    ) /
    2;
}


// ============================================================================
// UPDATE FLYTHROUGH
// ============================================================================

function updateFlythrough() {

  if (
    !flying ||
    !flyPath ||
    !activeModel
  ) {

    return;
  }


  const cycle =
    FLY_DURATION * 2;


  const elapsed =
    flyClock.getElapsedTime() %
    cycle;


  const rawT =
    elapsed <=
      FLY_DURATION

      ? elapsed /
      FLY_DURATION

      : 2 -
      elapsed /
      FLY_DURATION;


  const t =
    easeInOut(
      rawT
    );


  const position =
    flyPath.getPointAt(
      t
    );


  camera.position.copy(
    position
  );


  // Look slightly ahead.
  const lookT =
    Math.min(
      t + 0.035,
      1
    );


  const lookPoint =
    flyPath.getPointAt(
      lookT
    );


  const target =
    activeModel.center
      .clone()
      .lerp(
        lookPoint,
        0.4
      );


  camera.lookAt(
    target
  );
}


// ============================================================================
// AUTO ROTATE
// ============================================================================

function setAutoRotate(
  on
) {

  autoRotateOn =
    on;

  controls.autoRotate =
    on;


  btnAutoRotate?.classList.toggle(
    "is-active",
    on
  );
}


btnAutoRotate?.addEventListener(
  "click",
  () => {

    if (!activeModel) {

      setStatus(
        "Load a terrain before enabling auto rotate.",
        true
      );

      return;
    }


    setAutoRotate(
      !autoRotateOn
    );


    setStatus(
      autoRotateOn
        ? "Auto rotate enabled"
        : "Auto rotate disabled"
    );
  }
);


// ============================================================================
// MANUAL NAVIGATION
// ============================================================================

function enableManualNavigation() {

  stopFlythrough(false);

  setAutoRotate(false);

  controls.enabled =
    true;


  setRowStatus(
    statusIndicatorCameraEl,
    statusValueCameraEl,
    "MANUAL",
    "ok"
  );


  setStatus(
    "Manual navigation enabled"
  );
}


btnManualNavigation?.addEventListener(
  "click",
  enableManualNavigation
);


// ============================================================================
// RESET CAMERA
// ============================================================================

function resetCamera() {

  if (!activeModel) {

    setStatus(
      "Load a GLB before resetting camera.",
      true
    );

    return;
  }


  stopFlythrough(false);

  setAutoRotate(false);

  controls.enabled =
    true;


  frameModel(
    activeModel
  );


  setRowStatus(
    statusIndicatorCameraEl,
    statusValueCameraEl,
    "READY",
    "ok"
  );


  setStatus(
    "Camera reset"
  );
}


btnCameraResetBottom?.addEventListener(
  "click",
  resetCamera
);


// ============================================================================
// IMAGE -> BACKEND -> GLB
// ============================================================================

async function processImage(
  file
) {

  if (!file) {
    return;
  }


  setStatus(
    `Selected image: ${file.name}`
  );


  // ------------------------------------------------------------
  // No backend configured
  // ------------------------------------------------------------

  if (!BACKEND_ENDPOINT) {

    setStatus(
      "Image selected. Backend endpoint is not configured yet. Load a GLB directly for the viewer.",
      true
    );

    return;
  }


  try {

    showLoading(
      "PROCESSING IMAGE",
      "Uploading image..."
    );


    setStatusBusy(
      "Uploading image..."
    );


    const form =
      new FormData();


    form.append(
      IMAGE_FIELD_NAME,
      file
    );


    const response =
      await fetch(
        BACKEND_ENDPOINT,
        {
          method: "POST",
          body: form,
        }
      );


    if (!response.ok) {

      throw new Error(
        `Backend HTTP ${response.status}`
      );
    }


    updateLoadingProgress(
      50
    );


    setStatusBusy(
      "Processing terrain..."
    );


    const data =
      await response.json();


    const glbURL =
      data[
      RESPONSE_GLB_URL_FIELD
      ];


    if (!glbURL) {

      throw new Error(
        "Backend response did not contain glbUrl."
      );
    }


    updateLoadingProgress(
      75
    );


    await loadGLBFromURL(
      glbURL,
      "Generated Terrain"
    );


    updateLoadingProgress(
      100
    );


    setStatus(
      "Generated terrain ready"
    );

  }

  catch (error) {

    console.error(
      "[SASA MINE] IMAGE PROCESSING ERROR:",
      error
    );


    hideLoading();


    setStatus(
      "Image processing failed.",
      true
    );
  }
}


// ============================================================================
// BUTTON — LOAD GLB
// ============================================================================

btnLoadGlb?.addEventListener(
  "click",
  () => {

    console.log(
      "[SASA MINE] Load GLB button clicked"
    );

    inputGlb?.click();
  }
);


// ============================================================================
// INPUT — GLB
// ============================================================================

inputGlb?.addEventListener(
  "change",
  (event) => {

    const file =
      event.target.files?.[0];


    if (file) {

      loadGLBFromFile(
        file
      );
    }


    // Allows selecting same file again.
    event.target.value =
      "";
  }
);


// ============================================================================
// BUTTON — UPLOAD IMAGE
// ============================================================================

btnUploadImage?.addEventListener(
  "click",
  () => {

    console.log(
      "[SASA MINE] Upload image button clicked"
    );

    inputImage?.click();
  }
);


// ============================================================================
// INPUT — IMAGE
// ============================================================================

inputImage?.addEventListener(
  "change",
  (event) => {

    const file =
      event.target.files?.[0];


    if (file) {

      processImage(
        file
      );
    }


    event.target.value =
      "";
  }
);


// ============================================================================
// BUTTON — LOAD GRID
// ============================================================================

btnLoadGrid?.addEventListener(
  "click",
  () => {

    console.log(
      "[SASA MINE] Load grid button clicked"
    );

    inputGrid?.click();
  }
);


// ============================================================================
// INPUT — GRID
// ============================================================================

inputGrid?.addEventListener(
  "change",
  (event) => {

    const file =
      event.target.files?.[0];


    if (file) {

      loadGridFile(
        file
      );
    }


    event.target.value =
      "";
  }
);


// ============================================================================
// FLYTHROUGH BUTTONS
// ============================================================================

btnFlythroughStart?.addEventListener(
  "click",
  () => {

    console.log(
      "[SASA MINE] START FLYTHROUGH"
    );

    startFlythrough();
  }
);


btnFlythroughStop?.addEventListener(
  "click",
  () => {

    console.log(
      "[SASA MINE] STOP FLYTHROUGH"
    );

    stopFlythrough();
  }
);


// ============================================================================
// DRAG & DROP GLB
// ============================================================================

renderer.domElement.addEventListener(
  "dragover",
  (event) => {

    event.preventDefault();

    event.dataTransfer.dropEffect =
      "copy";
  }
);


renderer.domElement.addEventListener(
  "drop",
  (event) => {

    event.preventDefault();


    const file =
      event.dataTransfer.files?.[0];


    if (!file) {
      return;
    }


    const name =
      file.name.toLowerCase();


    if (
      name.endsWith(".glb") ||
      name.endsWith(".gltf")
    ) {

      loadGLBFromFile(
        file
      );

    }

    else {

      setStatus(
        "Please drop a .glb or .gltf file.",
        true
      );
    }
  }
);


// ============================================================================
// KEYBOARD SHORTCUTS
// ============================================================================

window.addEventListener(
  "keydown",
  (event) => {

    // R = reset
    if (
      event.key.toLowerCase() ===
      "r"
    ) {

      resetCamera();
    }


    // SPACE = flythrough
    if (
      event.code ===
      "Space"
    ) {

      event.preventDefault();


      if (flying) {

        stopFlythrough();

      }

      else {

        startFlythrough();
      }
    }


    // ESC = stop / clear
    if (
      event.key ===
      "Escape"
    ) {

      stopFlythrough();

      clearInspection();
    }
  }
);


// ============================================================================
// RESIZE
// ============================================================================

window.addEventListener(
  "resize",
  () => {

    camera.aspect =
      window.innerWidth /
      window.innerHeight;


    camera.updateProjectionMatrix();


    renderer.setSize(
      window.innerWidth,
      window.innerHeight
    );
  }
);


// ============================================================================
// TELEMETRY
// ============================================================================

function updateOrbitTelemetry() {

  if (
    !viewportAzimuthEl &&
    !viewportPitchEl
  ) {

    return;
  }


  const offset =
    camera.position
      .clone()
      .sub(
        controls.target
      );


  const horizontal = Math.hypot(offset.x, offset.y);
  const pitch = Math.atan2(offset.z, horizontal);
  const azimuth = Math.atan2(offset.y, offset.x);


  if (viewportAzimuthEl) {

    viewportAzimuthEl.textContent =
      `${THREE.MathUtils.radToDeg(azimuth).toFixed(0)}°`;
  }


  if (viewportPitchEl) {

    viewportPitchEl.textContent =
      `${THREE.MathUtils.radToDeg(pitch).toFixed(0)}°`;
  }
}


// ============================================================================
// ANIMATION LOOP
// ============================================================================

function animate() {

  requestAnimationFrame(
    animate
  );


  if (flying) {

    updateFlythrough();

  }

  else {

    controls.update();
  }


  updateOrbitTelemetry();

  updateMarkerScreenPosition();


  renderer.render(
    scene,
    camera
  );
}


// ============================================================================
// INITIALIZATION
// ============================================================================

function initialize() {

  hideLoading();


  // Before any grid data has loaded, show the "load a terrain" note
  // instead of a blank, unpainted (black) canvas.
  analysisMapCardEl?.classList.add("is-empty");


  setRowStatus(
    statusIndicatorModelEl,
    statusValueModelEl,
    "---",
    "idle"
  );


  setRowStatus(
    statusIndicatorGridEl,
    statusValueGridEl,
    "---",
    "idle"
  );


  setRowStatus(
    statusIndicatorRenderEl,
    statusValueRenderEl,
    "READY",
    "ok"
  );


  setRowStatus(
    statusIndicatorCameraEl,
    statusValueCameraEl,
    "---",
    "idle"
  );


  setSystemStatus(
    "READY",
    false,
    false
  );


  if (telemetryModelValueEl) {

    telemetryModelValueEl.textContent =
      "---";
  }


  setStatus(
    "Ready — Load a GLB terrain model"
  );


  console.log(
    "=========================================="
  );

  console.log(
    "SASA MINE 3D TERRAIN VIEWER"
  );

  console.log(
    "Three.js revision:",
    THREE.REVISION
  );

  console.log(
    "Viewer initialized successfully."
  );

  console.log(
    "=========================================="
  );
}


// ============================================================================
// LOAD CONFIGURED TERRAIN
// ============================================================================

let currentAnalysis = null;

function formatModeLabel(mode) {
  return String(mode)
    .replace(/_/g, " ")
    .replace(/\bsrtm\b/i, "SRTM")
    .replace(/\bm$/i, "(m)")
    .replace(/^./, (c) => c.toUpperCase());
}

// Clones the Calibration row so new rows inherit the card's own markup/styling.
function ensureMetadataRow(id, label, afterNode) {
  let row = document.getElementById(id);
  if (row) return row;
  const anchor = metadataCalibrationEl?.parentElement;
  if (!anchor) return null;
  row = anchor.cloneNode(true);
  row.id = id;
  const valueEl = row.querySelector("#metadata-calibration");
  if (!valueEl) return null;
  valueEl.removeAttribute("id");
  valueEl.setAttribute("data-metric-value", "1");
  const labelEl = Array.from(row.children).find((c) => c !== valueEl && !c.contains(valueEl));
  if (labelEl) labelEl.textContent = label;
  (afterNode || anchor).after(row);
  return row;
}

function updateMetricsRows() {
  const a = currentAnalysis;

  if (metadataCalibrationEl) {
    const base = a?.mode
      ? formatModeLabel(a.mode)
      : (viewerInputConfig.metadata?.calibration || "Relative depth (uncalibrated)");
    metadataCalibrationEl.textContent =
      typeof a?.noiseFloor === "number" ? `${base} | noise floor ${a.noiseFloor.toFixed(1)}` : base;
  }

  const m = a?.metrics;
  const fmt = (x, d) => (typeof x === "number" && Number.isFinite(x) ? x.toFixed(d) : "---");
  const defs = [
    ["metadata-row-rmse", "RMSE", "rmse", 2],
    ["metadata-row-mae", "MAE", "mae", 2],
    ["metadata-row-corr", "Correlation", "correlation", 3],
  ];

  let prev = null;
  for (const [id, label, key, digits] of defs) {
    const row = ensureMetadataRow(id, label, prev);
    if (!row) continue;
    prev = row;
    row.style.display = m ? "" : "none";
    const el = row.querySelector("[data-metric-value]");
    if (!el || !m) continue;
    const after = m.after?.pooled?.[key];
    const before = m.before?.pooled?.[key];
    el.textContent = comparisonReady && typeof before === "number"
      ? `${fmt(before, digits)} -> ${fmt(after, digits)}`
      : fmt(after ?? before, digits);
    el.title = `Whole-image held-out evaluation over ${(m.after ?? m.before).pooled.n ?? "?"} pixels - not a per-point value.` +
      (comparisonReady && typeof before === "number" ? " Shown as before -> after." : "");
  }
}

async function loadAnalysisSource(source) {
  const arrays = await loadNPZ(source.url);
  loadGridObject(arrays);
  if (!gridLoaded) throw new Error("Analysis file could not be parsed.");
  if (arrays.__warnings?.length) console.warn("[SASA MINE] Analysis file warnings:", arrays.__warnings);
  currentAnalysis = {
    mode: getString(arrays, ["mode"]),
    noiseFloor: getScalar(arrays, ["noise_floor"]),
    metrics: getMetrics(arrays),
  };
  updateMetricsRows();
}

function setupAnalysisSourceSwitcher(sources) {
  const anchor = metadataCalibrationEl?.parentElement;
  if (sources.length < 2 || !anchor || document.getElementById("analysis-source-select")) return;
  const select = document.createElement("select");
  select.id = "analysis-source-select";
  select.style.cssText = "width:100%;margin:6px 0;padding:6px;background:transparent;color:inherit;font:inherit;border:1px solid rgba(255,255,255,.2);border-radius:6px;";
  sources.forEach((s, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = s.label || `Source ${i + 1}`;
    opt.style.color = "#000";
    select.appendChild(opt);
  });
  select.addEventListener("change", async () => {
    const s = sources[Number(select.value)];
    try {
      setStatusBusy(`Loading ${s.label}...`);
      await loadAnalysisSource(s);
      setStatus(`Analysis source: ${s.label}`);
    } catch (e) {
      setStatus(`Could not load ${s.label}: ${e.message}`, true);
    }
  });
  anchor.parentElement.insertBefore(select, anchor);
}

async function loadConfiguredTerrain() {

  if (viewerInputConfig?.srtmUrl) {
    await loadSRTMMetadata(viewerInputConfig.srtmUrl);
  }

  const mesh = resolveMeshMode(viewerInputConfig);
  const sources = resolveAnalysisSources(viewerInputConfig);

  console.log("[SASA MINE] GLB after:", mesh.after, "| before:", mesh.before || "(none - 1-GLB mode)");
  console.log("[SASA MINE] Analysis sources:", sources);

  try {

    if (!mesh.after) {
      throw new Error("No GLB configured (set glbUrlAfter or glbUrl).");
    }

    // Analysis first, but a failure here must NOT stop the GLB from loading.
    let analysisError = null;

    if (sources.length) {
      setupAnalysisSourceSwitcher(sources);
      try {
        await loadAnalysisSource(sources[0]);
      } catch (error) {
        analysisError = error;
        console.error("[SASA MINE] Analysis load failed:", error);
        gridLoaded = false;
        terrainData = null;
        setRowStatus(statusIndicatorGridEl, statusValueGridEl, "No analysis data", "idle");
      }
    }

    const derivedLabel =
      viewerInputConfig?.metadata?.datasetName || deriveLabelFromUrl(mesh.after);

    const loadedModel = await loadGLBFromURL(mesh.after, derivedLabel);

    if (mesh.twoGlb && loadedModel?.root) {
      await loadComparisonPair(loadedModel.root, mesh.before);
    }

    updateMetricsRows();
    updateElevationPanels();
    hideLoading();

    if (analysisError) {
      setStatus(`Analysis data unavailable: ${analysisError.message}`, false);
    } else {
      setStatus("Terrain READY");
    }

  } catch (error) {

    console.error("[SASA MINE] Failed to load configured terrain:", error);
    hideLoading();
    setStatus(`Could not load terrain: ${error.message}`, true);
  }
}

// ============================================================================
// START VIEWER
// ============================================================================

initialize();

animate();
loadConfiguredTerrain();
window.debugTerrain = () => {
    console.log("ACTIVE MODEL:", activeModel);
    console.log("MODEL SIZE:", activeModel.size);
    console.log("MODEL CENTER:", activeModel.center);
    console.log("MODEL ROOT POSITION:", activeModel.root.position);
    console.log("MODEL ROOT SCALE:", activeModel.root.scale);
    console.log("CAMERA:", camera.position);
    console.log("CONTROLS TARGET:", controls.target);
    console.log("COMPARISON READY:", comparisonReady, "T:", comparisonT);
    scene.traverse((o) => {
    if (o.isMesh) {
        console.log(
            "TERRAIN MESH:",
            o,
            "visible =", o.visible,
            "material =", o.material,
            "frustumCulled =", o.frustumCulled
        );
    }
})
scene.traverse((o) => {
    if (o.isMesh) {
        o.frustumCulled = false;
    }
});;
};




// ============================================================================
// DEBUG / GLOBAL HOOKS
// ============================================================================

window.loadGLBFromFile =
    loadGLBFromFile;

window.loadGLBFromURL =
    loadGLBFromURL;

window.loadGridFile =
    loadGridFile;

window.loadGridObject =
    loadGridObject;

window.startFlythrough =
    startFlythrough;

window.stopFlythrough =
    stopFlythrough;

window.resetCamera =
    resetCamera;

window.inspectTerrain =
    inspectTerrain;

window.worldToGrid =
    worldToGrid;

// Debug: inspect the actual loaded grid data directly in console.
// e.g. window.getTerrainData().rows, window.getGridValue(window.getTerrainData().elevation, 643, 693, window.getTerrainData().cols)
window.getTerrainData =
    () => terrainData;

window.getGridValue =
    getGridValue;

window.getTerrainDataAt =
    getTerrainDataAt;

// Debug: Before/After comparison.
// e.g. window.setComparisonT(0.5) to force the midpoint,
// window.getComparisonState() to inspect readiness.
window.setComparisonT =
    (t) => applyComparisonT(Number(t));

window.getComparisonState =
    () => ({
        ready: comparisonReady,
        t: comparisonT,
        meshCount: comparisonEntries.length,
    });
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