import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import './style.css';

// ==========================================
// SCENE
// ==========================================

const scene = new THREE.Scene();

scene.background = new THREE.Color(0x9bd7ec);


// ==========================================
// CAMERA
// ==========================================

const camera = new THREE.PerspectiveCamera(
  55,
  window.innerWidth / window.innerHeight,
  0.1,
  5000
);

camera.position.set(
  0,
  180,
  300
);


// ==========================================
// RENDERER
// ==========================================

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  powerPreference: 'high-performance'
});

renderer.setSize(
  window.innerWidth,
  window.innerHeight
);

renderer.setPixelRatio(
  Math.min(window.devicePixelRatio, 2)
);

renderer.outputColorSpace =
  THREE.SRGBColorSpace;

renderer.toneMapping =
  THREE.ACESFilmicToneMapping;

renderer.toneMappingExposure = 1.2;

renderer.shadowMap.enabled = true;

renderer.shadowMap.type =
  THREE.PCFSoftShadowMap;

document.body.appendChild(
  renderer.domElement
);


// ==========================================
// ORBIT CONTROLS
// ==========================================

const controls = new OrbitControls(
  camera,
  renderer.domElement
);

controls.enableDamping = true;

controls.dampingFactor = 0.08;

controls.enableRotate = true;

controls.enableZoom = true;

controls.enablePan = true;

controls.minDistance = 10;

controls.maxDistance = 1000;

controls.target.set(
  0,
  0,
  0
);


// ==========================================
// LIGHTING
// ==========================================

// Main ambient light
const ambientLight =
  new THREE.AmbientLight(
    0xffffff,
    2.5
  );

scene.add(ambientLight);


// Hemisphere light
const hemisphereLight =
  new THREE.HemisphereLight(
    0xffffff,
    0x555555,
    2
  );

scene.add(hemisphereLight);


// Main sun
const sun =
  new THREE.DirectionalLight(
    0xffffff,
    4
  );

sun.position.set(
  150,
  300,
  150
);

sun.castShadow = true;

sun.shadow.mapSize.width = 2048;

sun.shadow.mapSize.height = 2048;

sun.shadow.camera.near = 1;

sun.shadow.camera.far = 1000;

scene.add(sun);


// Fill light
const fillLight =
  new THREE.DirectionalLight(
    0xffffff,
    1.5
  );

fillLight.position.set(
  -200,
  150,
  -150
);

scene.add(fillLight);


// ==========================================
// GLB LOADER
// ==========================================

const loader = new GLTFLoader();

loader.load(

  '/urban_depthwizard_3d.glb',

  (gltf) => {

    console.log('================================');
    console.log('🔥 GLB LOADED SUCCESSFULLY');
    console.log('================================');

    const terrain = gltf.scene;


    // ======================================
    // IMPORTANT FIX
    // ======================================
    // Original GLB:
    // X = terrain width
    // Y = terrain depth
    // Z = elevation
    //
    // Rotate +90° so elevation becomes
    // Three.js Y-axis.
    // ======================================

    terrain.rotation.x =
      Math.PI / 2;


    scene.add(terrain);


    // ======================================
    // ORIGINAL BOUNDS
    // ======================================

    const originalBox =
      new THREE.Box3()
        .setFromObject(terrain);

    const originalSize =
      originalBox.getSize(
        new THREE.Vector3()
      );

    const originalCenter =
      originalBox.getCenter(
        new THREE.Vector3()
      );


    console.log(
      'Original Size:',
      originalSize
    );

    console.log(
      'Original Center:',
      originalCenter
    );


    // ======================================
    // CENTER TERRAIN
    // ======================================

    terrain.position.x -=
      originalCenter.x;

    terrain.position.y -=
      originalCenter.y;

    terrain.position.z -=
      originalCenter.z;


    // ======================================
    // SCALE
    // ======================================

    const maxDimension =
      Math.max(
        originalSize.x,
        originalSize.y,
        originalSize.z
      );

    const targetSize = 300;

    const scale =
      targetSize / maxDimension;

    terrain.scale.setScalar(
      scale
    );


    // ======================================
    // MATERIAL SETTINGS
    // ======================================

    terrain.traverse(
      (child) => {

        if (child.isMesh) {

          child.castShadow = true;

          child.receiveShadow = true;

          // Make sure material renders
          // from both sides.
          if (child.material) {

            child.material.side =
              THREE.DoubleSide;

            child.material.needsUpdate =
              true;
          }
        }
      }
    );


    // ======================================
    // FINAL BOUNDS
    // ======================================

    const finalBox =
      new THREE.Box3()
        .setFromObject(terrain);

    const finalSize =
      finalBox.getSize(
        new THREE.Vector3()
      );

    const finalCenter =
      finalBox.getCenter(
        new THREE.Vector3()
      );


    console.log(
      'Final Size:',
      finalSize
    );

    console.log(
      'Final Center:',
      finalCenter
    );


    // ======================================
    // FINAL CENTERING
    // ======================================

    terrain.position.x -=
      finalCenter.x;

    terrain.position.y -=
      finalCenter.y;

    terrain.position.z -=
      finalCenter.z;


    // ======================================
    // CAMERA POSITION
    // ======================================

    camera.position.set(
      0,
      190,
      280
    );


    controls.target.set(
      0,
      0,
      0
    );

    controls.update();


    console.log(
      '🖱 Mouse controls ready'
    );

    console.log(
      '✅ TERRAIN READY'
    );
  },


  // ========================================
  // LOADING PROGRESS
  // ========================================

  (xhr) => {

    if (xhr.total > 0) {

      const percent =
        (xhr.loaded / xhr.total) * 100;

      console.log(
        `Loading: ${percent.toFixed(0)}%`
      );
    }
  },


  // ========================================
  // ERROR
  // ========================================

  (error) => {

    console.error(
      '❌ GLB LOAD ERROR:',
      error
    );
  }

);


// ==========================================
// ANIMATION LOOP
// ==========================================

function animate() {

  requestAnimationFrame(
    animate
  );

  controls.update();

  renderer.render(
    scene,
    camera
  );
}

animate();


// ==========================================
// WINDOW RESIZE
// ==========================================

window.addEventListener(
  'resize',
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