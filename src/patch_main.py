#!/usr/bin/env python3
"""Usage: python patch_main.py src/main.js
Applies 9 regex-anchored edits. Each must match exactly once or nothing is written.
A backup is saved as main.js.bak."""
import re, sys, shutil

path = sys.argv[1] if len(sys.argv) > 1 else "src/main.js"
src = open(path, encoding="utf-8").read()
edits = []

def edit(name, pattern, repl, flags=re.S):
    edits.append((name, re.compile(pattern, flags), repl))

# 1 -- imports
edit("import loader", r'import \{ loadNPZ \} from "\./npz-loader\.js";',
     'import { loadNPZ, getMetrics, getString, getScalar } from "./npz-loader.js";')
edit("import config", r'import viewerInputConfig from "\./viewer-input-config\.js";',
     'import viewerInputConfig, { resolveAnalysisSources, resolveMeshMode } from "./viewer-input-config.js";')

# 2 -- prepareModel: detect up-axis from the real mesh instead of assuming Z-up
edit("prepareModel orientation",
     r'root\.rotation\.set\(0, 0, 0\);\s*root\.updateMatrixWorld\(true\);',
     r'''root.position.set(0, 0, 0);
  root.rotation.set(0, 0, 0);
  root.updateMatrixWorld(true);

  // Measure the mesh in its own coordinates and detect which axis is "up"
  // (the thinnest one, for a terrain). glTF is normally Y-up; the viewer is Z-up.
  const mapping = buildMeshMapping(root);
  if (mapping.up === "y") {
    root.rotation.x = Math.PI / 2; // Y-up -> Z-up
  }
  root.updateMatrixWorld(true);''')

edit("prepareModel return",
     r'return \{\s*size,\s*center: finalCenter,\s*vertexCount,\s*\};',
     'return { size, center: finalCenter, vertexCount, mapping, upAxis: mapping.up };')

edit("activeModel fields",
     r'vertexCount:\s*measurements\.vertexCount,',
     'vertexCount: measurements.vertexCount,\n              mapping: measurements.mapping,\n              upAxis: measurements.upAxis,')

edit("calibrate after load",
     r'applySlopeHeatmapState\(\);\s*applyZExaggeration\(\);',
     'calibrateMeshMapping();\n            applySlopeHeatmapState();\n            applyZExaggeration();')

# 3 -- worldToGrid + mapping helpers (replaces the hard-coded 512x512 convention)
edit("worldToGrid", r'function worldToGrid\(.*?(?=\n// =+\n// RAYCASTER)', r'''// Mesh <-> grid mapping, derived from the REAL mesh bounding box.
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
''')

# 4 -- colorMeshFromGrid uses the same mapping (fixes partial coverage)
edit("colorMeshFromGrid",
     r'const color = new THREE\.Color\(\);\s*for \(let i = 0; i < positions\.count; i\+\+\) \{.*?value = getGridValue\(grid, row, col, terrainData\.cols\);\s*\}',
     r'''const color = new THREE.Color();

    activeModel.root.updateMatrixWorld(true);
    const toRoot = new THREE.Matrix4().multiplyMatrices(
      new THREE.Matrix4().copy(activeModel.root.matrixWorld).invert(),
      object.matrixWorld
    );
    const vtx = new THREE.Vector3();

    for (let i = 0; i < positions.count; i++) {

      vtx.fromBufferAttribute(positions, i).applyMatrix4(toRoot);
      const cell = localToCell(vtx, activeModel.mapping, terrainData.rows, terrainData.cols);
      const value = cell ? getGridValue(grid, cell.row, cell.col, terrainData.cols) : null;''')

# 5 -- Z exaggeration must scale the mesh's own height axis
edit("z exaggeration", r'activeModel\.root\.scale\.z =\s*zExaggeration;',
     'activeModel.root.scale[activeModel.upAxis === "y" ? "y" : "z"] = zExaggeration;')

# 6 -- recalibrate when a grid arrives after the model
edit("calibrate after grid", r'gridLoaded = true;\s*slopeRange = null;',
     'gridLoaded = true;\n\n    slopeRange = null;\n\n    if (activeModel) calibrateMeshMapping();')

# 7 -- config-driven loading, source switcher, metrics rows
edit("loadConfiguredTerrain", r'async function loadConfiguredTerrain\(\) \{.*?(?=\n// =+\n// START VIEWER)', r'''let currentAnalysis = null;

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
''')

# -- apply
out = src
for name, pat, repl in edits:
    found = pat.findall(out)
    if len(found) != 1:
        sys.exit(f"ABORTED: edit '{name}' matched {len(found)} times (expected 1). Nothing written.")
    out = pat.sub(lambda _m, r=repl: r, out, count=1)

shutil.copyfile(path, path + ".bak")
open(path, "w", encoding="utf-8").write(out)
print(f"OK: {len(edits)} edits applied. Backup: {path}.bak")