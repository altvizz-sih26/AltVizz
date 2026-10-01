import { Routes, Route, Link, useNavigate } from 'react-router-dom';
import { useEffect, useRef, useState } from 'react';
import Layout from './components/Layout.jsx';
import { DEMO_CONFIG } from './config/demoConfig';
import { clearDemoJob, downloadResult, getJob, getResults, startProcessing, uploadComparison, uploadImage, warmUpBackend } from './services/pipelineService';

const singleStages = ['Image Ingestion', 'Depth Estimation', 'Terrain Classification', 'Reference Elevation', 'Height Calibration', 'DSM Generation', '3D Reconstruction'];
const compareStages = ['Before reconstruction', 'Image alignment', 'After reconstruction', 'Elevation comparison', 'Saving comparison'];
const Arrow = () => <span className="arrow">→</span>;

function Home() {
  return <Layout>
    <section className="hero">
      <p className="eyebrow">SATELLITE TERRAIN INTELLIGENCE</p>
      <h1>From satellite imagery<br />to <i>3D terrain.</i></h1>
      <p className="lede">Transform a single optical image into elevation-aware terrain and an interactive 3D model.</p>
      <div className="actions">
        <Link className="button primary" to="/upload">Start Reconstruction <Arrow /></Link>
        <a className="button secondary" href="#workflow">Explore Workflow</a>
      </div>
    </section>

    <section id="workflow" className="section">
      <p className="eyebrow">THE RECONSTRUCTION PATH</p>
      <h2>Terrain intelligence, layer by layer.</h2>
      <div className="workflow-card">
        <div className="workflow">
          {['Satellite Image', 'Depth Estimation', 'Terrain Analysis', 'Elevation Calibration', 'DSM', '3D Terrain'].map((x, i) =>
            <div className="flow" key={x}>
              <div className="flow-num">0{i + 1}</div>
              <span>{x}</span>
              {i < 5 && <Arrow />}
            </div>
          )}
        </div>
      </div>
    </section>

    <section className="section">
      <p className="eyebrow">CHOOSE YOUR PATH</p>
      <h2>Two ways to reconstruct terrain.</h2>
      <div className="mode-grid">
        <Link className="mode-card" to="/upload">
          <span className="mode-icon">◭</span>
          <p className="eyebrow">FULL PIPELINE</p>
          <h3>Terrain Analysis Suite</h3>
          <p>Upload a single image, run depth estimation, calibrate with SRTM reference elevation, and inspect the DSM and 3D mesh.</p>
          <b>Start analysis <Arrow /></b>
        </Link>
        <Link className="mode-card" to="/compare">
          <span className="mode-icon">◒</span>
          <p className="eyebrow">COMPARATIVE ANALYSIS</p>
          <h3>Before / After Terrain Compare</h3>
          <p>Upload two images of the same site and compare elevation, DSM and 3D terrain side by side over time.</p>
          <b>Start comparison <Arrow /></b>
        </Link>
      </div>
    </section>

    <section className="split section">
      <div>
        <p className="eyebrow">FROM 2D TO TOPOGRAPHY</p>
        <h2>Seeing height where imagery sees colour.</h2>
      </div>
      <p>DepthWizard brings together monocular depth estimation, terrain-aware calibration and reference elevation data to create a meaningful 3D terrain interpretation—without overstating certainty.</p>
    </section>
  </Layout>;
}


function Upload() {
  const nav = useNavigate();
  const [file, setFile] = useState(null);
  const [error, setError] = useState('');
  const [drag, setDrag] = useState(false);
  const [uploading, setUploading] = useState(false);

  const pick = f => {
    if (!f) return;
    const ext = f.name.split('.').pop().toLowerCase();
    if (!DEMO_CONFIG.acceptedExtensions.includes(ext)) {
      setError('Please select a PNG, JPG, JPEG, TIFF, or TIF image.');
      return;
    }
    setError('');
    setFile(f);
  };

  // Upload, then navigate immediately to /processing, which owns waiting
  // for the real result (with a visible loading state).
  const go = async () => {
    setUploading(true);
    setError('');
    try {
      await uploadImage(file);
      nav('/processing');
    } catch (err) {
      setError(err.message || 'Upload failed. Please try again.');
      setUploading(false);
    }
  };

  return <Layout><section className="page narrow"><p className="eyebrow">STEP 01 / INPUT</p><h1>Upload satellite imagery.</h1><p className="lede">Upload a single optical satellite or remote-sensing image to begin the demo reconstruction.</p><label className={'dropzone ' + (drag ? 'dragging' : '')} onDragOver={e => { e.preventDefault(); setDrag(true) }} onDragLeave={() => setDrag(false)} onDrop={e => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]) }}><input type="file" accept=".png,.jpg,.jpeg,.tiff,.tif" onChange={e => pick(e.target.files[0])} /><div className="upload-icon">⇧</div><b>Drop your image here</b><span>or <u>browse files</u></span><small>PNG, JPG, JPEG, TIFF or TIF · one image only</small></label>{error && <p className="error">{error}</p>}{file && <div className="file-card">{file.type.startsWith('image/') && !/tiff/i.test(file.name) && <img src={URL.createObjectURL(file)} alt="Selected satellite preview" />}<div><p className="eyebrow">SELECTED INPUT</p><b>{file.name}</b><small>{file.type || 'Image file'} · {(file.size / 1024 / 1024).toFixed(2)} MB</small>{/tiff/i.test(file.name) && <small>TIFF selected — browser preview unavailable; it will be preserved for processing.</small>}</div><button onClick={() => setFile(null)} className="text-button">Remove</button></div>}<div className="actions right"><button disabled={!file || uploading} onClick={go} className="button primary">{uploading ? 'Uploading…' : <>Start Processing <Arrow /></>}</button></div></section></Layout>;
}

function CompareUpload() {
  const nav = useNavigate();
  const [before, setBefore] = useState(null);
  const [after, setAfter] = useState(null);
  const [error, setError] = useState('');
  const [dragBefore, setDragBefore] = useState(false);
  const [dragAfter, setDragAfter] = useState(false);
  const [uploading, setUploading] = useState(false);

  const pick = (f, which) => {
    if (!f) return;
    const ext = f.name.split('.').pop().toLowerCase();
    if (!DEMO_CONFIG.acceptedExtensions.includes(ext)) {
      setError('Please select a PNG, JPG, JPEG, TIFF, or TIF image.');
      return;
    }
    setError('');
    which === 'before' ? setBefore(f) : setAfter(f);
  };

  const go = async () => {
    setUploading(true);
    setError('');
    try {
      await uploadComparison(before, after);
      nav('/processing');
    } catch (err) {
      setError(err.message || 'Comparison upload failed. Please try again.');
      setUploading(false);
    }
  };

  const dropzone = (label, file, which, dragging, setDragging) => (
    <label
      className={'dropzone ' + (dragging ? 'dragging' : '')}
      onDragOver={e => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={e => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files[0], which); }}
    >
      <input type="file" accept=".png,.jpg,.jpeg,.tiff,.tif" onChange={e => pick(e.target.files[0], which)} />
      <div className="upload-icon">⇧</div>
      <b>{label}</b>
      <span>or <u>browse files</u></span>
      <small>PNG, JPG, JPEG, TIFF or TIF</small>
    </label>
  );

  const fileCard = (file, which, setFile) => file && (
    <div className="file-card">
      {file.type.startsWith('image/') && !/tiff/i.test(file.name) &&
        <img src={URL.createObjectURL(file)} alt={which + ' preview'} />}
      <div>
        <p className="eyebrow">{which.toUpperCase()} IMAGE</p>
        <b>{file.name}</b>
        <small>{file.type || 'Image file'} · {(file.size / 1024 / 1024).toFixed(2)} MB</small>
      </div>
      <button onClick={() => setFile(null)} className="text-button">Remove</button>
    </div>
  );

  return (
    <Layout>
      <section className="page narrow">
        <p className="eyebrow">STEP 01 / INPUT</p>
        <h1>Upload before &amp; after imagery.</h1>
        <p className="lede">Upload two satellite images of the same site to compare elevation, DSM and terrain change over time.</p>

        <div className="compare-grid">
          <div>
            <p className="eyebrow">BEFORE</p>
            {dropzone('Drop the "before" image here', before, 'before', dragBefore, setDragBefore)}
            {fileCard(before, 'before', setBefore)}
          </div>
          <div>
            <p className="eyebrow">AFTER</p>
            {dropzone('Drop the "after" image here', after, 'after', dragAfter, setDragAfter)}
            {fileCard(after, 'after', setAfter)}
          </div>
        </div>

        {error && <p className="error">{error}</p>}

        <div className="actions right">
          <button disabled={!before || !after || uploading} onClick={go} className="button primary">
            {uploading ? 'Processing…' : <>Start Comparison <Arrow /></>}
          </button>
        </div>
      </section>
    </Layout>
  );
}

function Processing() {
  const nav = useNavigate();
  const job = getJob();
  const [stage, setStage] = useState(job?.stage || 'Waiting for processing');
  const [progress, setProgress] = useState(job?.progress ?? 0);
  const [errorMsg, setErrorMsg] = useState('');
  const stages = job?.kind === 'compare' ? compareStages : singleStages;
  const active = progress >= 100
    ? stages.length
    : Math.min(stages.length - 1, Math.floor(progress / 100 * stages.length));

  useEffect(() => {
    if (!job) { nav('/upload'); return; }

    let cancelled = false;

    // Polls the real backend and keeps its reported stage/progress visible.
    startProcessing(data => {
      if (cancelled) return;
      setStage(data.stage || 'Processing');
      if (Number.isFinite(data.progress)) setProgress(data.progress);
    })
      .then(finishedJob => {
        if (cancelled) return;
        setStage(finishedJob.stage || 'Completed');
        setProgress(finishedJob.progress ?? 100);
        setTimeout(() => nav('/results'), 500);
      })
      .catch(err => {
        if (cancelled) return;
        setErrorMsg(err.message || 'Processing failed.');
      });

    return () => { cancelled = true; };
  }, []);

  if (errorMsg) {
    return <Layout><section className="page narrow processing"><p className="eyebrow">PIPELINE · {job?.fileName}</p><h1>Reconstruction failed</h1><p className="error">{errorMsg}</p><button className="button primary" onClick={() => nav('/upload')}>← Try a different image</button></section></Layout>;
  }

  return <Layout><section className="page narrow processing"><p className="eyebrow">PIPELINE · {job?.fileName}</p><h1>{progress === 100 ? 'Processing Complete' : job?.kind === 'compare' ? 'Comparing terrain…' : 'Reconstructing terrain…'}</h1><div className="progress"><i style={{ width: progress + '%' }} /></div><div className="progress-label"><span>{stage}</span><b>{progress}%</b></div><div className="stage-list">{stages.map((s, i) => <div className={'stage ' + (i < active ? 'done' : i === active ? 'active' : '')} key={s}><span className="stage-index">{String(i + 1).padStart(2, '0')}</span><b>{s}</b><em>{i < active ? 'Completed' : i === active ? 'Processing' : 'Waiting'}</em><span className="status">{i < active ? '✓' : i === active ? '◌' : '—'}</span></div>)}</div><p className="demo-note">Waiting on the real backend pipeline — this can take up to a minute depending on your machine.</p></section></Layout>;
}

// Shows the REAL generated height map when available; falls back to the
// decorative placeholder art only if no real result exists yet.
function TerrainArt({ dsmUrl, minHeightM, maxHeightM, comparison = false }) {
  if (dsmUrl) {
    return (
      <div className="terrain-art" aria-label="Generated DSM elevation visualization">
        <img src={dsmUrl} alt="Generated height map" style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: 'inherit' }} />
        {maxHeightM != null && <span>HIGH · {maxHeightM.toFixed(1)}m</span>}
        {minHeightM != null && <small>LOW · {minHeightM.toFixed(1)}m</small>}
      </div>
    );
  }
  return (
    <div className="terrain-art terrain-art-empty" aria-label={comparison ? 'Before and after DSM GeoTIFF exports' : 'DSM preview unavailable'}>
      <span>{comparison ? 'BEFORE / AFTER' : 'DSM PREVIEW UNAVAILABLE'}</span>
      {comparison && <small>GEOTIFF EXPORTS</small>}
    </div>
  );
}

function Results() {
  const nav = useNavigate();
  const [modal, setModal] = useState('');
  const [data, setData] = useState(null);

  useEffect(() => { getResults().then(setData); }, []);

  const fresh = () => { clearDemoJob(); nav('/upload'); };

  // "relative" mode means the numbers are an arbitrary uncalibrated
  // guess, not real elevation - flag that clearly.
  const isUncalibrated = data?.calibrationMode === 'relative' || (!data?.calibrationMode && data?.kind !== 'compare');
  const relativeComparison = data?.kind === 'compare' && data?.calibrationMode === 'relative';
  const outputFormats = [
    (data?.dsmUrl || data?.dsmGeotiffUrl || data?.beforeDsmGeotiffUrl || data?.afterDsmGeotiffUrl) && 'DSM',
    data?.glbUrl && 'GLB',
  ].filter(Boolean).join(' + ') || 'No output artifacts';

  return <Layout><section className="page"><p className="eyebrow">OUTPUT</p><h1>{data?.kind === 'compare' ? 'Comparison Complete.' : 'Reconstruction Complete.'}</h1><p className="lede">{data?.kind === 'compare' ? 'Before and after terrain outputs are ready to inspect.' : 'Your terrain package is ready to inspect and explore.'}</p><div className="input-line"><span>{data?.kind === 'compare' ? 'BEFORE / AFTER' : 'INPUT IMAGE'}</span><b>{data?.fileName || DEMO_CONFIG.defaultFileName}</b><i>{data?.fileType || 'image/tiff'}</i></div>

    {/* NEW: shown when the live backend failed and we fell back to the sample result */}
    {data?.notice && (
      <p style={{ marginBottom: '1rem', padding: '10px 14px', background: '#fff4d6', color: '#6b4e00', borderRadius: 8, fontSize: 14 }}>
        ℹ {data.notice}
      </p>
    )}

    {/* CHANGED: only show the uncalibrated warning for real (non-demo) results */}
    {isUncalibrated && data && !data.demoMode && data.kind !== 'compare' && (
      <p className="error" style={{ marginBottom: '1rem' }}>
        ⚠ No reference elevation data (SRTM) was provided for this upload — the height values below are an uncalibrated estimate, not real-world elevation.
      </p>
    )}
    {relativeComparison && (
      <p className="demo-note">No SRTM reference was used. Comparison elevations are relative estimates, not absolute real-world heights.</p>
    )}

    <div className="results-grid">
      <article className="result-card">
        <TerrainArt dsmUrl={data?.dsmUrl} minHeightM={data?.minHeightM} maxHeightM={data?.maxHeightM} comparison={data?.kind === 'compare'} />
        <div className="card-copy">
          <p className="eyebrow">ELEVATION / DSM</p>
          <h2>{data?.kind === 'compare' ? 'Before / after DSMs' : 'Digital Surface Model'}</h2>
          {data?.meanHeightM != null
            ? <p>Mean height: {data.meanHeightM.toFixed(1)}m {data.calibrationMode && `· mode: ${data.calibrationMode}`}{data.correlation != null && ` · fit correlation: ${data.correlation.toFixed(2)}`}</p>
            : <p>{data?.kind === 'compare' ? 'GeoTIFF exports are available below. No raster preview was generated.' : 'Terrain-relative elevation visualisation generated for this output.'}</p>
          }
          {data?.dsmUrl && <button className="button secondary" onClick={() => setModal('dsm')}>View DSM</button>}
          <div className="artifact-links">
            {data?.dsmGeotiffUrl && <a className="text-button" href={data.dsmGeotiffUrl} download>Download DSM GeoTIFF</a>}
            {data?.kind === 'compare' && data?.beforeDsmGeotiffUrl && <a className="text-button" href={data.beforeDsmGeotiffUrl} download>Download before DSM GeoTIFF</a>}
            {data?.kind === 'compare' && data?.afterDsmGeotiffUrl && <a className="text-button" href={data.afterDsmGeotiffUrl} download>Download after DSM GeoTIFF</a>}
          </div>
        </div>
      </article>
      <article className="result-card">
        {data?.glbUrl
          ? <div className="model-preview"><span>◒</span><div>GLB<br />TERRAIN</div></div>
          : <div className="model-preview model-preview-empty">GLB NOT GENERATED</div>}
        <div className="card-copy">
          <p className="eyebrow">3D TERRAIN MODEL</p>
          <h2>Terrain mesh</h2>
          <p>{data?.glbUrl ? `${data.glbName} · GLB format` : 'Mesh generation did not produce a GLB for this job.'}</p>
          <div className="button-row"><button className="button secondary" disabled={!data?.glbUrl} onClick={() => setModal('glb')}>View GLB Output</button><button className="icon-button" title="Download GLB" disabled={!data?.glbUrl} onClick={downloadResult}>↓</button></div>
        </div>
      </article>
      <article className="summary-card">
        <p className="eyebrow">PROCESSING SUMMARY</p>
        <dl>
          <dt>Input format</dt><dd>{data?.fileType || 'image/tiff'}</dd>
          <dt>Output format</dt><dd>{outputFormats}</dd>
          <dt>Reconstruction</dt><dd className="success">{data?.status || 'Complete'}</dd>
          <dt>Calibration</dt><dd>{data?.calibrationMode || 'relative (uncalibrated)'}</dd>
        </dl>

        {data?.glbUrl
          ? <a className="button primary full" href={viewerHref(data)} target="_blank" rel="noopener noreferrer">View 3D Model <Arrow /></a>
          : <button className="button primary full" disabled>3D model unavailable</button>}
      </article>
    </div>
    <button className="text-button back" onClick={fresh}>← Upload New Image</button>
  </section>{modal && <div className="modal-backdrop" onMouseDown={() => setModal('')}><div className="modal" onMouseDown={e => e.stopPropagation()}><button className="close" onClick={() => setModal('')}>×</button>{modal === 'dsm' ? <><TerrainArt dsmUrl={data?.dsmUrl} minHeightM={data?.minHeightM} maxHeightM={data?.maxHeightM} /><h2>Digital Surface Model</h2><p>{data?.demoMode ? 'Sample result — live processing was unavailable.' : isUncalibrated ? 'Uncalibrated estimate — no reference elevation data was used.' : `Calibrated against real elevation data (${data?.calibrationMode} mode).`}</p></> : <><div className="model-preview big">GLB</div><h2>GLB Terrain Output</h2><p>{data?.glbUrl ? 'This GLB is ready for download or exploration in the interactive viewer.' : 'Real mesh generation isn\u2019t built server-side yet — showing a demo terrain mesh instead.'}</p><div className="actions"><button className="button primary" onClick={downloadResult}>Download GLB</button></div></>}</div></div>}</Layout>;
}

const viewerHref = d => {
  const q = new URLSearchParams({ glb: d.glbUrl, label: d.glbName });
  if (d.analysisUrl) q.set('analysis', d.analysisUrl);
  if (d.glbBeforeUrl) q.set('glbBefore', d.glbBeforeUrl);
  return `/viewer.html?${q}`;
};

function ViewerCanvas({ top, reset, fly }) {
  const ref = useRef();
  useEffect(() => {
    const c = ref.current, ctx = c.getContext('2d');
    let angle = top ? 0 : .52, zoom = 1, drag = false, last;
    const draw = () => {
      const w = c.clientWidth, h = c.clientHeight;
      c.width = w * devicePixelRatio; c.height = h * devicePixelRatio;
      ctx.scale(devicePixelRatio, devicePixelRatio);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#071228'; ctx.fillRect(0, 0, w, h);
      const cols = 37, rows = 29, pts = [];
      for (let y = 0; y < rows; y++) {
        pts[y] = [];
        for (let x = 0; x < cols; x++) {
          let xx = (x - cols / 2) * 15, yy = (y - rows / 2) * 15, z = (Math.sin(x * .35) * 28 + Math.cos(y * .32) * 23 + Math.sin((x + y) * .19) * 18) * zoom;
          let rx = xx * Math.cos(angle) - yy * Math.sin(angle), ry = xx * Math.sin(angle) + yy * Math.cos(angle);
          pts[y][x] = [w / 2 + rx * zoom, h / 2 + ry * .4 - z * .8];
        }
      }
      for (let y = 0; y < rows - 1; y++) for (let x = 0; x < cols - 1; x++) {
        ctx.beginPath(); ctx.moveTo(...pts[y][x]); ctx.lineTo(...pts[y][x + 1]); ctx.lineTo(...pts[y + 1][x + 1]); ctx.lineTo(...pts[y + 1][x]); ctx.closePath();
        ctx.fillStyle = `hsla(${205 + (y / rows) * 45},65%,${22 + (Math.sin(x * .4 + y * .3) + 1) * 9}%,.95)`;
        ctx.fill(); ctx.strokeStyle = 'rgba(176,209,255,.15)'; ctx.stroke();
      }
      if (fly) angle += .006;
      requestAnimationFrame(draw);
    };
    draw();
    const down = e => { drag = true; last = e.clientX }, move = e => { if (drag) { angle += (e.clientX - last) * .01; last = e.clientX } }, up = () => drag = false, wheel = e => { zoom = Math.max(.55, Math.min(1.6, zoom - e.deltaY * .001)) };
    c.addEventListener('pointerdown', down); c.addEventListener('pointermove', move); addEventListener('pointerup', up); c.addEventListener('wheel', wheel);
    return () => { c.removeEventListener('pointerdown', down); c.removeEventListener('pointermove', move); removeEventListener('pointerup', up); c.removeEventListener('wheel', wheel) };
  }, [top, reset, fly]);
  return <canvas ref={ref} className="terrain-canvas" />;
}

function Viewer() {
  const nav = useNavigate();
  const [top, setTop] = useState(false);
  const [reset, setReset] = useState(0);
  const [fly, setFly] = useState(false);
  const wrap = useRef();
  const full = () => wrap.current.requestFullscreen?.();
  // NOTE: this is still the placeholder 2D canvas, not the real Three.js
  // viewer (main.js) with actual GLB models.
  return <Layout><section className="viewer-page"><div className="viewer-title"><div><p className="eyebrow">INTERACTIVE TERRAIN VIEWER</p><h1>Demo terrain model</h1><span className="online">● GLB ready · local demo asset</span></div><button className="text-button" onClick={() => nav('/results')}>← Back to results</button></div><div className="viewer-shell" ref={wrap}><ViewerCanvas top={top} reset={reset} fly={fly} /><div className="viewer-hint">Drag to orbit · scroll to zoom</div><div className="viewer-controls"><button onClick={() => setReset(x => x + 1)}>Reset view</button><button onClick={() => setTop(!top)}>{top ? 'Perspective' : 'Top view'}</button><button className={fly ? 'selected' : ''} onClick={() => setFly(!fly)}>Flythrough</button><button onClick={full}>Fullscreen</button><button onClick={downloadResult}>↓ Download GLB</button></div></div></section></Layout>;
}

export default function App() {
  // Wake the free Render backend as soon as the site opens
  useEffect(() => { warmUpBackend(); }, []);

  return <Routes><Route path="/" element={<Home />} /><Route path="/upload" element={<Upload />} /><Route path="/compare" element={<CompareUpload />} /><Route path="/processing" element={<Processing />} /><Route path="/results" element={<Results />} /><Route path="/viewer" element={<Viewer />} /><Route path="*" element={<Home />} /></Routes>;
}