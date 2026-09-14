import { Routes, Route, Link, useNavigate } from 'react-router-dom';
import { useEffect, useRef, useState } from 'react';
import Layout from './components/Layout';
import { DEMO_CONFIG } from './config/demoConfig';
import { clearDemoJob, downloadResult, getJob, getResults, startProcessing, uploadImage } from './services/pipelineService';

const stages = ['Image Ingestion', 'Depth Estimation', 'Terrain Classification', 'Reference Elevation', 'Height Calibration', 'DSM Generation', '3D Reconstruction'];
const Arrow = () => <span className="arrow">→</span>;

function Home() {
  return <Layout><section className="hero"><p className="eyebrow">SATELLITE TERRAIN INTELLIGENCE</p><h1>From satellite imagery<br />to <i>3D terrain.</i></h1><p className="lede">Transform a single optical image into elevation-aware terrain and an interactive 3D model.</p><div className="actions"><Link className="button primary" to="/upload">Start Reconstruction <Arrow /></Link><a className="button secondary" href="#workflow">Explore Workflow</a></div></section><section id="workflow" className="section"><p className="eyebrow">THE RECONSTRUCTION PATH</p><h2>Terrain intelligence, layer by layer.</h2><div className="workflow">{['Satellite Image', 'Depth Estimation', 'Terrain Analysis', 'Elevation Calibration', 'DSM', '3D Terrain'].map((x, i) => <div className="flow" key={x}><div className="flow-num">0{i + 1}</div><span>{x}</span>{i < 5 && <Arrow />}</div>)}</div></section><section className="split section"><div><p className="eyebrow">FROM 2D TO TOPOGRAPHY</p><h2>Seeing height where imagery sees colour.</h2></div><p>DepthWizard brings together monocular depth estimation, terrain-aware calibration and reference elevation data to create a meaningful 3D terrain interpretation—without overstating certainty.</p></section></Layout>;
}

function Upload() {
  const nav = useNavigate();
  const [file, setFile] = useState(null);
  const [error, setError] = useState('');
  const [drag, setDrag] = useState(false);
  const [uploading, setUploading] = useState(false); // NEW: real loading state

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

  // FIXED: previously this awaited startProcessing() (which can take up to
  // ~60s against the real backend) before navigating anywhere, so the
  // button just sat there frozen with zero feedback. Now we upload, then
  // navigate immediately to /processing, and let THAT page own waiting
  // for the real result (with an actual visible loading state).
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

function Processing() {
  const nav = useNavigate();
  const job = getJob();
  const [active, setActive] = useState(0);
  const [progress, setProgress] = useState(2);
  const [errorMsg, setErrorMsg] = useState('');

  useEffect(() => {
    if (!job) { nav('/upload'); return; }

    let cancelled = false;

    // Cosmetic progress ticker: we don't get granular real progress from
    // the backend (just pending/processing/completed/failed), so this
    // creeps toward 90% while we wait for the REAL result, then jumps to
    // 100% only once the backend genuinely finishes. It no longer
    // completes on a fixed timer regardless of real status.
    const ticker = setInterval(() => {
      setProgress(p => (p < 90 ? p + 1 : p));
      setActive(a => (a < stages.length - 2 ? a + 1 : a));
    }, 400);

    // This is the REAL call - it polls the actual backend job status
    // until it's completed or failed.
    startProcessing()
      .then(() => {
        if (cancelled) return;
        clearInterval(ticker);
        setProgress(100);
        setActive(stages.length);
        setTimeout(() => nav('/results'), 500);
      })
      .catch(err => {
        if (cancelled) return;
        clearInterval(ticker);
        setErrorMsg(err.message || 'Processing failed.');
      });

    return () => { cancelled = true; clearInterval(ticker); };
  }, []);

  if (errorMsg) {
    return <Layout><section className="page narrow processing"><p className="eyebrow">DEMO PIPELINE · {job?.fileName}</p><h1>Reconstruction failed</h1><p className="error">{errorMsg}</p><button className="button primary" onClick={() => nav('/upload')}>← Try a different image</button></section></Layout>;
  }

  return <Layout><section className="page narrow processing"><p className="eyebrow">PIPELINE · {job?.fileName}</p><h1>{progress === 100 ? 'Reconstruction Complete' : 'Reconstructing terrain…'}</h1><div className="progress"><i style={{ width: progress + '%' }} /></div><div className="progress-label"><span>Overall progress</span><b>{progress}%</b></div><div className="stage-list">{stages.map((s, i) => <div className={'stage ' + (i < active ? 'done' : i === active ? 'active' : '')} key={s}><span className="stage-index">{String(i + 1).padStart(2, '0')}</span><b>{s}</b><em>{i < active ? 'Completed' : i === active ? 'Processing' : 'Waiting'}</em><span className="status">{i < active ? '✓' : i === active ? '◌' : '—'}</span></div>)}</div><p className="demo-note">Waiting on the real backend pipeline — this can take up to a minute depending on your machine.</p></section></Layout>;
}

// Shows the REAL generated height map when available; falls back to the
// decorative placeholder art only if no real result exists yet.
function TerrainArt({ dsmUrl, minHeightM, maxHeightM }) {
  if (dsmUrl) {
    return (
      <div className="terrain-art" aria-label="Generated DSM elevation visualization">
        <img src={dsmUrl} alt="Generated height map" style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: 'inherit' }} />
        {maxHeightM != null && <span>HIGH · {maxHeightM.toFixed(1)}m</span>}
        {minHeightM != null && <small>LOW · {minHeightM.toFixed(1)}m</small>}
      </div>
    );
  }
  return <div className="terrain-art" aria-label="Demo DSM elevation visualization"><div className="contours" /><span>HIGH · 842m</span><small>LOW · 215m</small></div>;
}

function Results() {
  const nav = useNavigate();
  const [modal, setModal] = useState('');
  const [data, setData] = useState(null);

  useEffect(() => { getResults().then(setData); }, []);

  const fresh = () => { clearDemoJob(); nav('/upload'); };

  // "relative" mode means the numbers are an arbitrary uncalibrated
  // guess, not real elevation - flag that clearly rather than presenting
  // it as a confident result.
  const isUncalibrated = data?.calibrationMode === 'relative' || !data?.calibrationMode;

  return <Layout><section className="page"><p className="eyebrow">OUTPUT</p><h1>Reconstruction Complete.</h1><p className="lede">Your terrain package is ready to inspect and explore.</p><div className="input-line"><span>INPUT IMAGE</span><b>{data?.fileName || DEMO_CONFIG.defaultFileName}</b><i>{data?.fileType || 'image/tiff'}</i></div>

    {isUncalibrated && data && (
      <p className="error" style={{ marginBottom: '1rem' }}>
        ⚠ No reference elevation data (SRTM) was provided for this upload — the height values below are an uncalibrated estimate, not real-world elevation.
      </p>
    )}

    <div className="results-grid">
      <article className="result-card">
        <TerrainArt dsmUrl={data?.dsmUrl} minHeightM={data?.minHeightM} maxHeightM={data?.maxHeightM} />
        <div className="card-copy">
          <p className="eyebrow">ELEVATION / DSM</p>
          <h2>Digital Surface Model</h2>
          {data?.meanHeightM != null
            ? <p>Mean height: {data.meanHeightM.toFixed(1)}m {data.calibrationMode && `· mode: ${data.calibrationMode}`}{data.correlation != null && ` · fit correlation: ${data.correlation.toFixed(2)}`}</p>
            : <p>Terrain-relative elevation visualisation generated for this output.</p>
          }
          <button className="button secondary" onClick={() => setModal('dsm')}>View DSM</button>
        </div>
      </article>
      <article className="result-card">
        <div className="model-preview"><span>◒</span><div>GLB<br />TERRAIN</div></div>
        <div className="card-copy">
          <p className="eyebrow">3D TERRAIN MODEL</p>
          <h2>Terrain mesh</h2>
          <p>{data?.glbName || DEMO_CONFIG.demoGlbName} · GLB format{!data?.glbUrl && ' (placeholder — real mesh generation not yet built)'}</p>
          <div className="button-row"><button className="button secondary" onClick={() => setModal('glb')}>View GLB Output</button><button className="icon-button" title="Download GLB" onClick={downloadResult}>↓</button></div>
        </div>
      </article>
      <article className="summary-card">
        <p className="eyebrow">PROCESSING SUMMARY</p>
        <dl>
          <dt>Input format</dt><dd>{data?.fileType || 'image/tiff'}</dd>
          <dt>Output format</dt><dd>DSM + GLB</dd>
          <dt>Reconstruction</dt><dd className="success">{data?.status || 'Complete'}</dd>
          <dt>Calibration</dt><dd>{data?.calibrationMode || 'relative (uncalibrated)'}</dd>
        </dl>
        <a className="button primary full" href="/viewer.html" target="_blank" rel="noopener noreferrer">View 3D Model <Arrow /></a>
      </article>
    </div>
    <button className="text-button back" onClick={fresh}>← Upload New Image</button>
  </section>{modal && <div className="modal-backdrop" onMouseDown={() => setModal('')}><div className="modal" onMouseDown={e => e.stopPropagation()}><button className="close" onClick={() => setModal('')}>×</button>{modal === 'dsm' ? <><TerrainArt dsmUrl={data?.dsmUrl} minHeightM={data?.minHeightM} maxHeightM={data?.maxHeightM} /><h2>Digital Surface Model</h2><p>{isUncalibrated ? 'Uncalibrated estimate — no reference elevation data was used.' : `Calibrated against real elevation data (${data?.calibrationMode} mode).`}</p></> : <><div className="model-preview big">GLB</div><h2>GLB Terrain Output</h2><p>This GLB is ready for download or exploration in the interactive viewer.</p><div className="actions"><button className="button secondary" onClick={downloadResult}>Download GLB</button><button className="button primary" onClick={() => nav('/viewer')}>Open Viewer</button></div></>}</div></div>}</Layout>;
}

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
  // viewer (main.js) with actual GLB models. Flagged as a follow-up - see
  // chat discussion on whether to redirect this route to the real viewer.
  return <Layout><section className="viewer-page"><div className="viewer-title"><div><p className="eyebrow">INTERACTIVE TERRAIN VIEWER</p><h1>Demo terrain model</h1><span className="online">● GLB ready · local demo asset</span></div><button className="text-button" onClick={() => nav('/results')}>← Back to results</button></div><div className="viewer-shell" ref={wrap}><ViewerCanvas top={top} reset={reset} fly={fly} /><div className="viewer-hint">Drag to orbit · scroll to zoom</div><div className="viewer-controls"><button onClick={() => setReset(x => x + 1)}>Reset view</button><button onClick={() => setTop(!top)}>{top ? 'Perspective' : 'Top view'}</button><button className={fly ? 'selected' : ''} onClick={() => setFly(!fly)}>Flythrough</button><button onClick={full}>Fullscreen</button><button onClick={downloadResult}>↓ Download GLB</button></div></div></section></Layout>;
}

export default function App() {
  return <Routes><Route path="/" element={<Home />} /><Route path="/upload" element={<Upload />} /><Route path="/processing" element={<Processing />} /><Route path="/results" element={<Results />} /><Route path="/viewer" element={<Viewer />} /><Route path="*" element={<Home />} /></Routes>;
}