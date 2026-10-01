import { DEMO_CONFIG } from '../config/demoConfig';

// Backend base URL. Set VITE_API_BASE in Vercel to your Render URL.
const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';
const DEMO_FALLBACK_ENABLED = import.meta.env.VITE_ENABLE_DEMO_FALLBACK === 'true';

const STATE_KEY = 'depthwizard-demo-job';

export const getJob = () => JSON.parse(sessionStorage.getItem(STATE_KEY) || 'null');

function saveJob(job) {
  sessionStorage.setItem(STATE_KEY, JSON.stringify(job));
  return job;
}

// Call once when the app opens: wakes a sleeping free Render instance
export const warmUpBackend = () => {
  fetch(`${API_BASE}/health`).catch(() => {});
};

async function fetchWithTimeout(url, options = {}, ms = 90000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), ms);
  try {
    return await fetch(url, { ...options, signal: ctrl.signal });
  } finally {
    clearTimeout(timer);
  }
}

function buildBaseJob(file) {
  return {
    backendJobId: null,
    fileName: file.name,
    fileType: file.type || `image/${file.name.split('.').pop()}`,
    fileSize: file.size,
    previewUrl: file.type.startsWith('image/') && !/tiff/i.test(file.name)
      ? URL.createObjectURL(file)
      : null,
    createdAt: Date.now(),
  };
}

/**
 * Uploads an image (and optionally a matching SRTM elevation file) to the
 * real backend.
 */
export const uploadImage = async (file, srtmFile = null, location = null) => {
  const base = buildBaseJob(file);

  const formData = new FormData();
  formData.append('file', file);
  if (srtmFile) formData.append('srtm_file', srtmFile);
  if (location) {
    formData.append('center_lat', String(location.centerLat));
    formData.append('center_lon', String(location.centerLon));
    formData.append('ground_width_m', String(location.groundWidthM));
  }

  let response;
  try {
    response = await fetchWithTimeout(`${API_BASE}/api/upload`, {
      method: 'POST',
      body: formData,
    });
  } catch (err) {
    throw new Error(`Could not reach the processing backend: ${err.message}`);
  }

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Upload failed (${response.status})`);
  }

  const data = await response.json();
  return saveJob({
    ...base,
    backendJobId: data.id,
    demoMode: false,
    status: data.status,
  });
};

export const uploadComparison = async (before, after, srtmFile = null, location = null) => {
  const base = buildBaseJob(before);
  const formData = new FormData();
  formData.append('before_file', before);
  formData.append('after_file', after);
  if (srtmFile) formData.append('srtm_file', srtmFile);
  if (location) {
    formData.append('center_lat', String(location.centerLat));
    formData.append('center_lon', String(location.centerLon));
    formData.append('ground_width_m', String(location.groundWidthM));
  }

  let response;
  try {
    response = await fetchWithTimeout(`${API_BASE}/api/upload/compare`, {
      method: 'POST',
      body: formData,
    });
  } catch (err) {
    throw new Error(`Could not reach the processing backend: ${err.message}`);
  }

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Upload failed (${response.status})`);
  }

  const data = await response.json();
  return saveJob({
    ...base,
    fileName: `${before.name} / ${after.name}`,
    afterFileName: after.name,
    kind: data.kind || 'compare',
    backendJobId: data.id,
    demoMode: false,
    status: data.status,
    stage: data.stage,
    progress: data.progress ?? 0,
  });
};

/**
 * Polls the backend until the job finishes or reports a real failure.
 */
export const startProcessing = async (onProgress = () => {}) => {
  const job = getJob();
  if (!job) throw new Error('Select an image first.');

  // Upload already fell back to demo: nothing to poll
  if (job.demoMode && !DEMO_FALLBACK_ENABLED) {
    throw new Error('This saved job is a sample result. Demo fallback is disabled.');
  }
  if (job.demoMode) {
    await new Promise((r) => setTimeout(r, 1500)); // short "processing" feel
    return job;
  }
  if (!job.backendJobId) throw new Error('Select an image first.');

  const maxAttempts = 60;   // ~2 minutes at 2s intervals
  const intervalMs = 2000;
  let consecutiveErrors = 0;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      const response = await fetchWithTimeout(
        `${API_BASE}/api/jobs/${job.backendJobId}`, {}, 15000
      );
      if (!response.ok) throw new Error(`Status check ${response.status}`);

      consecutiveErrors = 0;
      const data = await response.json();
      onProgress(data);
      saveJob({
        ...job,
        status: data.status,
        stage: data.stage,
        progress: data.progress ?? job.progress ?? 0,
      });

      if (data.status === 'completed') {
        return saveJob({
          ...job,
          status: 'completed',
          stage: data.stage,
          progress: data.progress ?? 100,
          result: data.result,
        });
      }
      if (data.status === 'failed') {
        const error = new Error(data.error_message || 'Processing failed');
        error.jobFailure = true;
        throw error;
      }
    } catch (err) {
      if (err.jobFailure) throw err;
      consecutiveErrors++;
      // Backend crashed (out of memory) or unreachable: give up after 3 misses
      if (consecutiveErrors >= 3) {
        throw new Error(`Could not read processing status: ${err.message}`);
      }
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }

  throw new Error('Processing timed out before the backend completed the job.');
};

/**
 * Returns the final result in the shape the UI expects.
 * In demo mode, the height-map stats are null and the GLB is the demo asset.
 */
export const getResults = async () => {
  const job = getJob();
  if (!job) {
    if (!DEMO_FALLBACK_ENABLED) {
      throw new Error('No completed processing job is available.');
    }
    return {
      fileName: DEMO_CONFIG.defaultFileName,
      fileType: 'image/tiff',
      kind: 'single',
      status: 'Complete',
      demoMode: true,
      glbUrl: DEMO_CONFIG.demoGlb,
      glbName: DEMO_CONFIG.demoGlbName,
    };
  }
  if (job.demoMode && !DEMO_FALLBACK_ENABLED) {
    throw new Error('This saved job is a sample result. Demo fallback is disabled.');
  }

  const result = job.result || {};
  const derivedAnalysisPath = result.flythrough_path?.replace(/_terrain\.glb$/i, '_analysis.npz');
  const analysisPath = result.analysis_path || (derivedAnalysisPath !== result.flythrough_path ? derivedAnalysisPath : null);
  let fitReport = null;
  let changeSummary = null;
  try {
    fitReport = result.fit_report ? JSON.parse(result.fit_report) : null;
  } catch (error) {
    console.warn('Could not parse calibration report:', error);
  }
  try {
    changeSummary = result.change_summary ? JSON.parse(result.change_summary) : null;
  } catch (error) {
    console.warn('Could not parse comparison summary:', error);
  }

  return {
    fileName: job.fileName,
    fileType: job.fileType,
    kind: job.kind || 'single',
    status: job.status === 'completed' ? 'Complete' : job.status,

    // Set when we fell back to the sample result
    demoMode: !!job.demoMode,
    notice: null,

    // Real backend output
    dsmUrl: result.height_map_path ? `${API_BASE}${result.height_map_path}` : null,
    dsmGeotiffUrl: result.dsm_geotiff_path ? `${API_BASE}${result.dsm_geotiff_path}` : null,
    beforeDsmGeotiffUrl: result.before_dsm_geotiff_path ? `${API_BASE}${result.before_dsm_geotiff_path}` : null,
    afterDsmGeotiffUrl: result.after_dsm_geotiff_path ? `${API_BASE}${result.after_dsm_geotiff_path}` : null,
    dsmName: result.height_map_path ? result.height_map_path.split('/').pop() : null,
    minHeightM: result.min_height_m ?? null,
    maxHeightM: result.max_height_m ?? null,
    meanHeightM: result.mean_height_m ?? null,

    calibrationMode: result.calibration_mode ?? null,
    errorMean: result.error_mean ?? null,
    correlation: result.correlation ?? null,
    srtmStatus: result.srtm_status || fitReport?.srtm_status || changeSummary?.srtm_status ||
      (result.calibration_mode === 'relative' ? 'SRTM unavailable; output is relative and not metric.' : null),
    changeSummary,

    glbUrl: result.flythrough_path ? `${API_BASE}${result.flythrough_path}` : job.demoMode ? DEMO_CONFIG.demoGlb : null,
    glbName: result.flythrough_path ? result.flythrough_path.split('/').pop() : job.demoMode ? DEMO_CONFIG.demoGlbName : null,
    analysisUrl: analysisPath ? `${API_BASE}${analysisPath}` : null,
    glbBeforeUrl: result.before_flythrough_path ? `${API_BASE}${result.before_flythrough_path}` : null,
  };
};

export const clearDemoJob = () => sessionStorage.removeItem(STATE_KEY);

/**
 * Downloads the GLB if available, otherwise the DSM image.
 */
export const downloadResult = async () => {
  const results = await getResults();
  const url = results.glbUrl || results.dsmUrl;
  const filename = results.glbUrl ? results.glbName : results.dsmName;

  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
};