import { DEMO_CONFIG } from '../config/demoConfig';

// Backend base URL. Set VITE_API_BASE in Vercel to your Render URL.
const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

const STATE_KEY = 'depthwizard-demo-job';

const FALLBACK_NOTICE =
  'Live processing is unavailable right now. Showing a sample result.';

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

// Turns any job into a sample-result job so the UI keeps working
function toDemoJob(job, reason) {
  console.warn('Falling back to demo result:', reason);
  return saveJob({
    ...job,
    demoMode: true,
    status: 'completed',
    result: null,
    fallbackReason: String(reason),
  });
}

/**
 * Uploads an image (and optionally a matching SRTM elevation file) to the
 * real backend. If the backend is down, asleep or crashed, we fall back
 * to the demo result instead of showing an error.
 */
export const uploadImage = async (file, srtmFile = null) => {
  const base = buildBaseJob(file);

  const formData = new FormData();
  formData.append('file', file);
  if (srtmFile) formData.append('srtm_file', srtmFile);

  let response;
  try {
    response = await fetchWithTimeout(`${API_BASE}/api/upload`, {
      method: 'POST',
      body: formData,
    });
  } catch (err) {
    // Network error, backend asleep or crashed, timeout
    return toDemoJob({ ...base, demoMode: true }, err.message);
  }

  if (!response.ok) {
    // 5xx = backend trouble -> demo. 4xx = bad user file -> show the real error
    if (response.status >= 500) {
      return toDemoJob({ ...base, demoMode: true }, `Upload ${response.status}`);
    }
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

/**
 * Polls the backend until the job finishes. If the backend crashes or
 * the job fails, we fall back to the demo result.
 */
export const startProcessing = async () => {
  const job = getJob();
  if (!job) throw new Error('Select an image first.');

  // Upload already fell back to demo: nothing to poll
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

      if (data.status === 'completed') {
        return saveJob({ ...job, status: 'completed', result: data.result });
      }
      if (data.status === 'failed') {
        return toDemoJob(job, data.error_message || 'Processing failed');
      }
    } catch (err) {
      consecutiveErrors++;
      // Backend crashed (out of memory) or unreachable: give up after 3 misses
      if (consecutiveErrors >= 3) return toDemoJob(job, err.message);
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }

  return toDemoJob(job, 'Processing timed out');
};

/**
 * Returns the final result in the shape the UI expects.
 * In demo mode, the height-map stats are null and the GLB is the demo asset.
 */
export const getResults = async () => {
  const job = getJob();
  if (!job) {
    return { fileName: DEMO_CONFIG.defaultFileName, fileType: 'image/tiff', status: 'Complete' };
  }

  const result = job.result || {};

  return {
    fileName: job.fileName,
    fileType: job.fileType,
    status: job.status === 'completed' ? 'Complete' : job.status,

    // Set when we fell back to the sample result
    demoMode: !!job.demoMode,
    notice: job.demoMode ? FALLBACK_NOTICE : null,

    // Real backend output
    dsmUrl: result.height_map_path ? `${API_BASE}${result.height_map_path}` : null,
    dsmName: result.height_map_path ? result.height_map_path.split('/').pop() : DEMO_CONFIG.demoDsmName,
    minHeightM: result.min_height_m ?? null,
    maxHeightM: result.max_height_m ?? null,
    meanHeightM: result.mean_height_m ?? null,

    calibrationMode: result.calibration_mode ?? null,
    errorMean: result.error_mean ?? null,
    correlation: result.correlation ?? null,

    // Flythrough isn't generated server-side yet - falls back to the demo GLB
    glbUrl: result.flythrough_path ? `${API_BASE}${result.flythrough_path}` : DEMO_CONFIG.demoGlb,
    glbName: result.flythrough_path ? result.flythrough_path.split('/').pop() : DEMO_CONFIG.demoGlbName,
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