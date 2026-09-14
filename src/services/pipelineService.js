import { DEMO_CONFIG } from '../config/demoConfig';

// Backend base URL. Change this if your API runs somewhere else
// (e.g. a deployed URL later).
const API_BASE = 'http://127.0.0.1:8000';

const STATE_KEY = 'depthwizard-demo-job';

export const getJob = () => JSON.parse(sessionStorage.getItem(STATE_KEY) || 'null');

function saveJob(job) {
  sessionStorage.setItem(STATE_KEY, JSON.stringify(job));
  return job;
}

/**
 * Uploads an image (and optionally a matching SRTM elevation file) to the
 * real backend. Kicks off a background processing job on the server side.
 *
 * `srtmFile` is optional - without it, height results come back uncalibrated
 * ("relative" mode, an arbitrary 0-50m scale, NOT real elevation). Pass a
 * real SRTM GeoTIFF for the same location to get calibrated results.
 */
export const uploadImage = async (file, srtmFile = null) => {
  const formData = new FormData();
  formData.append('file', file);
  if (srtmFile) {
    formData.append('srtm_file', srtmFile);
  }

  const response = await fetch(`${API_BASE}/api/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Upload failed (${response.status})`);
  }

  const data = await response.json();

  const job = {
    backendJobId: data.id,
    fileName: file.name,
    fileType: file.type || `image/${file.name.split('.').pop()}`,
    fileSize: file.size,
    previewUrl: file.type.startsWith('image/') && !/tiff/i.test(file.name)
      ? URL.createObjectURL(file)
      : null,
    demoMode: false,
    status: data.status, // "pending"
    createdAt: Date.now(),
  };

  return saveJob(job);
};

/**
 * Polls the backend until the job finishes (completed or failed).
 * Throws if the job fails, so callers can show an error state.
 */
export const startProcessing = async () => {
  const job = getJob();
  if (!job || !job.backendJobId) throw new Error('Select an image first.');

  const maxAttempts = 60;   // ~2 minutes at 2s intervals - CPU inference can be slow
  const intervalMs = 2000;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const response = await fetch(`${API_BASE}/api/jobs/${job.backendJobId}`);
    if (!response.ok) throw new Error(`Failed to check job status (${response.status})`);

    const data = await response.json();

    if (data.status === 'completed') {
      return saveJob({ ...job, status: 'completed', result: data.result });
    }
    if (data.status === 'failed') {
      saveJob({ ...job, status: 'failed', errorMessage: data.error_message });
      throw new Error(data.error_message || 'Processing failed.');
    }

    // still pending/processing - wait and poll again
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }

  throw new Error('Processing timed out. The job may still finish - check back later.');
};

/**
 * Returns the final result in the shape the UI expects.
 *
 * IMPORTANT: `glbName` is NOT a real generated 3D flythrough yet - that
 * step isn't built server-side. It falls back to one of the pre-made demo
 * GLBs in /public so the viewer still has something to show. `dsmName`
 * DOES point at a real generated height map from the backend.
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

    // Real backend output
    dsmUrl: result.height_map_path ? `${API_BASE}${result.height_map_path}` : null,
    dsmName: result.height_map_path ? result.height_map_path.split('/').pop() : DEMO_CONFIG.demoDsmName,
    minHeightM: result.min_height_m ?? null,
    maxHeightM: result.max_height_m ?? null,
    meanHeightM: result.mean_height_m ?? null,

    // Calibration quality - "relative" mode means these numbers are NOT
    // real-world calibrated; only trust them in "global"/"terrain_aware" mode.
    calibrationMode: result.calibration_mode ?? null,
    errorMean: result.error_mean ?? null,
    correlation: result.correlation ?? null,

    // Flythrough isn't generated server-side yet - fall back to a static demo asset
    glbUrl: result.flythrough_path ? `${API_BASE}${result.flythrough_path}` : DEMO_CONFIG.demoGlb,
    glbName: result.flythrough_path ? result.flythrough_path.split('/').pop() : DEMO_CONFIG.demoGlbName,
  };
};

export const clearDemoJob = () => sessionStorage.removeItem(STATE_KEY);

/**
 * Downloads the real generated GLB mesh if available, otherwise falls
 * back to the placeholder demo GLB.
 *
 * FIXED: this used to check `results.dsmUrl` first, so clicking
 * "Download GLB" almost always downloaded the DSM height-map PNG instead
 * of an actual .glb file (dsmUrl is present whenever the backend has run,
 * while glbUrl was only ever the demo fallback). This function is only
 * ever wired to "Download GLB" buttons, so it should always prefer the
 * GLB output over the DSM image.
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