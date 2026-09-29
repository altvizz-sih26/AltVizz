const viewerInputConfig = {
    // ---- MESHES: both paths supported -------------------------------------
    // 2-GLB mode: set glbUrlBefore AND glbUrlAfter (Before/After slider).
    // 1-GLB mode: set only glbUrlAfter (or legacy glbUrl); leave glbUrlBefore "".
    glbUrl: "",            // legacy alias for glbUrlAfter
    meshUpAxis: "auto",    // "auto" | "y" | "z" — which GLB axis is height
    glbUrlBefore: "/before (2).glb",
    glbUrlAfter: "/during (1).glb",

    // ---- ANALYSIS ZIP(S) --------------------------------------------------
    // Packed .npz or zip of loose .npy files; both work.
    // One entry = loaded directly. Several = viewer shows a switcher.
    analysisSources: [
        { label: "SRTM-calibrated", url: "/santa-rosa-wildfire_00000342_srtm_finetuned.npz" },
        { label: "Height above ground (m)", url: "/santa-rosa-wildfire_00000342_meters_finetuned.npz" }
    ],
    analysisUrl: "",       // legacy single-URL alias, used only if analysisSources is empty

    srtmUrl: "/srtm_santa-rosa-wildfire_00000342_pre_disaster_aligned copy.tif",

    // Optional override if metrics ever come from outside the zip:
    // a JSON URL shaped like { before:{pooled:{...}}, after:{pooled:{...}} }
    metricsUrl: "",

    metadata: {
        datasetName: null,
        spatialResolution: null,
        coordinates: null,
        horizontalDatum: null,
        verticalDatum: null,
        calibration: null
    }
};

export function resolveAnalysisSources(cfg = viewerInputConfig) {
    if (Array.isArray(cfg.analysisSources) && cfg.analysisSources.length) return cfg.analysisSources;
    return cfg.analysisUrl ? [{ label: "Analysis", url: cfg.analysisUrl }] : [];
}

export function resolveMeshMode(cfg = viewerInputConfig) {
    const after = cfg.glbUrlAfter || cfg.glbUrl;
    return { after, before: cfg.glbUrlBefore || null, twoGlb: Boolean(cfg.glbUrlBefore && after) };
}

export default viewerInputConfig;