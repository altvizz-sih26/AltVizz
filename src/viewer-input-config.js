const viewerInputConfig = {
    /*
     * Local testing only.
     *
     * These are relative URLs, not computer-specific file paths.
     * Later, Person 6/backend can replace these values dynamically.
     */

    // ------------------------------------------------------------------
    // BEFORE / AFTER TERRAIN COMPARISON
    //
    // Person 3 now hands over two GLBs for the same site: the terrain
    // state before mining activity and the state after (what used to
    // be a single "during" snapshot). Put each file's URL below.
    //
    // - glbUrlAfter is what the viewer loads and displays by default
    //   (matches prior single-model behavior).
    // - glbUrlBefore is optional. When set, main.js silently loads it
    //   in the background and, if its mesh topology (vertex count per
    //   mesh, in the same traversal order) actually matches the After
    //   model, reveals a Before/After slider docked over the viewport
    //   that morphs vertex positions live between the two states.
    // - If the topologies don't match, the slider stays hidden and a
    //   status message explains why — the After model still loads and
    //   displays normally either way.
    //
    // glbUrl is kept as a legacy alias: if glbUrlAfter is left blank,
    // its value is used instead, so older configs that only set
    // glbUrl keep working unchanged.
    // ------------------------------------------------------------------

    glbUrl: "/urban_depthwizard_3d.glb",

    glbUrlBefore: "/urban_depthwizard_3d.glb",
    glbUrlAfter: "/urban_depthwizard_3d.glb",

    analysisUrl: "public/urban.npz",
    srtmUrl: "public/srtm_urban_aligned.tif",

    metadata: {
        datasetName: null,
        spatialResolution: null,
        coordinates: null,
        horizontalDatum: null,
        verticalDatum: null,
        calibration: null
    }
};

export default viewerInputConfig;