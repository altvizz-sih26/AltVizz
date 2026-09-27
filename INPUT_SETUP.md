# SASA MINE viewer input setup

## Local test

Edit only:

`src/viewer-input-config.js`

Set:

- `glbUrl`
- `analysisUrl`
- `datasetLabel`
- `sourceLabel`

The analysis URL can point to the `.npz` produced by Person 2.

The viewer reads these arrays from the received NPZ:

- elevation_before
- elevation_after
- confidence_before
- confidence_after
- elevation_change
- class_changed
- combined_confidence
- significant_change

## Backend integration

The viewer does not need its source code changed.

Open it with:

`?glb=<GLB_URL>&analysis=<ANALYSIS_URL>&label=<LABEL>`

URL parameters take priority over the local configuration file.

The viewer therefore does not contain a Windows path, local filename, Person 2 folder name, or Person 3 output filename.

## Important source-data limitation

The supplied Person 2 NPZ contains no explicit terrain-type array and no explicit slope array.

The viewer therefore:
- computes slope from elevation_before/elevation_after data
- does not invent a terrain type
- displays change-related classification when class_changed/significant_change are available
