export const DEMO_CONFIG = {
  demoMode: true,
  // NOTE: these used to point at /demo/depthwizard-demo-*.* which does not
  // exist anywhere in /public - every "View GLB Output" / download click
  // was 404ing silently (fetch/img/anchor to a missing file just renders
  // nothing, no visible error). Pointed at the real demo assets that DO
  // ship in /public instead.
  assetDirectory: '/',
  demoDsm: '/depthwizard_bare_terrain_3d.glb',
  demoGlb: '/depthwizard_bare_terrain_3d.glb',
  demoDsmName: 'depthwizard_bare_terrain_3d.glb',
  demoGlbName: 'depthwizard_bare_terrain_3d.glb',
  defaultFileName: 'demo_satellite_observation.tif',
  acceptedExtensions: ['png', 'jpg', 'jpeg', 'tiff', 'tif']
};