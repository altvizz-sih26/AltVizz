import { defineConfig } from 'vite';
import { resolve } from 'path';

// There was no vite.config.js at all before this. That's fine for `npm run
// dev` (Vite's dev server will happily serve /viewer.html on request), but
// `npm run build` only bundles whatever is listed as an input — by default
// just index.html. Without this, a production build would silently drop
// viewer.html (and src/main.js, the REAL Three.js GLB viewer) and the
// "View 3D Model" link would 404 once deployed.
// (No @vitejs/plugin-react here on purpose — it isn't in package.json, and
// this project was already relying on Vite's built-in esbuild JSX handling
// for .jsx files, which keeps working unchanged with this config.)
export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, 'index.html'),
        viewer: resolve(import.meta.dirname, 'viewer.html'),
      },
    },
  },
});