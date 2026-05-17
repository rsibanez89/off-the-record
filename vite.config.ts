import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  statSync,
} from 'node:fs';
import { extname, join, resolve } from 'node:path';

const crossOriginIsolation = {
  name: 'cross-origin-isolation',
  configureServer(server: any) {
    server.middlewares.use((_req: any, res: any, next: any) => {
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
      res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin');
      next();
    });
  },
  configurePreviewServer(server: any) {
    server.middlewares.use((_req: any, res: any, next: any) => {
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
      res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin');
      next();
    });
  },
};

// Serve the onnxruntime-web runtime under `/ort/` directly from node_modules.
// Two reasons we cannot use `public/ort/`:
//   1. Files in `public/` cannot be dynamically `import()`-ed; Vite blocks it.
//      ORT's threaded variant uses `import(...)` to load its `.mjs` loader.
//   2. Sourcing from node_modules in both dev and build guarantees the WASM
//      and the JS we ship come from the same locked package version.
// `silero/sileroVad.ts` sets `ort.env.wasm.wasmPaths = '/ort/'` to point ORT
// at this path. Same-origin, no CDN.
const ORT_DIST = resolve(__dirname, 'node_modules', 'onnxruntime-web', 'dist');
const ORT_URL_PREFIX = '/ort/';
const ORT_OUT_DIR = resolve(__dirname, 'dist', 'ort');

const ortRuntime = {
  name: 'ort-runtime',
  configureServer(server: any) {
    server.middlewares.use((req: any, res: any, next: any) => {
      if (!req.url || !req.url.startsWith(ORT_URL_PREFIX)) return next();
      const filename = req.url.slice(ORT_URL_PREFIX.length).split('?')[0];
      if (!filename) return next();
      const filepath = join(ORT_DIST, filename);
      if (!existsSync(filepath) || statSync(filepath).isDirectory()) return next();
      const ext = extname(filename);
      if (ext === '.wasm') res.setHeader('Content-Type', 'application/wasm');
      else if (ext === '.mjs' || ext === '.js')
        res.setHeader('Content-Type', 'application/javascript');
      res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin');
      res.end(readFileSync(filepath));
    });
  },
  configurePreviewServer(server: any) {
    // After build, ORT lives in dist/ort/ thanks to closeBundle below.
    // Preview just needs to serve it with the right Content-Type for .wasm.
    server.middlewares.use((req: any, res: any, next: any) => {
      if (req.url && req.url.startsWith(ORT_URL_PREFIX) && req.url.endsWith('.wasm')) {
        res.setHeader('Content-Type', 'application/wasm');
      }
      next();
    });
  },
  closeBundle() {
    // Copy the ORT runtime into the production build output so the bundle
    // is self-contained and works behind any static host.
    if (!existsSync(ORT_DIST)) return;
    mkdirSync(ORT_OUT_DIR, { recursive: true });
    for (const f of readdirSync(ORT_DIST)) {
      if (f.endsWith('.wasm') || f.endsWith('.mjs')) {
        copyFileSync(join(ORT_DIST, f), join(ORT_OUT_DIR, f));
      }
    }
  },
};

export default defineConfig({
  plugins: [react(), crossOriginIsolation, ortRuntime],
  worker: {
    format: 'es',
  },
  optimizeDeps: {
    exclude: ['@huggingface/transformers', 'onnxruntime-web'],
  },
});
