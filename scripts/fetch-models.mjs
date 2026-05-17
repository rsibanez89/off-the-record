#!/usr/bin/env node
// Fetch and verify build-time assets:
//   1. Silero VAD v5 ONNX file (downloaded from Hugging Face, sha256-pinned).
//
// The onnxruntime-web WASM and .mjs loaders are served by a Vite plugin
// (see `vite.config.ts::ortRuntime`) directly from
// `node_modules/onnxruntime-web/dist/`, so no copy step is needed for them.
//
// Run automatically via `npm run predev` and `npm run prebuild`. Idempotent.
// Hash mismatch on the model is a hard failure (we refuse to write a tampered
// file).
//
// Source: huggingface.co/onnx-community/silero-vad (MIT, mirror of upstream
// snakers4/silero-vad).

import { createHash } from 'node:crypto';
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const MODEL_URL =
  'https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx';

// Pinned SHA-256 of the canonical model.onnx as fetched at 2026-05-17.
// Mismatch is a hard failure. To rotate: re-run this script with the new
// hash here, verify the file in a PR.
const MODEL_SHA256 =
  'a4a068cd6cf1ea8355b84327595838ca748ec29a25bc91fc82e6c299ccdc5808';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '..');
const outDir = join(repoRoot, 'public', 'models');
const outPath = join(outDir, 'silero_vad_v5.onnx');

function sha256(buf) {
  return createHash('sha256').update(buf).digest('hex');
}

async function fetchModel() {
  process.stdout.write(`[fetch-models] downloading ${MODEL_URL}\n`);
  const res = await fetch(MODEL_URL, { redirect: 'follow' });
  if (!res.ok) {
    throw new Error(`download failed: ${res.status} ${res.statusText}`);
  }
  const buf = Buffer.from(await res.arrayBuffer());
  process.stdout.write(`[fetch-models] downloaded ${buf.length} bytes\n`);
  return buf;
}

async function ensureModel() {
  if (existsSync(outPath)) {
    const buf = await readFile(outPath);
    const got = sha256(buf);
    if (got === MODEL_SHA256) {
      process.stdout.write(
        `[fetch-models] silero_vad_v5.onnx already present and verified (${MODEL_SHA256.slice(0, 12)}...).\n`,
      );
      return;
    }
    process.stdout.write(
      `[fetch-models] existing file has wrong hash (got ${got}); re-fetching.\n`,
    );
    await rm(outPath, { force: true });
  }

  await mkdir(outDir, { recursive: true });
  const buf = await fetchModel();
  const got = sha256(buf);
  if (got !== MODEL_SHA256) {
    // Hard fail. Do NOT write the bad file to disk: someone could ship it.
    throw new Error(
      `silero_vad_v5.onnx hash mismatch.\n  expected: ${MODEL_SHA256}\n  actual:   ${got}\nrefusing to install a tampered model.`,
    );
  }
  await writeFile(outPath, buf);
  process.stdout.write(
    `[fetch-models] wrote ${outPath} (sha256=${MODEL_SHA256.slice(0, 12)}...)\n`,
  );
}

async function main() {
  await ensureModel();
}

main().catch((err) => {
  process.stderr.write(`[fetch-models] ERROR: ${err.message}\n`);
  process.exit(1);
});
