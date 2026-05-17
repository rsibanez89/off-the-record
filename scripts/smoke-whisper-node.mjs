// Smoke test: confirm @huggingface/transformers + onnxruntime-node can load
// the same `_timestamped` Whisper export the browser uses, and that it runs
// inference on a Float32Array of audio. If this works, the integration test
// harness can use this same path; if not, we need Playwright instead.
//
// Run: node scripts/smoke-whisper-node.mjs

import { pipeline, env } from '@huggingface/transformers';

env.allowLocalModels = false;
// Cache weights under the repo so the test run can reuse them across CI runs.
env.cacheDir = './.cache/transformers';

const modelId = 'onnx-community/whisper-tiny.en_timestamped';

console.log(`[smoke] loading ${modelId} on cpu...`);
const t0 = performance.now();
const transcriber = await pipeline(
  'automatic-speech-recognition',
  modelId,
  {
    device: 'cpu',
    dtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' },
  },
);
const loadMs = performance.now() - t0;
console.log(`[smoke] loaded in ${loadMs.toFixed(0)} ms`);

// 1 second of a 440 Hz sine at 16 kHz so Whisper has actual audio to look at
// (silence is most likely to return empty; a tone is more interesting and
// proves the inference path runs end to end without crashing on real data).
const SR = 16_000;
const samples = new Float32Array(SR);
for (let i = 0; i < SR; i++) samples[i] = 0.1 * Math.sin((2 * Math.PI * 440 * i) / SR);

console.log(`[smoke] running inference on 1 s of tone...`);
const t1 = performance.now();
const result = await transcriber(samples, {
  chunk_length_s: 30,
  stride_length_s: 0,
  return_timestamps: 'word',
  no_repeat_ngram_size: 3,
  top_k: 0,
  do_sample: false,
});
const inferMs = performance.now() - t1;
console.log(`[smoke] inference ${inferMs.toFixed(0)} ms`);
console.log(`[smoke] result:`, JSON.stringify(result, null, 2));
