// Like smoke-fixture.mjs but uses whisper-base.en (a stronger model) so the
// output is a higher-confidence reference for picking a ground-truth string.
// Run: node scripts/smoke-fixture-base.mjs <wav-path>

import { pipeline, env } from '@huggingface/transformers';
import { readFileSync } from 'node:fs';
import wavefile from 'wavefile';
const { WaveFile } = wavefile;

env.allowLocalModels = false;
env.cacheDir = './.cache/transformers';

const wavPath = process.argv[2];
if (!wavPath) {
  console.error('usage: node scripts/smoke-fixture-base.mjs <wav-path>');
  process.exit(1);
}

const wav = new WaveFile(readFileSync(wavPath));
wav.toBitDepth('32f');
const raw = wav.getSamples();
const samples = Array.isArray(raw) ? raw[0] : raw;
console.log(`[smoke] ${wavPath}: ${(samples.length / wav.fmt.sampleRate).toFixed(2)} s`);

const transcriber = await pipeline(
  'automatic-speech-recognition',
  'onnx-community/whisper-base.en_timestamped',
  { device: 'cpu', dtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' } },
);
const result = await transcriber(samples, {
  chunk_length_s: 30,
  stride_length_s: 0,
  return_timestamps: 'word',
  no_repeat_ngram_size: 3,
  top_k: 0,
  do_sample: false,
});
console.log('[smoke base.en] transcript:', result.text);
