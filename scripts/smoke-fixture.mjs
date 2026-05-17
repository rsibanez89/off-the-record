// Quick verification that a candidate fixture WAV is something tiny.en can
// transcribe cleanly enough to use as an integration test. Prints the
// transcript so a human can sanity-check before committing the fixture.
//
// Run: node scripts/smoke-fixture.mjs <path-to-16k-mono-wav>

import { pipeline, env } from '@huggingface/transformers';
import { readFileSync } from 'node:fs';
import wavefile from 'wavefile';
const { WaveFile } = wavefile;

env.allowLocalModels = false;
env.cacheDir = './.cache/transformers';

const wavPath = process.argv[2];
if (!wavPath) {
  console.error('usage: node scripts/smoke-fixture.mjs <wav-path>');
  process.exit(1);
}

const wav = new WaveFile(readFileSync(wavPath));
wav.toBitDepth('32f');
const raw = wav.getSamples();
const samples = Array.isArray(raw) ? raw[0] : raw;
const sr = wav.fmt.sampleRate;
console.log(`[smoke] ${wavPath}: ${samples.length} samples @ ${sr} Hz = ${(samples.length / sr).toFixed(2)} s`);

const transcriber = await pipeline(
  'automatic-speech-recognition',
  'onnx-community/whisper-tiny.en_timestamped',
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
console.log('[smoke] transcript:', result.text);
