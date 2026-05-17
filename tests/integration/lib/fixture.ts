// Audio fixture loader: reads a 16-kHz mono WAV via `wavefile`, returns a
// `Float32Array` aligned with what the live consumer sees in production
// (resampled-to-16-kHz mono).

import { readFileSync } from 'node:fs';
import { WaveFile } from 'wavefile';

export interface AudioFixture {
  id: string;
  samples: Float32Array;
  sampleRateHz: number;
  durationS: number;
}

export interface Groundtruth {
  id: string;
  source: string;
  license: string;
  sampleRateHz: number;
  channels: number;
  durationS: number;
  transcript: string;
  notes?: string;
}

export function loadGroundtruth(path: string): Groundtruth {
  return JSON.parse(readFileSync(path, 'utf-8')) as Groundtruth;
}

export function loadWav16kMono(path: string, id: string): AudioFixture {
  const buf = readFileSync(path);
  const wav = new WaveFile(buf);
  wav.toBitDepth('32f');
  // `getSamples()` returns Float32Array for mono, or [L, R] for stereo. Our
  // fixtures are all 16 kHz mono so we expect the array shape.
  const samplesRaw = wav.getSamples();
  const samples = Array.isArray(samplesRaw) ? samplesRaw[0] : samplesRaw;
  const fmt = wav.fmt as { sampleRate: number };
  if (fmt.sampleRate !== 16_000) {
    throw new Error(`fixture ${id}: expected 16 kHz, got ${fmt.sampleRate}`);
  }
  return {
    id,
    samples: samples as Float32Array,
    sampleRateHz: fmt.sampleRate,
    durationS: samples.length / fmt.sampleRate,
  };
}

/**
 * Split a Float32Array into 1-second chunks (16_000 samples at 16 kHz) to
 * mirror what the producer worker would write to `db.chunks` in production.
 * Sub-second remainders are dropped to match `producer.worker.ts:213-216`.
 */
export function chunkIntoSeconds(samples: Float32Array, sampleRateHz: number): Float32Array[] {
  const chunkSize = sampleRateHz; // 1 second
  const chunks: Float32Array[] = [];
  for (let off = 0; off + chunkSize <= samples.length; off += chunkSize) {
    chunks.push(samples.subarray(off, off + chunkSize));
  }
  return chunks;
}
