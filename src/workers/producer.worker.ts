/// <reference lib="webworker" />
import { db } from '../lib/db';
import { CHUNK_SAMPLES, TARGET_SAMPLE_RATE } from '../lib/audio';

type InMessage =
  | { type: 'start'; sourceSampleRate: number }
  | { type: 'frame'; samples: Float32Array }
  | { type: 'stop' };

type OutMessage = { type: 'started' } | { type: 'stopped' } | { type: 'error'; message: string };

const wakeup = new BroadcastChannel('new-chunk');

let sourceRate = 0;
let resampleRatio = 1;
let accum: number[] = [];
let chunkStartedAt = 0;
let running = false;

// Resampler state. `resamplePos` is the next emit position in the current
// frame's local coordinates (it can be negative, meaning "between the previous
// frame's last sample and this frame's first sample"). `prevTail` is the last
// sample of the previous frame, used to interpolate across the frame seam.
let resamplePos = 0;
let prevTail = 0;

function postOut(msg: OutMessage) {
  (self as DedicatedWorkerGlobalScope).postMessage(msg);
}

// Linear-interp resample from sourceRate to TARGET_SAMPLE_RATE. Treats
// successive frames as one continuous stream by carrying `resamplePos` and
// `prevTail` across calls.
function resampleAndPush(samples: Float32Array) {
  if (sourceRate === TARGET_SAMPLE_RATE) {
    for (let i = 0; i < samples.length; i++) accum.push(samples[i]);
    return;
  }
  while (true) {
    const i0 = Math.floor(resamplePos);
    const i1 = i0 + 1;
    if (i1 >= samples.length) break;
    const frac = resamplePos - i0;
    const a = i0 < 0 ? prevTail : samples[i0];
    const b = samples[i1];
    accum.push(a * (1 - frac) + b * frac);
    resamplePos += resampleRatio;
  }
  // Carry into next frame's coordinate system.
  resamplePos -= samples.length;
  prevTail = samples[samples.length - 1];
}

async function flushChunkIfReady() {
  while (accum.length >= CHUNK_SAMPLES) {
    const slice = accum.slice(0, CHUNK_SAMPLES);
    accum = accum.slice(CHUNK_SAMPLES);
    const samples = new Float32Array(slice);
    const startedAt = chunkStartedAt;
    chunkStartedAt += CHUNK_SAMPLES / TARGET_SAMPLE_RATE;
    try {
      await db.chunks.add({ startedAt, samples });
      wakeup.postMessage({ type: 'new-chunk' });
    } catch (err) {
      postOut({ type: 'error', message: `db write failed: ${(err as Error).message}` });
    }
  }
}

self.onmessage = async (e: MessageEvent<InMessage>) => {
  const msg = e.data;
  if (msg.type === 'start') {
    sourceRate = msg.sourceSampleRate;
    resampleRatio = sourceRate / TARGET_SAMPLE_RATE;
    accum = [];
    resamplePos = 0;
    prevTail = 0;
    chunkStartedAt = performance.now() / 1000;
    running = true;
    postOut({ type: 'started' });
    return;
  }
  if (msg.type === 'frame') {
    if (!running) return;
    resampleAndPush(msg.samples);
    await flushChunkIfReady();
    return;
  }
  if (msg.type === 'stop') {
    running = false;
    // Drop any partial sub-second remainder; the window will pick up final audio
    // from the last full chunk already written.
    accum = [];
    postOut({ type: 'stopped' });
    return;
  }
};
