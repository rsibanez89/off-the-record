/// <reference lib="webworker" />
import { db } from '../lib/db';
import { CHUNK_SAMPLES, TARGET_SAMPLE_RATE } from '../lib/audio';

type InMessage =
  | { type: 'start'; sourceSampleRate: number }
  | { type: 'frame'; samples: Float32Array }
  | { type: 'stop' };

type OutMessage =
  | { type: 'started' }
  | { type: 'stopped' }
  | { type: 'error'; message: string }
  | {
      type: 'stats';
      totalChunksWritten: number;
      accumSamples: number;
      running: boolean;
      sourceSampleRate: number;
      lastChunkAt: number;
    };

const wakeup = new BroadcastChannel('new-chunk');

let sourceRate = 0;
let resampleRatio = 1;
let accum: number[] = [];
let chunkStartedAt = 0;
let running = false;

// Dev stats. Lifetime counters since the most recent 'start'. UI computes rate
// from successive snapshots.
let totalChunksWritten = 0;
let lastChunkAt = 0;
let statsTimer: number | null = null;

// Resampler state. `resamplePos` is the next emit position in the current
// frame's local coordinates (it can be negative, meaning "between the previous
// frame's last sample and this frame's first sample"). `prevTail` is the last
// sample of the previous frame, used to interpolate across the frame seam.
let resamplePos = 0;
let prevTail = 0;

function postOut(msg: OutMessage) {
  (self as DedicatedWorkerGlobalScope).postMessage(msg);
}

function postStats() {
  postOut({
    type: 'stats',
    totalChunksWritten,
    accumSamples: accum.length,
    running,
    sourceSampleRate: sourceRate,
    lastChunkAt,
  });
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
      // Write to both tables in one transaction. The live consumer evicts
      // chunks past its committed-audio anchor; audioArchive keeps the full
      // recording for the batch worker to read on Stop.
      await db.transaction('rw', db.chunks, db.audioArchive, async () => {
        await db.chunks.add({ startedAt, samples });
        await db.audioArchive.add({ startedAt, samples });
      });
      wakeup.postMessage({ type: 'new-chunk' });
      totalChunksWritten++;
      lastChunkAt = performance.now();
      postStats();
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
    totalChunksWritten = 0;
    lastChunkAt = 0;
    if (statsTimer === null) {
      // 500 ms is fine-grained enough to show the accum buffer filling between
      // chunk writes, but cheap enough not to flood the main thread.
      statsTimer = (self as DedicatedWorkerGlobalScope).setInterval(postStats, 500) as unknown as number;
    }
    postOut({ type: 'started' });
    postStats();
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
    if (statsTimer !== null) {
      clearInterval(statsTimer);
      statsTimer = null;
    }
    postStats();
    postOut({ type: 'stopped' });
    return;
  }
};
