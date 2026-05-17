/// <reference lib="webworker" />
import { CHUNK_SAMPLES, TARGET_SAMPLE_RATE } from '../lib/audio';
import { DexieChunkPublisher } from '../lib/repositories/dexieRepositories';
import {
  Framer,
  LinearResampler,
  SILERO_FRAME,
  createVadEngine,
  type VadEngine,
} from '../lib/vad';

type InMessage =
  | { type: 'start'; sourceSampleRate: number }
  | { type: 'frame'; samples: Float32Array; sampleRate?: number }
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
const publisher = new DexieChunkPublisher();

let sourceRate = 0;
let resampler: LinearResampler | null = null;
// Ring buffer for assembling 1-second (TARGET_SAMPLE_RATE samples) chunks
// from the 16 kHz stream. Pre-allocated to four chunks so we never resize in
// the common case. (Improvement 3.3: replaces the previous `number[] accum`.)
let chunkRing = new Float32Array(CHUNK_SAMPLES * 4);
let chunkRingLength = 0;
let chunkStartedAt = 0;
let running = false;

// VAD pipeline (lazy). The first half-second of a session may write chunks
// with `speechProbability: undefined` while the model loads; recording must
// not block on VAD.
let vadEngine: VadEngine | null = null;
let vadFramer: Framer | null = null;
// Max VAD probability accumulated over the current outgoing chunk.
let vadMaxProbForChunk: number | undefined = undefined;
let vadReady = false;
let vadInitError: string | null = null;
// In-flight load promise. Used to dedupe concurrent `loadVad()` calls so a
// fast Record/Stop/Record cycle does not start a second 2.2 MB fetch while
// the first one is still running.
let vadLoadPromise: Promise<void> | null = null;

// Dev stats. Lifetime counters since the most recent 'start'. UI computes rate
// from successive snapshots.
let totalChunksWritten = 0;
let lastChunkAt = 0;
let statsTimer: number | null = null;

function postOut(msg: OutMessage) {
  (self as DedicatedWorkerGlobalScope).postMessage(msg);
}

function postStats() {
  postOut({
    type: 'stats',
    totalChunksWritten,
    accumSamples: chunkRingLength,
    running,
    sourceSampleRate: sourceRate,
    lastChunkAt,
  });
}

function ensureChunkRingCapacity(needed: number) {
  if (chunkRing.length >= needed) return;
  let next = chunkRing.length;
  while (next < needed) next *= 2;
  const grown = new Float32Array(next);
  grown.set(chunkRing.subarray(0, chunkRingLength));
  chunkRing = grown;
}

function appendToChunkRing(samples: Float32Array) {
  ensureChunkRingCapacity(chunkRingLength + samples.length);
  chunkRing.set(samples, chunkRingLength);
  chunkRingLength += samples.length;
}

async function runVadOnSamples(samples: Float32Array): Promise<void> {
  if (!vadEngine || !vadFramer || !vadReady) return;
  const frames = vadFramer.push(samples);
  for (const f of frames) {
    try {
      const p = await vadEngine.process(f);
      if (vadMaxProbForChunk === undefined || p > vadMaxProbForChunk) {
        vadMaxProbForChunk = p;
      }
    } catch (err) {
      // VAD failure must NOT take down the recording. Log and disable VAD
      // for the rest of the session; chunks fall back to `undefined`.
      postOut({
        type: 'error',
        message: `vad inference failed (disabling VAD): ${(err as Error).message}`,
      });
      vadReady = false;
      return;
    }
  }
}

async function flushChunkIfReady() {
  while (chunkRingLength >= CHUNK_SAMPLES) {
    const slice = chunkRing.slice(0, CHUNK_SAMPLES);
    chunkRing.copyWithin(0, CHUNK_SAMPLES, chunkRingLength);
    chunkRingLength -= CHUNK_SAMPLES;
    const startedAt = chunkStartedAt;
    chunkStartedAt += CHUNK_SAMPLES / TARGET_SAMPLE_RATE;
    const speechProbability = vadMaxProbForChunk;
    // Reset per-chunk max for the next 1-second window.
    vadMaxProbForChunk = undefined;
    try {
      // Atomic dual-table publish: chunks (evictable by the consumer
      // anchor) and audioArchive (kept for the batch worker) in one
      // transaction. Atomicity prevents an orphan chunk in one table.
      await publisher.publish({ startedAt, samples: slice, speechProbability });
      wakeup.postMessage({ type: 'new-chunk' });
      totalChunksWritten++;
      lastChunkAt = performance.now();
      postStats();
    } catch (err) {
      postOut({ type: 'error', message: `db write failed: ${(err as Error).message}` });
    }
  }
}

async function loadVadOnce(): Promise<void> {
  // Lazy import + initialize. Errors are logged but do not block recording.
  // `vadInitError` is cleared on every attempt so a transient fetch failure
  // (e.g. flaky network during the first 2.2 MB download) is retryable on
  // the user's next Record click rather than sticky for the tab lifetime.
  vadInitError = null;
  try {
    const engine = createVadEngine();
    await engine.initialize();
    vadEngine = engine;
    vadFramer = new Framer(SILERO_FRAME);
    vadReady = true;
  } catch (err) {
    vadInitError = (err as Error).message;
    postOut({ type: 'error', message: `vad init failed: ${vadInitError}` });
  }
}

function ensureVadLoaded(): Promise<void> {
  if (vadEngine && vadReady) return Promise.resolve();
  if (vadLoadPromise) return vadLoadPromise;
  vadLoadPromise = loadVadOnce().finally(() => {
    vadLoadPromise = null;
  });
  return vadLoadPromise;
}

self.onmessage = async (e: MessageEvent<InMessage>) => {
  const msg = e.data;
  if (msg.type === 'start') {
    sourceRate = msg.sourceSampleRate;
    resampler = new LinearResampler(sourceRate, TARGET_SAMPLE_RATE);
    chunkRingLength = 0;
    chunkStartedAt = performance.now() / 1000;
    running = true;
    totalChunksWritten = 0;
    lastChunkAt = 0;
    vadMaxProbForChunk = undefined;
    if (vadFramer) vadFramer.reset();
    if (vadEngine) vadEngine.reset();
    // Kick off VAD load (or wait on an in-flight one). Cheap when the
    // engine is already loaded. Retries on every Record if the previous
    // attempt failed; transient fetch errors recover without a tab reload.
    // Fire and forget: recording must not block on VAD.
    if (vadInitError) {
      console.warn(
        `[producer] VAD load previously failed (${vadInitError}); retrying on this Record`,
      );
    }
    void ensureVadLoaded();
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
    if (!running || !resampler) return;
    // If the AudioContext sample rate changes mid-stream (rare, but
    // possible on device switch on macOS), rebuild the resampler so the
    // 16 kHz target stream stays correct. Forward-only check: a one-time
    // rebuild is fine; the previous resampler's tail state is discarded
    // along with the old ratio.
    if (msg.sampleRate != null && msg.sampleRate !== sourceRate) {
      sourceRate = msg.sampleRate;
      resampler = new LinearResampler(sourceRate, TARGET_SAMPLE_RATE);
    }
    const resampled = resampler.process(msg.samples);
    appendToChunkRing(resampled);
    // VAD runs on the 16 kHz resampled stream in 512-sample frames. Awaiting
    // here serialises VAD inference with chunk writes; for Silero (sub-ms
    // per frame on WASM SIMD) this is fine. If we ever swap in a heavier
    // engine, batch the frames or move them off the critical path.
    await runVadOnSamples(resampled);
    await flushChunkIfReady();
    return;
  }
  if (msg.type === 'stop') {
    running = false;
    // Drop any partial sub-second remainder. (See PLAN.md note: padding
    // remainder to a full chunk is improvement 3.6, deferred.)
    chunkRingLength = 0;
    if (vadFramer) vadFramer.reset();
    if (vadEngine) vadEngine.reset();
    vadMaxProbForChunk = undefined;
    if (statsTimer !== null) {
      clearInterval(statsTimer);
      statsTimer = null;
    }
    postStats();
    postOut({ type: 'stopped' });
    return;
  }
};
