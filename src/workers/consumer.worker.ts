/// <reference lib="webworker" />
import { db, type TranscriptToken } from '../lib/db';
import { DEFAULT_MODEL, type ModelId } from '../lib/audio';
import { HypothesisBuffer } from '../lib/transcription/hypothesisBuffer';
import {
  createPipeline,
  detectBackend,
  type Backend,
  type WhisperPipeline,
} from '../lib/transcription/whisperAdapter';
import {
  LiveTranscriptionLoop,
  type TickKind,
  type TickOutcome,
} from '../lib/transcription/liveLoop';
import {
  DexieAudioChunkRepository,
  DexieTranscriptRepository,
} from '../lib/repositories/dexieRepositories';

// Thin shell around `LiveTranscriptionLoop`. The worker owns:
//   - Whisper pipeline lifecycle (init, model, backend)
//   - Message dispatch (init / reset / flush)
//   - Timer-paced tick scheduling and the `BroadcastChannel` wake-up
//   - Stats wiring back to the main thread
//
// The algorithm (LocalAgreement-2, silence gate, anchor advancement, force
// slide, drain) lives in `LiveTranscriptionLoop`. See `src/lib/transcription/
// liveLoop.ts` for the design.

type InMessage =
  | { type: 'init'; modelId: ModelId }
  | { type: 'reset' }
  | { type: 'flush' };

type OutMessage =
  | { type: 'ready'; backend: 'webgpu' | 'wasm' }
  | { type: 'progress'; file: string; loaded: number; total: number; status: string }
  | { type: 'error'; message: string }
  | { type: 'display'; tokens: TranscriptToken[] }
  | { type: 'reset-done' }
  | { type: 'flush-done' }
  | {
      type: 'stats';
      tickCount: number;
      lastTickMs: number;
      lastTickAt: number;
      lastTickKind: TickKind;
      windowDurationS: number;
      committedWordsCount: number;
      prevHypothesisCount: number;
      stableFullTicks: number;
      committedAudioStartS: number;
      processing: boolean;
      draining: boolean;
    };

declare const self: DedicatedWorkerGlobalScope;

const wakeup = new BroadcastChannel('new-chunk');
const buffer = new HypothesisBuffer();
const audioRepo = new DexieAudioChunkRepository();
const transcriptRepo = new DexieTranscriptRepository();

let loop: LiveTranscriptionLoop | null = null;
let pipelinePromise: Promise<WhisperPipeline> | null = null;
let modelId: ModelId = DEFAULT_MODEL;
let backend: Backend = 'wasm';
let processing = false;
let pendingTick = false;
let draining = false;

// Dev stats.
let tickCount = 0;
let lastTickMs = 0;
let lastTickAt = 0;
let lastTickKind: TickKind = 'idle';
let lastWindowDurationS = 0;

function postOut(msg: OutMessage) {
  self.postMessage(msg);
}

function postStats() {
  if (!loop) return;
  postOut({
    type: 'stats',
    tickCount,
    lastTickMs,
    lastTickAt,
    lastTickKind,
    windowDurationS: lastWindowDurationS,
    committedWordsCount: loop.getCommittedWordCount(),
    prevHypothesisCount: loop.getTentativeWordCount(),
    stableFullTicks: 0, // retained for stats wire compatibility; not used by LA-2
    committedAudioStartS: loop.getCommittedAudioStartS(),
    processing,
    draining,
  });
}

function applyTickOutcome(outcome: TickOutcome) {
  lastTickKind = outcome.kind;
  lastWindowDurationS = outcome.windowDurationS;
  if (outcome.errorMessage) {
    postOut({ type: 'error', message: outcome.errorMessage });
  }
  if (outcome.rowsChanged) {
    postOut({ type: 'display', tokens: outcome.rows });
  }
}

async function init(id: ModelId) {
  modelId = id;
  backend = await detectBackend();
  pipelinePromise = createPipeline(modelId, backend, (p) => {
    postOut({ type: 'progress', ...p });
  });
  const pipeline = await pipelinePromise;
  loop = new LiveTranscriptionLoop({
    pipeline,
    buffer,
    audioRepo,
    transcriptRepo,
    modelId,
  });
  await resetState();
  postOut({ type: 'ready', backend });
  postStats();
  scheduleTick();
}

async function resetState() {
  if (loop) loop.reset();
  tickCount = 0;
  lastTickMs = 0;
  lastTickAt = 0;
  lastTickKind = 'idle';
  lastWindowDurationS = 0;
  await db.transcript.clear();
  await db.chunks.clear();
}

let tickTimer: number | null = null;
function scheduleTick() {
  if (draining) return; // Flush owns the loop; don't let wakeups slip a tick in.
  if (processing) {
    pendingTick = true;
    return;
  }
  if (tickTimer !== null) return;
  tickTimer = self.setTimeout(async () => {
    tickTimer = null;
    if (draining || !loop) return;
    processing = true;
    postStats(); // post BEFORE work so the UI sees processing=true
    const t0 = performance.now();
    try {
      const outcome = await loop.tick();
      applyTickOutcome(outcome);
    } catch (err) {
      lastTickKind = 'error';
      postOut({ type: 'error', message: (err as Error).message });
    } finally {
      lastTickMs = performance.now() - t0;
      lastTickAt = performance.now();
      tickCount++;
      processing = false;
      postStats();
      if (pendingTick && !draining) {
        pendingTick = false;
        scheduleTick();
      }
    }
  }, 50) as unknown as number;
}

wakeup.onmessage = () => {
  if (draining) return;
  scheduleTick();
};

self.onmessage = async (e: MessageEvent<InMessage>) => {
  const msg = e.data;
  if (msg.type === 'init') {
    try {
      await init(msg.modelId);
    } catch (err) {
      postOut({ type: 'error', message: `init failed: ${(err as Error).message}` });
    }
    return;
  }
  if (msg.type === 'reset') {
    draining = true;
    try {
      if (tickTimer !== null) {
        clearTimeout(tickTimer);
        tickTimer = null;
      }
      pendingTick = false;
      while (processing) {
        await new Promise((r) => setTimeout(r, 30));
      }
      await resetState();
      postOut({ type: 'reset-done' });
    } finally {
      // Post stats AFTER flipping `draining` back so the UI sees the clean
      // idle state instead of the intermediate "draining=true" frame.
      draining = false;
      postStats();
    }
    return;
  }
  if (msg.type === 'flush') {
    draining = true;
    try {
      if (tickTimer !== null) {
        clearTimeout(tickTimer);
        tickTimer = null;
      }
      pendingTick = false;
      while (processing) {
        await new Promise((r) => setTimeout(r, 30));
      }
      if (!loop) {
        postOut({ type: 'flush-done' });
        return;
      }
      processing = true;
      postStats();
      const t0 = performance.now();
      try {
        const rows = await loop.drainAndFinalise();
        postOut({ type: 'display', tokens: rows });
        lastTickKind = 'inference';
      } catch (err) {
        lastTickKind = 'error';
        postOut({ type: 'error', message: `flush failed: ${(err as Error).message}` });
      } finally {
        lastTickMs = performance.now() - t0;
        lastTickAt = performance.now();
        processing = false;
      }
      console.log(`[consumer] flushed; committed=${loop.getCommittedWordCount()}`);
      postOut({ type: 'flush-done' });
    } finally {
      draining = false;
      postStats();
    }
  }
};
