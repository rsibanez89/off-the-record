/// <reference lib="webworker" />
import { db, type TranscriptToken } from '../lib/db';
import { TARGET_SAMPLE_RATE, DEFAULT_MODEL, isMultilingual, type ModelId } from '../lib/audio';
import { HypothesisBuffer } from '../lib/transcription/hypothesisBuffer';
import { isHallucinationLine, isSilent, rms } from '../lib/transcription/heuristics';
import {
  createPipeline,
  detectBackend,
  runWhisper,
  type Backend,
} from '../lib/transcription/whisperAdapter';

// Window-size policy. Whisper is trained on 30 s windows. Growth beyond
// MAX_WINDOW_S is bounded by a force-slide safety net. The anchor advances
// naturally as words get committed via LA-2.
//
// MIN_WINDOW_S = 0 means "run on the first chunk available". User-feedback
// optimisation: a grey wrong word at t=1.3s is better than 3 s of empty UI
// where the user wonders if the mic is even working. LA-2 will revise junk
// on subsequent ticks, and the hallucination filter catches obvious fillers
// ("thanks for watching", "[BLANK_AUDIO]") so they never commit.
const MIN_WINDOW_S = 0;
const MIN_DRAIN_WINDOW_S = 0; // drain everything left in the buffer on Stop
const MAX_WINDOW_S = 24.0;

type InMessage =
  | { type: 'init'; modelId: ModelId }
  | { type: 'reset' }
  | { type: 'flush' };

type TickKind =
  | 'idle'
  | 'short-window'
  | 'silence'
  | 'hallucination'
  | 'inference'
  | 'force-slide'
  | 'error';

type OutMessage =
  | { type: 'ready'; backend: 'webgpu' | 'wasm' }
  | { type: 'progress'; file: string; loaded: number; total: number; status: string }
  | { type: 'error'; message: string }
  | { type: 'log'; message: string }
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
const hypBuffer = new HypothesisBuffer();

let pipelinePromise: Promise<any> | null = null;
let modelId: ModelId = DEFAULT_MODEL;
let backend: Backend = 'wasm';
let processing = false;
let pendingTick = false;
let draining = false;
let committedAudioStartS = 0;

// Dev stats.
let tickCount = 0;
let lastTickMs = 0;
let lastTickAt = 0;
let lastTickKind: TickKind = 'idle';
let lastWindowDurationS = 0;

function postOut(msg: OutMessage) {
  self.postMessage(msg);
}

function log(message: string) {
  console.log(message);
  postOut({ type: 'log', message });
}

function postStats() {
  postOut({
    type: 'stats',
    tickCount,
    lastTickMs,
    lastTickAt,
    lastTickKind,
    windowDurationS: lastWindowDurationS,
    committedWordsCount: hypBuffer.getCommitted().length,
    prevHypothesisCount: hypBuffer.getTentative().length,
    stableFullTicks: 0, // retained for stats wire compatibility; not used by LA-2
    committedAudioStartS,
    processing,
    draining,
  });
}

async function loadPipeline() {
  return createPipeline(modelId, backend, (p) => {
    postOut({ type: 'progress', ...p });
  });
}

async function resetState() {
  committedAudioStartS = 0;
  hypBuffer.reset();
  tickCount = 0;
  lastTickMs = 0;
  lastTickAt = 0;
  lastTickKind = 'idle';
  lastWindowDurationS = 0;
  await db.transcript.clear();
  await db.chunks.clear();
}

async function init(id: ModelId) {
  modelId = id;
  backend = await detectBackend();
  pipelinePromise = loadPipeline();
  await pipelinePromise;
  await resetState();
  postOut({ type: 'ready', backend });
  postStats();
  scheduleTick();
}

async function collectAudioFrom(startS: number): Promise<{ samples: Float32Array; t0: number; t1: number } | null> {
  const chunks = await db.chunks.where('startedAt').aboveOrEqual(startS).sortBy('startedAt');
  if (chunks.length === 0) return null;
  const total = chunks.reduce((s, c) => s + c.samples.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const c of chunks) {
    out.set(c.samples, offset);
    offset += c.samples.length;
  }
  const t0 = chunks[0].startedAt;
  const t1 = t0 + total / TARGET_SAMPLE_RATE;
  return { samples: out, t0, t1 };
}

async function writeTranscript(): Promise<void> {
  // Rebuild the transcript table from in-memory state. We bulkPut all rows
  // first, then delete only the tail beyond the new length. This avoids the
  // "clear + rewrite" flicker that Dexie live queries would otherwise show.
  const rows: TranscriptToken[] = [];
  for (const w of hypBuffer.getCommitted()) {
    rows.push({ tokenId: rows.length, text: w.text, t: w.start, isFinal: 1 });
  }
  for (const w of hypBuffer.getTentative()) {
    rows.push({ tokenId: rows.length, text: w.text, t: w.start, isFinal: 0 });
  }

  await db.transaction('rw', db.transcript, async () => {
    if (rows.length === 0) {
      await db.transcript.clear();
    } else {
      await db.transcript.bulkPut(rows);
      await db.transcript.where('tokenId').aboveOrEqual(rows.length).delete();
    }
  });
  postOut({ type: 'display', tokens: rows });
}

async function advanceAnchor(toS: number) {
  if (toS <= committedAudioStartS) return;
  committedAudioStartS = toS;
  await db.chunks.where('startedAt').below(committedAudioStartS).delete();
}

async function runOnce(minDur: number = MIN_WINDOW_S) {
  lastTickKind = 'idle';
  lastWindowDurationS = 0;
  if (!pipelinePromise) return;
  const pipeline = await pipelinePromise;

  const audio = await collectAudioFrom(committedAudioStartS);
  if (!audio) return;
  const dur = audio.samples.length / TARGET_SAMPLE_RATE;
  lastWindowDurationS = dur;
  if (dur < minDur) {
    lastTickKind = 'short-window';
    return;
  }

  // Silence detection happens before inference. On silence we force-commit any
  // tentative words (we know there won't be further agreement) and skip
  // Whisper entirely, advancing the anchor past the silent region.
  if (isSilent(audio.samples)) {
    const committed = hypBuffer.forceCommit();
    await writeTranscript();
    await advanceAnchor(audio.t1);
    lastTickKind = 'silence';
    log(`[consumer] silence dur=${dur.toFixed(2)}s rms=${rms(audio.samples).toFixed(4)} force-committed=${committed.length}`);
    return;
  }

  let result;
  try {
    result = await runWhisper(pipeline as any, audio.samples, {
      language: isMultilingual(modelId) ? 'en' : undefined,
      offsetSeconds: audio.t0,
    });
  } catch (err) {
    lastTickKind = 'error';
    postOut({ type: 'error', message: `inference failed: ${(err as Error).message}` });
    return;
  }

  log(`[consumer] dur=${dur.toFixed(2)}s rms=${rms(audio.samples).toFixed(3)} text="${result.text.slice(0, 120)}"`);

  if (isHallucinationLine(result.text)) {
    // Treat hallucination the same as silence: pin tentative, advance anchor,
    // and discard chunks. Without this, the chunks accumulate, audio keeps
    // growing past MAX_WINDOW_S, and force-slide eventually pins junk.
    const committed = hypBuffer.forceCommit();
    await writeTranscript();
    await advanceAnchor(audio.t1);
    lastTickKind = 'hallucination';
    log(`[consumer] hallucination force-committed=${committed.length} text="${result.text.slice(0, 80)}"`);
    return;
  }

  const justCommitted = hypBuffer.ingest(result.words);
  await writeTranscript();
  lastTickKind = 'inference';
  log(
    `[consumer] LA2 just-committed=${justCommitted.length} ` +
      `committed=${hypBuffer.getCommitted().length} tentative=${hypBuffer.getTentative().length}`
  );

  // Advance the audio anchor to the last committed word's end time so the
  // next inference only sees uncommitted audio. Whisper re-transcribing 30 s
  // every tick is wasteful and amplifies revision wobble.
  const lastT = hypBuffer.getLastCommittedTime();
  if (lastT > committedAudioStartS) {
    await advanceAnchor(lastT);
  }

  // Force-slide safety net at MAX_WINDOW_S. With LA-2 advancing the anchor
  // naturally on every commit, this fires rarely: only when Whisper keeps
  // revising and no word ever stabilises (e.g. continuous mumbled speech).
  if (dur > MAX_WINDOW_S) {
    const committed = hypBuffer.forceCommit();
    await writeTranscript();
    await advanceAnchor(audio.t1);
    lastTickKind = 'force-slide';
    log(`[consumer] force-slide dur=${dur.toFixed(2)}s force-committed=${committed.length}`);
  }
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
    if (draining) return;
    processing = true;
    postStats(); // post BEFORE work so the UI sees processing=true
    const t0 = performance.now();
    try {
      await runOnce();
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

async function drainAndFinalise() {
  for (let i = 0; i < 20; i++) {
    const audio = await collectAudioFrom(committedAudioStartS);
    if (!audio) break;
    const dur = audio.samples.length / TARGET_SAMPLE_RATE;
    if (dur < MIN_DRAIN_WINDOW_S) break;

    const beforeAnchor = committedAudioStartS;
    const beforeCommittedLen = hypBuffer.getCommitted().length;
    const beforeTentativeLen = hypBuffer.getTentative().length;

    processing = true;
    postStats();
    const t0 = performance.now();
    try {
      await runOnce(MIN_DRAIN_WINDOW_S);
    } catch (err) {
      lastTickKind = 'error';
      postOut({ type: 'error', message: (err as Error).message });
      break;
    } finally {
      lastTickMs = performance.now() - t0;
      lastTickAt = performance.now();
      tickCount++;
      processing = false;
      postStats();
    }

    const settled =
      committedAudioStartS === beforeAnchor &&
      hypBuffer.getCommitted().length === beforeCommittedLen &&
      hypBuffer.getTentative().length === beforeTentativeLen;
    if (settled) break;
  }

  // Pin any remaining tentative words: this is the final transcript.
  hypBuffer.forceCommit();
  await writeTranscript();
  await db.chunks.clear();
  committedAudioStartS = 0;
}

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
      await drainAndFinalise();
      log(`[consumer] flushed; committed=${hypBuffer.getCommitted().length}`);
      postOut({ type: 'flush-done' });
    } finally {
      draining = false;
      postStats();
    }
  }
};
