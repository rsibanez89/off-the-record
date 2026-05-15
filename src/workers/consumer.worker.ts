/// <reference lib="webworker" />
import { db, type TranscriptToken } from '../lib/db';
import { TARGET_SAMPLE_RATE, DEFAULT_MODEL, isMultilingual, type ModelId } from '../lib/audio';

// Whisper is trained on 30 s windows. Short input hallucinates; very long input
// blows past the model's max. We anchor audio start at the last committed
// boundary and grow toward MAX. Stable hypotheses are committed early; MAX is
// a fallback that commits the current hypothesis and starts fresh.
const MIN_WINDOW_S = 3.0;
const MIN_DRAIN_WINDOW_S = 1.5; // on Stop, accept shorter windows to catch trailing speech
const MAX_WINDOW_S = 24.0;
const STABLE_COMMIT_TICKS = 2;
const SILENCE_RMS_THRESHOLD = 0.005;

// Common Whisper filler outputs to drop. These appear when fed silence or noise.
const HALLUCINATION_PATTERNS = [
  /^>+$/,
  /^\.+$/,
  /^[♪♫]+$/,
  /^\[.*\]$/,
  /^\(.*\)$/,
  /^thank you\.?$/i,
  /^thanks for watching\.?$/i,
  /^you$/i,
  /^bye\.?$/i,
];

const norm = (s: string) => s.trim().toLowerCase().replace(/[.,!?;:"'()[\]\-–—]/g, '');

function isHallucinationLine(text: string): boolean {
  const t = text.trim();
  if (!t) return true;
  if (HALLUCINATION_PATTERNS.some((p) => p.test(t))) return true;
  // Strip every bracketed/parenthesised filler (e.g. "[BLANK_AUDIO]",
  // "(water splashing)") and music notation. If nothing meaningful remains,
  // it's a hallucination. Catches combined fillers like
  // "(water splashing) [BLANK_AUDIO]" that don't match a single pattern.
  const stripped = t
    .replace(/\[[^\]]*\]/g, '')
    .replace(/\([^)]*\)/g, '')
    .replace(/[♪♫]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  return stripped.length === 0;
}

function isHallucinationWord(w: string): boolean {
  const t = w.trim();
  if (!t) return true;
  return /^>+$/.test(t) || /^\.+$/.test(t) || /^[-–—]+$/.test(t) || /^[♪♫]+$/.test(t);
}

function rms(samples: Float32Array): number {
  if (samples.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

function tokenize(text: string): string[] {
  return text.split(/\s+/).filter(Boolean).filter((w) => !isHallucinationWord(w));
}

type InMessage =
  | { type: 'init'; modelId: ModelId }
  | { type: 'reset' }
  | { type: 'flush' };

type OutMessage =
  | { type: 'ready'; backend: 'webgpu' | 'wasm' }
  | { type: 'progress'; file: string; loaded: number; total: number; status: string }
  | { type: 'error'; message: string }
  | { type: 'log'; message: string }
  | { type: 'display'; tokens: TranscriptToken[] }
  | { type: 'reset-done' }
  | { type: 'flush-done' };

declare const self: DedicatedWorkerGlobalScope;

const wakeup = new BroadcastChannel('new-chunk');

let pipelinePromise: Promise<any> | null = null;
let modelId: ModelId = DEFAULT_MODEL;
let backend: 'webgpu' | 'wasm' = 'wasm';
let processing = false;
let pendingTick = false;
// While `draining`, new wakeups must not start ticks. The flush handler is
// running the drain loop itself and a parallel tick would write stale state.
let draining = false;

// Streaming state. `committedAudioStartS` anchors the left edge of the audio
// fed to Whisper; everything before it has been finalised. `committedWords`
// is the finalised transcript so far. `prevHypothesis` is the previous tick's
// full Whisper output, used for longest-prefix stability checks.
let committedAudioStartS = 0;
let committedWords: string[] = [];
let prevHypothesis: string[] = [];
let stableFullTicks = 0;

function postOut(msg: OutMessage) {
  self.postMessage(msg);
}

function log(message: string) {
  // Forward to main thread so the message lands in the page console where
  // browser tooling can read it (worker console output isn't captured).
  console.log(message);
  postOut({ type: 'log', message });
}

async function detectBackend(): Promise<'webgpu' | 'wasm'> {
  const anyNav = navigator as any;
  if (anyNav.gpu) {
    try {
      const adapter = await anyNav.gpu.requestAdapter();
      if (adapter) return 'webgpu';
    } catch {
      // fall through
    }
  }
  return 'wasm';
}

async function loadPipeline() {
  const transformers = await import('@huggingface/transformers');
  // Silence ORT's "Some nodes were not assigned to the preferred EP" warnings.
  // They're informational and unavoidable on WebGPU. ORT keeps cheap shape
  // ops on CPU on purpose. 3 = Error severity; real errors still log.
  try {
    (transformers.env.backends as any).onnx.logSeverityLevel = 3;
    (transformers.env.backends as any).onnx.logLevel = 'error';
  } catch {
    // Older transformers.js versions may not expose this; safe to ignore.
  }
  // Encoder dtype: fp32 is the default-quality baseline. Only large-v3-turbo
  // ships a published fp16 encoder; on smaller checkpoints fp16 either isn't
  // exported or measurably degrades accuracy on noisy/accented speech.
  // Decoder stays q4: that's where the speed win is.
  const isTurbo = modelId === 'onnx-community/whisper-large-v3-turbo';
  const dtype =
    backend === 'webgpu'
      ? { encoder_model: isTurbo ? 'fp16' : 'fp32', decoder_model_merged: 'q4' }
      : { encoder_model: 'fp32', decoder_model_merged: 'q4' };
  return await transformers.pipeline('automatic-speech-recognition', modelId, {
    device: backend,
    dtype: dtype as any,
    session_options: { logSeverityLevel: 3 } as any,
    progress_callback: (p: any) => {
      if (p && typeof p === 'object') {
        postOut({
          type: 'progress',
          file: p.file ?? '',
          loaded: p.loaded ?? 0,
          total: p.total ?? 0,
          status: p.status ?? '',
        });
      }
    },
  });
}

async function resetState() {
  committedAudioStartS = 0;
  committedWords = [];
  prevHypothesis = [];
  stableFullTicks = 0;
  await db.transcript.clear();
  await db.chunks.clear();
}

async function init(id: ModelId) {
  modelId = id;
  backend = await detectBackend();
  pipelinePromise = loadPipeline();
  await pipelinePromise;
  // Fresh model = fresh transcript. Old chunks would belong to a different run.
  await resetState();
  postOut({ type: 'ready', backend });
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

function longestCommonPrefix(a: string[], b: string[]): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && norm(a[i]) === norm(b[i])) i++;
  return i;
}

function overlapTailToHead(left: string[], right: string[]): number {
  const max = Math.min(left.length, right.length);
  for (let n = max; n > 0; n--) {
    let matches = true;
    for (let i = 0; i < n; i++) {
      if (norm(left[left.length - n + i]) !== norm(right[i])) {
        matches = false;
        break;
      }
    }
    if (matches) return n;
  }
  return 0;
}

function appendCommitted(words: string[]): number {
  if (words.length === 0) return 0;
  const overlap = overlapTailToHead(committedWords, words);
  const nextWords = words.slice(overlap);
  if (nextWords.length > 0) {
    committedWords = committedWords.concat(nextWords);
  }
  return nextWords.length;
}

async function writeTranscript(stable: string[], volatile_: string[]) {
  // Reconcile the display transcript from in-memory state. Avoid clear+rewrite:
  // Dexie live queries can briefly render the cleared/short table, which looks
  // like the transcript is flashing. Put the new rows first, then delete only
  // rows beyond the new length.
  await db.transaction('rw', db.transcript, async () => {
    const rows: TranscriptToken[] = [];
    for (const w of committedWords) {
      const tokenId = rows.length;
      rows.push({ tokenId, text: w, t: tokenId, isFinal: 1 as const });
    }
    for (const w of stable) {
      const tokenId = rows.length;
      rows.push({ tokenId, text: w, t: tokenId, isFinal: 1 as const });
    }
    for (const w of volatile_) {
      const tokenId = rows.length;
      rows.push({ tokenId, text: w, t: tokenId, isFinal: 0 as const });
    }

    if (rows.length === 0) {
      await db.transcript.clear();
      postOut({ type: 'display', tokens: [] });
      return;
    }

    await db.transcript.bulkPut(rows);
    await db.transcript.where('tokenId').aboveOrEqual(rows.length).delete();
    postOut({ type: 'display', tokens: rows });
  });
}

async function runOnce(minDur: number = MIN_WINDOW_S) {
  if (!pipelinePromise) return;
  const pipeline = await pipelinePromise;

  const audio = await collectAudioFrom(committedAudioStartS);
  if (!audio) return;
  const dur = audio.samples.length / TARGET_SAMPLE_RATE;
  if (dur < minDur) return;

  const energy = rms(audio.samples);
  if (energy < SILENCE_RMS_THRESHOLD) {
    // Pure silence is a natural commit point. Pin the last hypothesis before
    // resetting prevHypothesis so visible words do not fall back to grey.
    if (prevHypothesis.length > 0) {
      const appended = appendCommitted(prevHypothesis);
      await writeTranscript([], []);
      log(`[consumer] silence commit +${appended}: ${prevHypothesis.join(' ')}`);
    }
    committedAudioStartS = audio.t1;
    prevHypothesis = [];
    stableFullTicks = 0;
    await db.chunks.where('startedAt').below(committedAudioStartS).delete();
    log(`[consumer] silence skip dur=${dur.toFixed(2)}s rms=${energy.toFixed(4)}`);
    return;
  }

  let text = '';
  try {
    const opts: Record<string, unknown> = {
      chunk_length_s: 30,
      stride_length_s: 0,
      return_timestamps: false,
      no_repeat_ngram_size: 3,
      // Explicit greedy decoding. transformers.js usually defaults this way,
      // but being explicit avoids surprises across versions.
      top_k: 0,
      do_sample: false,
    };
    // `.en` checkpoints have no language/task tokens; passing them errors out.
    if (isMultilingual(modelId)) {
      opts.language = 'en';
      opts.task = 'transcribe';
    }
    const result = await pipeline(audio.samples, opts);
    text = (Array.isArray(result) ? result[0]?.text : result?.text) ?? '';
  } catch (err) {
    postOut({ type: 'error', message: `inference failed: ${(err as Error).message}` });
    return;
  }

  log(`[consumer] dur=${dur.toFixed(2)}s rms=${energy.toFixed(3)} text="${text.slice(0, 120)}"`);

  if (isHallucinationLine(text)) {
    // Treat hallucination the same as silence: pin prev, advance the anchor,
    // and discard chunks. Without this, the chunks accumulate, audio keeps
    // growing past MAX_WINDOW_S, and force-commit eventually pins junk.
    if (prevHypothesis.length > 0) {
      const appended = appendCommitted(prevHypothesis);
      await writeTranscript([], []);
      log(`[consumer] hallucination commit prev +${appended}`);
    }
    committedAudioStartS = audio.t1;
    prevHypothesis = [];
    stableFullTicks = 0;
    await db.chunks.where('startedAt').below(committedAudioStartS).delete();
    log(`[consumer] dropped as hallucination (text="${text.slice(0, 80)}")`);
    return;
  }

  const currWords = tokenize(text);

  // Longest-prefix agreement is used for COLOURING only: the prefix where the
  // previous and current hypotheses agree is "stable" (black). The rest is
  // "volatile" (grey). Words can move both directions across ticks: if
  // Whisper revises a word that was stable last tick, it becomes grey again.
  // True permanent commits only happen at the MAX_WINDOW_S boundary below.
  const lcp = longestCommonPrefix(prevHypothesis, currWords);
  const stable = currWords.slice(0, lcp);
  const volatile_ = currWords.slice(lcp);
  log(`[consumer] stable=${stable.length} volatile=${volatile_.length}`);

  await writeTranscript(stable, volatile_);
  prevHypothesis = currWords;

  if (currWords.length > 0 && volatile_.length === 0) {
    stableFullTicks++;
    if (stableFullTicks >= STABLE_COMMIT_TICKS) {
      const appended = appendCommitted(currWords);
      committedAudioStartS = audio.t1;
      prevHypothesis = [];
      stableFullTicks = 0;
      await db.chunks.where('startedAt').below(committedAudioStartS).delete();
      await writeTranscript([], []);
      log(`[consumer] stable commit +${appended} after ${STABLE_COMMIT_TICKS} ticks`);
      return;
    }
  } else {
    stableFullTicks = 0;
  }

  // Force-slide at MAX_WINDOW_S. Whisper does not give us word timestamps here,
  // so splitting a hypothesis by a guessed audio fraction is unsafe. Once the
  // window is too long, pin the whole current hypothesis and start a fresh
  // audio window after it.
  if (dur > MAX_WINDOW_S && currWords.length > 0) {
    const appended = appendCommitted(currWords);
    committedAudioStartS = audio.t1;
    prevHypothesis = [];
    stableFullTicks = 0;
    await db.chunks.where('startedAt').below(committedAudioStartS).delete();
    await writeTranscript([], []);
    log(`[consumer] force-slide +${appended} (lcp=${lcp}, slid 100%)`);
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
    try {
      await runOnce();
    } catch (err) {
      postOut({ type: 'error', message: (err as Error).message });
    } finally {
      processing = false;
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

async function flushTentativeToFinal() {
  await db.transaction('rw', db.transcript, async () => {
    const rows = await db.transcript.where('isFinal').equals(0).toArray();
    for (const r of rows) {
      await db.transcript.put({ ...r, isFinal: 1 });
    }
  });
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
      draining = false;
    }
    return;
  }
  if (msg.type === 'flush') {
    // Drain semantics: stop producer is already happening on main; here we
    // finish transcribing whatever audio is already in IDB, then finalise the
    // entire transcript to black. `draining` blocks concurrent wakeup ticks.
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

      // Drain loop. Run ticks until the queue is short, stability settles with no
      // anchor advancement, or the safety cap fires.
      for (let i = 0; i < 20; i++) {
        const audio = await collectAudioFrom(committedAudioStartS);
        if (!audio) break;
        const dur = audio.samples.length / TARGET_SAMPLE_RATE;
        if (dur < MIN_DRAIN_WINDOW_S) break;

        const beforePrev = prevHypothesis.join('|');
        const beforeAnchor = committedAudioStartS;
        const beforeCommittedLen = committedWords.length;

        processing = true;
        try {
          await runOnce(MIN_DRAIN_WINDOW_S);
        } catch (err) {
          postOut({ type: 'error', message: (err as Error).message });
          break;
        } finally {
          processing = false;
        }

        const settled =
          prevHypothesis.join('|') === beforePrev &&
          committedAudioStartS === beforeAnchor &&
          committedWords.length === beforeCommittedLen;
        if (settled) break;
      }

      // Pin the final hypothesis into committedWords. This is the entire
      // session's transcript now.
      if (prevHypothesis.length > 0) {
        appendCommitted(prevHypothesis);
        prevHypothesis = [];
        stableFullTicks = 0;
      }

      // Force the DB to match in-memory state: only committedWords, all final.
      // Promote any leftover tentatives belt-and-suspenders, then rewrite.
      await flushTentativeToFinal();
      await writeTranscript([], []);

      // Reset chunks and anchor so the next Record starts clean.
      await db.chunks.clear();
      committedAudioStartS = 0;
      log(`[consumer] flushed and drained; committedWords.length=${committedWords.length}`);
      postOut({ type: 'flush-done' });
    } finally {
      draining = false;
    }
  }
};
