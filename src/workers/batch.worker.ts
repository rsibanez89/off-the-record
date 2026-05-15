/// <reference lib="webworker" />
import { db, type TranscriptToken } from '../lib/db';
import { TARGET_SAMPLE_RATE, DEFAULT_MODEL, isMultilingual, type ModelId } from '../lib/audio';
import {
  createPipeline,
  detectBackend,
  runWhisper,
  type Backend,
} from '../lib/transcription/whisperAdapter';

// One-shot batch transcription worker. Loads its own Whisper pipeline,
// independent of the live consumer. On `transcribe` it reads the full
// `audioArchive` table, concatenates the chunks, and runs Whisper once with
// `chunk_length_s: 30` (transformers.js handles longer-than-30s audio
// internally via overlapping windows).

type InMessage =
  | { type: 'init'; modelId: ModelId }
  | { type: 'transcribe'; sessionId: number };

type OutMessage =
  | { type: 'ready'; backend: Backend }
  | { type: 'progress'; file: string; loaded: number; total: number; status: string }
  | { type: 'error'; message: string }
  | { type: 'log'; message: string }
  | { type: 'transcribe-start'; sessionId: number }
  | {
      type: 'transcribe-done';
      sessionId: number;
      tokens: TranscriptToken[];
      durationS: number;
      inferenceMs: number;
    };

declare const self: DedicatedWorkerGlobalScope;

let pipelinePromise: Promise<any> | null = null;
let modelId: ModelId = DEFAULT_MODEL;
let backend: Backend = 'wasm';

function postOut(msg: OutMessage) {
  self.postMessage(msg);
}

function log(message: string) {
  console.log(message);
  postOut({ type: 'log', message });
}

async function init(id: ModelId) {
  modelId = id;
  backend = await detectBackend();
  pipelinePromise = createPipeline(modelId, backend, (p) => {
    postOut({ type: 'progress', ...p });
  });
  await pipelinePromise;
  postOut({ type: 'ready', backend });
}

async function loadFullAudio(): Promise<Float32Array | null> {
  const chunks = await db.audioArchive.orderBy('startedAt').toArray();
  if (chunks.length === 0) return null;
  const total = chunks.reduce((s, c) => s + c.samples.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const c of chunks) {
    out.set(c.samples, offset);
    offset += c.samples.length;
  }
  return out;
}

async function transcribe(sessionId: number) {
  if (!pipelinePromise) {
    postOut({ type: 'error', message: 'batch pipeline not initialised yet' });
    return;
  }
  const pipeline = await pipelinePromise;

  postOut({ type: 'transcribe-start', sessionId });
  const samples = await loadFullAudio();
  if (!samples) {
    postOut({
      type: 'transcribe-done',
      sessionId,
      tokens: [],
      durationS: 0,
      inferenceMs: 0,
    });
    return;
  }
  const durationS = samples.length / TARGET_SAMPLE_RATE;
  log(`[batch] transcribing ${durationS.toFixed(1)}s of audio`);

  const t0 = performance.now();
  try {
    const result = await runWhisper(pipeline as any, samples, {
      language: isMultilingual(modelId) ? 'en' : undefined,
      offsetSeconds: 0,
    });
    const inferenceMs = performance.now() - t0;
    const tokens: TranscriptToken[] = result.words.map((w, i) => ({
      tokenId: i,
      text: w.text,
      t: w.start,
      isFinal: 1,
    }));
    log(`[batch] done in ${inferenceMs.toFixed(0)}ms: ${tokens.length} words`);
    postOut({ type: 'transcribe-done', sessionId, tokens, durationS, inferenceMs });
  } catch (err) {
    postOut({
      type: 'error',
      message: `batch transcribe failed: ${(err as Error).message}`,
    });
  }
}

self.onmessage = async (e: MessageEvent<InMessage>) => {
  const msg = e.data;
  if (msg.type === 'init') {
    try {
      await init(msg.modelId);
    } catch (err) {
      postOut({ type: 'error', message: `batch init failed: ${(err as Error).message}` });
    }
    return;
  }
  if (msg.type === 'transcribe') {
    await transcribe(msg.sessionId);
    return;
  }
};
