/// <reference lib="webworker" />
import { type TranscriptToken } from '../lib/db';
import {
  TARGET_SAMPLE_RATE,
  DEFAULT_MODEL,
  isMultilingual,
  supportsWordTimestamps,
  type ModelId,
} from '../lib/audio';
import { type Backend } from '../lib/transcription/whisperAdapter';
import { WhisperEngine } from '../lib/transcription/whisperEngine';
import { DexieAudioArchiveRepository } from '../lib/repositories/dexieRepositories';

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
  | { type: 'transcribe-start'; sessionId: number }
  | {
      type: 'transcribe-done';
      sessionId: number;
      tokens: TranscriptToken[];
      durationS: number;
      inferenceMs: number;
    };

declare const self: DedicatedWorkerGlobalScope;

let engine: WhisperEngine | null = null;
let modelId: ModelId = DEFAULT_MODEL;
const audioArchive = new DexieAudioArchiveRepository();

function postOut(msg: OutMessage) {
  self.postMessage(msg);
}

function log(message: string) {
  console.log(message);
}

async function init(id: ModelId) {
  modelId = id;
  engine = new WhisperEngine(modelId);
  await engine.load((p) => {
    postOut({ type: 'progress', ...p });
  });
  postOut({ type: 'ready', backend: engine.getBackend() });
}

async function transcribe(sessionId: number) {
  if (!engine) {
    postOut({ type: 'error', message: 'batch pipeline not initialised yet' });
    return;
  }

  postOut({ type: 'transcribe-start', sessionId });
  const samples = await audioArchive.toFloat32();
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
    // Batch transcribes the full audio in one pass with no preceding context,
    // so we deliberately do NOT pass `initialPrompt`. Live uses it; batch
    // benefits from a clean run without confabulation guards interfering.
    const result = await engine.run(samples, {
      language: isMultilingual(modelId) ? 'en' : undefined,
      offsetSeconds: 0,
      // Models without cross-attention exports (distil-large-v3.5,
      // moonshine) cannot satisfy `return_timestamps: 'word'`. The text-
      // only fallback in `parseResult` synthesises evenly-distributed
      // word entries so the batch panel still renders a transcript.
      requestWordTimestamps: supportsWordTimestamps(modelId),
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
