// Thin wrapper around transformers.js's Whisper pipeline. Hides the option
// shape, parses word-timestamp output into our `TimedWord[]` shape, and lets
// the worker code stay focused on orchestration.
//
// Also owns backend detection and pipeline instantiation, so both the live
// consumer worker and the batch worker share identical Whisper setup.

import type { TimedWord } from './hypothesisBuffer';
import { isHallucinationWord } from './heuristics';
import { isTurbo, type ModelId } from '../audio';

export type Backend = 'webgpu' | 'wasm';

export interface ProgressEntry {
  file: string;
  loaded: number;
  total: number;
  status: string;
}

export async function detectBackend(): Promise<Backend> {
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

export async function createPipeline(
  modelId: ModelId,
  backend: Backend,
  onProgress: (p: ProgressEntry) => void
): Promise<(samples: Float32Array, opts: Record<string, unknown>) => Promise<unknown>> {
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
  // ships a published fp16 encoder; smaller checkpoints either do not export
  // fp16 or measurably degrade accuracy on noisy/accented speech.
  // Decoder stays q4: that's where the speed win is.
  const turbo = isTurbo(modelId);
  const dtype =
    backend === 'webgpu'
      ? { encoder_model: turbo ? 'fp16' : 'fp32', decoder_model_merged: 'q4' }
      : { encoder_model: 'fp32', decoder_model_merged: 'q4' };
  const pipeline = await transformers.pipeline(
    'automatic-speech-recognition',
    modelId,
    {
      device: backend,
      dtype: dtype as any,
      session_options: { logSeverityLevel: 3 } as any,
      progress_callback: (p: any) => {
        if (p && typeof p === 'object') {
          onProgress({
            file: p.file ?? '',
            loaded: p.loaded ?? 0,
            total: p.total ?? 0,
            status: p.status ?? '',
          });
        }
      },
    }
  );
  return pipeline as any;
}

export interface WhisperRunResult {
  text: string;
  words: TimedWord[];
}

export interface WhisperRunOptions {
  /** ISO language code if the model is multilingual; undefined for `.en`. */
  language?: string;
  /**
   * Time offset (seconds) to add to every word timestamp. Whisper reports
   * timestamps relative to the audio passed in; the offset converts them into
   * the consumer's absolute chunk-startedAt timeline.
   */
  offsetSeconds: number;
}

interface WhisperChunk {
  text: string;
  timestamp: [number | null, number | null];
}

export async function runWhisper(
  pipeline: (samples: Float32Array, opts: Record<string, unknown>) => Promise<unknown>,
  samples: Float32Array,
  opts: WhisperRunOptions
): Promise<WhisperRunResult> {
  const callOpts: Record<string, unknown> = {
    chunk_length_s: 30,
    stride_length_s: 0,
    return_timestamps: 'word',
    no_repeat_ngram_size: 3,
    // Explicit greedy decoding. transformers.js usually defaults this way,
    // but being explicit avoids surprises across versions.
    top_k: 0,
    do_sample: false,
  };
  if (opts.language) {
    callOpts.language = opts.language;
    callOpts.task = 'transcribe';
  }
  const raw = await pipeline(samples, callOpts);
  return parseResult(raw, opts.offsetSeconds);
}

function parseResult(raw: unknown, offsetSeconds: number): WhisperRunResult {
  const obj = Array.isArray(raw) ? raw[0] : raw;
  const r = (obj ?? {}) as { text?: string; chunks?: WhisperChunk[] };
  const text = r.text ?? '';
  const chunks = Array.isArray(r.chunks) ? r.chunks : [];
  const words: TimedWord[] = [];
  for (const chunk of chunks) {
    const t = (chunk.text ?? '').trim();
    if (!t) continue;
    if (isHallucinationWord(t)) continue;
    const [start, end] = chunk.timestamp ?? [null, null];
    // Whisper sometimes returns nulls for word-level timestamps near the edge
    // of the audio window. Drop those words: their position is unreliable, so
    // LocalAgreement can't safely commit them.
    if (start == null || end == null) continue;
    words.push({
      text: t,
      start: start + offsetSeconds,
      end: end + offsetSeconds,
    });
  }
  return { text, words };
}
