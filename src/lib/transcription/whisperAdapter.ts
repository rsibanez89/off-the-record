// Thin wrapper around transformers.js's Whisper pipeline. Hides the option
// shape, parses word-timestamp output into our `TimedWord[]` shape, and lets
// the worker code stay focused on orchestration.
//
// Also owns backend detection and pipeline instantiation, so both the live
// consumer worker and the batch worker share identical Whisper setup.

import type { TimedWord } from './hypothesisBuffer';
import { isHallucinationWord } from './heuristics';
import { isDistil, isMoonshine, isTurbo, type ModelId } from '../audio';

export type Backend = 'webgpu' | 'wasm';

export interface ProgressEntry {
  file: string;
  loaded: number;
  total: number;
  status: string;
}

/**
 * The transformers.js Whisper pipeline is callable AND carries auxiliary
 * objects (tokenizer, processor, model) as properties. We type both surfaces
 * so the adapter can tokenize prompts without leaking `any` to call sites.
 */
export interface WhisperTokenizer {
  encode(text: string, options?: { add_special_tokens?: boolean }): number[];
}

export interface WhisperPipeline {
  (samples: Float32Array, opts: Record<string, unknown>): Promise<unknown>;
  tokenizer?: WhisperTokenizer;
  // Other properties (processor, model) exist but are not used here.
}

/** Whisper's prompt window is ~224 tokens. Stay safely under that. */
const MAX_PROMPT_TOKENS = 200;

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
): Promise<WhisperPipeline> {
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
  // Encoder dtype:
  //   - Turbo on WebGPU has a published fp16 encoder; we use it for speed.
  //   - Distil-large-v3.5 fp32 encoder is ~500 MB single contiguous, which
  //     transformers.js cannot allocate on a typical fragmented browser
  //     heap (RangeError: Array buffer allocation failed). fp16 (~250 MB)
  //     halves the size while keeping precision the encoder benefits
  //     from. Smaller dtypes (int8, q4) were tried: int8 needs a
  //     ConvInteger operator that ORT-node does not ship; q4 is
  //     state-of-the-art-doc-flagged as degrading accuracy on noisy or
  //     accented speech. fp16 is the safest middle ground.
  //   - tiny.en stays fp32 (small enough, no allocation pressure).
  //   - Moonshine ships pre-quantized encoder+decoder files. fp32 here
  //     would 404 because the matching files are not published.
  // Decoder stays q4 across Whisper-family models for the speed win.
  const turbo = isTurbo(modelId);
  const moonshine = isMoonshine(modelId);
  const distil = isDistil(modelId);
  let dtype: Record<string, string>;
  if (moonshine) {
    dtype = { encoder_model: 'q8', decoder_model_merged: 'q8' };
  } else if (distil) {
    dtype = { encoder_model: 'fp16', decoder_model_merged: 'q4' };
  } else if (backend === 'webgpu') {
    dtype = { encoder_model: turbo ? 'fp16' : 'fp32', decoder_model_merged: 'q4' };
  } else {
    dtype = { encoder_model: 'fp32', decoder_model_merged: 'q4' };
  }
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
  return pipeline as unknown as WhisperPipeline;
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
  /**
   * Optional text prompt passed to Whisper as preceding context. Tokenised
   * and capped at `MAX_PROMPT_TOKENS` (~200). Used by the live consumer to
   * feed the tail of already-committed transcript so the model continues
   * the user's actual sentence instead of confabulating a fresh narrative
   * (a common large-v3-turbo failure mode on isolated short windows).
   */
  initialPrompt?: string;
  /**
   * Whether to request `return_timestamps: 'word'` from the pipeline.
   * Defaults to true (the live LA-2 path always needs it). Callers that
   * use a model whose ONNX export does not surface cross-attention (e.g.
   * `distil-whisper/distil-large-v3.5-ONNX`, `onnx-community/moonshine-*`)
   * must pass false; otherwise transformers.js throws "Model outputs must
   * contain cross attentions to extract timestamps". `parseResult` falls
   * through to the text-only synthesis path in that case.
   */
  requestWordTimestamps?: boolean;
}

interface WhisperChunk {
  text: string;
  timestamp: [number | null, number | null];
}

export async function runWhisper(
  pipeline: WhisperPipeline,
  samples: Float32Array,
  opts: WhisperRunOptions
): Promise<WhisperRunResult> {
  const requestWordTimestamps = opts.requestWordTimestamps ?? true;
  const callOpts: Record<string, unknown> = {
    chunk_length_s: 30,
    stride_length_s: 0,
    no_repeat_ngram_size: 3,
    // Explicit greedy decoding. transformers.js usually defaults this way,
    // but being explicit avoids surprises across versions.
    top_k: 0,
    do_sample: false,
  };
  if (requestWordTimestamps) {
    callOpts.return_timestamps = 'word';
  }
  if (opts.language) {
    callOpts.language = opts.language;
    callOpts.task = 'transcribe';
  }
  if (opts.initialPrompt && pipeline.tokenizer) {
    const promptIds = encodePromptIds(pipeline.tokenizer, opts.initialPrompt);
    if (promptIds.length > 0) {
      callOpts.prompt_ids = promptIds;
    }
  }
  const raw = await pipeline(samples, callOpts);
  const audioDurationS = samples.length / 16_000;
  const parsed = parseResult(raw, opts.offsetSeconds, audioDurationS);
  if (opts.initialPrompt) {
    // Whisper sometimes regurgitates the entire prompt verbatim at the head
    // of its output instead of using it purely as conditioning context. The
    // downstream LA-2 dedup compares committed-tail against new-head, so it
    // cannot strip a long whole-prompt re-emission. Strip it here, where we
    // know exactly what prompt we passed, before LA-2 ever sees the words.
    parsed.words = stripLeadingPromptRegurgitation(parsed.words, opts.initialPrompt);
  }
  return parsed;
}

const PROMPT_MATCH_MIN = 3;

/**
 * If the head of `words` matches the start of `prompt`, strip those words.
 * Comparison is case- and punctuation-insensitive. Requires at least
 * `PROMPT_MATCH_MIN` (3) matching tokens so an incidental single-word
 * coincidence (e.g. "the") does not accidentally swallow real content.
 */
function stripLeadingPromptRegurgitation(words: TimedWord[], prompt: string): TimedWord[] {
  if (words.length === 0) return words;
  const promptTokens = prompt
    .split(/\s+/)
    .map(normPromptWord)
    .filter((s) => s.length > 0);
  if (promptTokens.length === 0) return words;
  let i = 0;
  while (i < words.length && i < promptTokens.length) {
    if (normPromptWord(words[i].text) !== promptTokens[i]) break;
    i++;
  }
  return i >= PROMPT_MATCH_MIN ? words.slice(i) : words;
}

function normPromptWord(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9']/g, '');
}

/**
 * Tokenise a prompt and keep only the tail that fits inside Whisper's prompt
 * window. Special tokens are deliberately omitted: Whisper's generate logic
 * adds `<|startofprev|>` etc. around the prompt itself.
 */
function encodePromptIds(tokenizer: WhisperTokenizer, prompt: string): number[] {
  const trimmed = prompt.trim();
  if (!trimmed) return [];
  let ids: number[];
  try {
    ids = tokenizer.encode(trimmed, { add_special_tokens: false });
  } catch {
    // If the tokenizer is unavailable or the call shape differs across
    // transformers.js versions, fail soft: no prompt is better than a crash.
    return [];
  }
  if (ids.length <= MAX_PROMPT_TOKENS) return ids;
  return ids.slice(ids.length - MAX_PROMPT_TOKENS);
}

function parseResult(
  raw: unknown,
  offsetSeconds: number,
  audioDurationS: number,
): WhisperRunResult {
  const obj = Array.isArray(raw) ? raw[0] : raw;
  const r = (obj ?? {}) as { text?: string; chunks?: WhisperChunk[] };
  const text = r.text ?? '';
  const chunks = Array.isArray(r.chunks) ? r.chunks : [];

  if (chunks.length > 0) {
    const words: TimedWord[] = [];
    for (const chunk of chunks) {
      const t = (chunk.text ?? '').trim();
      if (!t) continue;
      if (isHallucinationWord(t)) continue;
      const [start, end] = chunk.timestamp ?? [null, null];
      // Whisper sometimes returns nulls for word-level timestamps near the
      // edge of the audio window. Drop those words: their position is
      // unreliable, so LocalAgreement can't safely commit them.
      if (start == null || end == null) continue;
      words.push({
        text: t,
        start: start + offsetSeconds,
        end: end + offsetSeconds,
      });
    }
    return { text, words };
  }

  // Fallback path: some models (e.g. Moonshine via transformers.js) return
  // text-only output with no `chunks` array. Synthesise per-word entries
  // from whitespace tokenisation, distributing timestamps evenly across
  // `audioDurationS` so downstream consumers (batch panel rendering, WER
  // scoring) still have something to work with. These models are filtered
  // out of the live picker (`supportsWordTimestamps: false`) because the
  // synthesised timestamps are not accurate enough for LA-2 agreement,
  // but the batch panel still displays a useful transcript.
  const tokens = text
    .split(/\s+/)
    .map((t) => t.trim())
    .filter((t) => t.length > 0 && !isHallucinationWord(t));
  if (tokens.length === 0) return { text, words: [] };
  const slot = audioDurationS > 0 ? audioDurationS / tokens.length : 0;
  const synthesised: TimedWord[] = tokens.map((t, i) => ({
    text: t,
    start: offsetSeconds + i * slot,
    end: offsetSeconds + (i + 1) * slot,
  }));
  return { text, words: synthesised };
}
