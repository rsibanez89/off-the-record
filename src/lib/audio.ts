export const TARGET_SAMPLE_RATE = 16_000;
export const CHUNK_DURATION_S = 1.0;

export const CHUNK_SAMPLES = TARGET_SAMPLE_RATE * CHUNK_DURATION_S;

/**
 * Panel role for label disambiguation. Live and batch have different sweet
 * spots: tiny.en is the best LIVE option on most hardware because it
 * hallucinates less and infers fast enough for LocalAgreement-2 to stabilise.
 * large-v3-turbo is the best BATCH option because it sees the whole audio
 * at once with no streaming pressure.
 */
export type ModelPanelRole = 'live' | 'batch';

// Whisper entries use the `_timestamped` ONNX exports because LA-2 needs
// per-word timestamps via DTW from cross-attention outputs. Models flagged
// `supportsWordTimestamps: false` are hidden from the live panel (LA-2
// requires word timing) and only offered in the batch panel for
// comparison: distil and moonshine fall in this bucket, distil because
// the upstream `distil-whisper/distil-large-v3.5-ONNX` export does not
// document cross-attention surfacing, moonshine because the transformers.js
// pipeline only emits text-level output for it.
//
// Labels are panel-aware on purpose: a "best, multilingual" tag on the live
// panel was misleading and pushed users onto turbo for live, where it
// confabulates story-like prefixes on isolated short windows.
export const MODELS = [
  {
    id: 'onnx-community/whisper-tiny.en_timestamped',
    liveLabel: 'tiny.en (fastest, low hallucination, English)',
    batchLabel: 'tiny.en (fastest, English)',
    supportsWordTimestamps: true,
  },
  {
    id: 'distil-whisper/distil-large-v3.5-ONNX',
    liveLabel: 'distil-large-v3.5 (batch-only, not yet live-capable)',
    batchLabel: 'distil-large-v3.5 (HF/distil, ~500 MB, beats turbo on OOD WER)',
    supportsWordTimestamps: false,
  },
  {
    id: 'onnx-community/moonshine-base-ONNX',
    liveLabel: 'moonshine-base (batch-only, streaming architecture, no word timestamps)',
    batchLabel: 'moonshine-base (UsefulSensors, ~120 MB, native streaming encoder)',
    supportsWordTimestamps: false,
  },
  {
    id: 'onnx-community/whisper-large-v3-turbo_timestamped',
    liveLabel: 'large-v3-turbo (multilingual, may hallucinate on short windows)',
    batchLabel: 'large-v3-turbo (recommended for batch, multilingual)',
    supportsWordTimestamps: true,
  },
] as const;

export type ModelId = (typeof MODELS)[number]['id'];

export function modelLabel(id: ModelId, role: ModelPanelRole): string {
  const m = MODELS.find((entry) => entry.id === id);
  if (!m) return id;
  return role === 'live' ? m.liveLabel : m.batchLabel;
}

/**
 * Filter the picker by panel role. Live hides models that lack word-level
 * timestamp support (LA-2 needs them); batch shows everything.
 */
export function modelsForRole(role: ModelPanelRole): typeof MODELS[number][] {
  if (role === 'batch') return MODELS.slice();
  return MODELS.filter((m) => m.supportsWordTimestamps);
}

/** Default model when the user has no localStorage preference yet. */
export const DEFAULT_MODEL: ModelId = 'onnx-community/whisper-tiny.en_timestamped';
/**
 * Per-panel first-run defaults. Live prefers the smaller English model to
 * keep LocalAgreement-2 settling fast and to suppress turbo's confabulation
 * on isolated short windows; batch prefers turbo for raw accuracy.
 */
export const DEFAULT_LIVE_MODEL: ModelId = 'onnx-community/whisper-tiny.en_timestamped';
export const DEFAULT_BATCH_MODEL: ModelId = 'onnx-community/whisper-large-v3-turbo_timestamped';

export function isValidModel(id: string | null): id is ModelId {
  return !!id && MODELS.some((m) => m.id === id);
}

/**
 * True when the model's ONNX export emits cross-attention outputs (so
 * transformers.js can compute per-word timestamps via DTW). Callers of
 * `runWhisper` should pass the result as `requestWordTimestamps` so
 * unsupported models do not trip the "Model outputs must contain cross
 * attentions to extract timestamps" error.
 */
export function supportsWordTimestamps(id: ModelId): boolean {
  const m = MODELS.find((entry) => entry.id === id);
  return m?.supportsWordTimestamps ?? false;
}

/**
 * `.en` variants are English-only and don't carry language/task tokens, so
 * passing them throws in transformers.js. Multilingual checkpoints (turbo,
 * distil-large-v3.5, moonshine) need explicit language/task.
 */
export function isMultilingual(id: ModelId): boolean {
  // Strip the `_timestamped` suffix when checking, otherwise every model
  // appears multilingual.
  return !id.replace(/_timestamped$/, '').endsWith('.en');
}

export function isTurbo(id: ModelId): boolean {
  return id.startsWith('onnx-community/whisper-large-v3-turbo');
}

/**
 * Distil-Whisper checkpoints keep the full Whisper encoder, so fp16 on
 * WebGPU is not always safe. Use fp32 encoder, q4 decoder (same recipe as
 * non-turbo Whisper checkpoints).
 */
export function isDistil(id: ModelId): boolean {
  return id.startsWith('distil-whisper/');
}

/**
 * Moonshine v2 has a different pipeline shape than Whisper (sliding-window
 * encoder, no 30s padding) and the transformers.js pipeline currently emits
 * text-only output. Surface this so dtype branching and adapter logic can
 * special-case it. Marked batch-only above until word-level timestamps land
 * upstream.
 */
export function isMoonshine(id: ModelId): boolean {
  return id.startsWith('onnx-community/moonshine');
}
