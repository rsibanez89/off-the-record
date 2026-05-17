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

// All entries use the `_timestamped` ONNX exports. Those export the cross-
// attention outputs that transformers.js needs to compute word-level
// timestamps via DTW. The non-timestamped variants throw "Model outputs must
// contain cross attentions to extract timestamps" when fed `return_timestamps:
// 'word'`, which our LocalAgreement-2 implementation requires.
//
// Labels are panel-aware on purpose: a "best, multilingual" tag on the live
// panel was misleading and pushed users onto turbo for live, where it
// confabulates story-like prefixes on isolated short windows.
export const MODELS = [
  {
    id: 'onnx-community/whisper-tiny.en_timestamped',
    liveLabel: 'tiny.en (fastest, low hallucination, English)',
    batchLabel: 'tiny.en (fastest, English)',
  },
  {
    id: 'onnx-community/whisper-base.en_timestamped',
    liveLabel: 'base.en (recommended for live, English)',
    batchLabel: 'base.en (English)',
  },
  {
    id: 'onnx-community/whisper-large-v3-turbo_timestamped',
    liveLabel: 'large-v3-turbo (multilingual, may hallucinate on short windows)',
    batchLabel: 'large-v3-turbo (recommended for batch, multilingual)',
  },
] as const;

export type ModelId = (typeof MODELS)[number]['id'];

export function modelLabel(id: ModelId, role: ModelPanelRole): string {
  const m = MODELS.find((entry) => entry.id === id);
  if (!m) return id;
  return role === 'live' ? m.liveLabel : m.batchLabel;
}

/** Default model when the user has no localStorage preference yet. */
export const DEFAULT_MODEL: ModelId = 'onnx-community/whisper-base.en_timestamped';
/**
 * Per-panel first-run defaults. Live prefers the smaller English model to
 * keep LocalAgreement-2 settling fast and to suppress turbo's confabulation
 * on isolated short windows; batch prefers turbo for raw accuracy.
 */
export const DEFAULT_LIVE_MODEL: ModelId = 'onnx-community/whisper-base.en_timestamped';
export const DEFAULT_BATCH_MODEL: ModelId = 'onnx-community/whisper-large-v3-turbo_timestamped';

export function isValidModel(id: string | null): id is ModelId {
  return !!id && MODELS.some((m) => m.id === id);
}

/**
 * `.en` variants are English-only and don't carry language/task tokens, so
 * passing them throws in transformers.js. Multilingual checkpoints (only
 * large-v3-turbo in our picker) need explicit language/task.
 */
export function isMultilingual(id: ModelId): boolean {
  // Strip the `_timestamped` suffix when checking, otherwise every model
  // appears multilingual.
  return !id.replace(/_timestamped$/, '').endsWith('.en');
}

export function isTurbo(id: ModelId): boolean {
  return id.startsWith('onnx-community/whisper-large-v3-turbo');
}
