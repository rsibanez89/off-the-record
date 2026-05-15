export const TARGET_SAMPLE_RATE = 16_000;
export const CHUNK_DURATION_S = 1.0;

export const CHUNK_SAMPLES = TARGET_SAMPLE_RATE * CHUNK_DURATION_S;

// All entries use the `_timestamped` ONNX exports. Those export the cross-
// attention outputs that transformers.js needs to compute word-level
// timestamps via DTW. The non-timestamped variants throw "Model outputs must
// contain cross attentions to extract timestamps" when fed `return_timestamps:
// 'word'`, which our LocalAgreement-2 implementation requires.
export const MODELS = [
  { id: 'onnx-community/whisper-tiny.en_timestamped', label: 'tiny.en (fast, weaker)' },
  { id: 'onnx-community/whisper-base.en_timestamped', label: 'base.en' },
  { id: 'onnx-community/whisper-large-v3-turbo_timestamped', label: 'large-v3-turbo (best, multilingual)' },
] as const;

export type ModelId = (typeof MODELS)[number]['id'];

export const DEFAULT_MODEL: ModelId = 'onnx-community/whisper-base.en_timestamped';

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
