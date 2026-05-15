export const TARGET_SAMPLE_RATE = 16_000;
export const CHUNK_DURATION_S = 1.0;

export const CHUNK_SAMPLES = TARGET_SAMPLE_RATE * CHUNK_DURATION_S;

export const MODELS = [
  { id: 'onnx-community/whisper-tiny.en', label: 'tiny.en (fast, weaker)' },
  { id: 'onnx-community/whisper-base.en', label: 'base.en' },
  { id: 'onnx-community/whisper-large-v3-turbo', label: 'large-v3-turbo (best, multilingual)' },
] as const;

export type ModelId = (typeof MODELS)[number]['id'];

export const DEFAULT_MODEL: ModelId = 'onnx-community/whisper-base.en';

export function isValidModel(id: string | null): id is ModelId {
  return !!id && MODELS.some((m) => m.id === id);
}

/**
 * `.en` variants are English-only and don't carry language/task tokens, so
 * passing them throws in transformers.js. Multilingual checkpoints (only
 * large-v3-turbo in our picker) need explicit language/task.
 */
export function isMultilingual(id: ModelId): boolean {
  return !id.endsWith('.en');
}
