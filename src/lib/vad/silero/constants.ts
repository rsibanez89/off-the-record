// Silero VAD v5 runtime contract. Source: the upstream ONNX C++ reference
// at `snakers4/silero-vad/examples/cpp/silero-vad-onnx.cpp`.

/** Required input sample rate. */
export const SR = 16_000;

/** Samples per inference. 512 / 16 kHz = 32 ms per frame. */
export const FRAME = 512;

/** Samples of past context prepended to each frame. The model's input shape
 *  is `[1, CONTEXT + FRAME]` = `[1, 576]`. */
export const CONTEXT = 64;

/** Shape of the recurrent state tensor. */
export const STATE_SHAPE: readonly number[] = [2, 1, 128];

/** Total floats in the state tensor. */
export const STATE_SIZE = STATE_SHAPE.reduce((a, b) => a * b, 1);

/** Default location served by Vite's `public/` directory at the app root. */
export const DEFAULT_MODEL_URL = '/models/silero_vad_v5.onnx';

/**
 * SHA-256 of the canonical `silero_vad_v5.onnx`, fetched once at build time
 * and verified by `scripts/fetch-models.mjs`. Mismatch is a hard failure.
 */
export const MODEL_SHA256 =
  'a4a068cd6cf1ea8355b84327595838ca748ec29a25bc91fc82e6c299ccdc5808';
