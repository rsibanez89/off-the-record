// Centralised live-transcription thresholds. Add new tunables here as
// they appear. Workers should import from this file rather than redefining
// constants locally.
//
// Open / Closed: future per-model overrides or A/B tests can wrap these
// without touching call sites.

/**
 * Speech-probability cutoff for the VAD-aware silence check. Chunks whose
 * aggregate Silero speech probability falls below this value are treated as
 * silence and bypass Whisper. Aligned with the conventional Silero v5
 * positive-speech threshold of 0.3.
 */
export const VAD_SILENCE_THRESHOLD = 0.3;

// Live-loop window-size policy. See PLAN.md "Live transcription design
// invariants". MIN_WINDOW_S = 0 means "run on the first chunk available"
// so the live panel gives feedback fast; LA-2 revises junk on subsequent
// ticks. MIN_DRAIN_WINDOW_S = 0 mirrors that on Stop.
export const MIN_WINDOW_S = 0;
export const MIN_DRAIN_WINDOW_S = 0;

/**
 * Hard safety net. If the window has grown beyond this many seconds without
 * any natural commit (continuous unstable speech), force-slide: commit
 * tentative and trim the anchor.
 */
export const MAX_WINDOW_S = 24.0;

/**
 * Audio anchor advancement: preserve intra-sentence context for accuracy.
 * Trim only on sentence ends, except force-trim past FAST_TRIM_THRESHOLD_S
 * and force-slide past MAX_WINDOW_S as safety nets. CONTEXT_LOOKBACK_S
 * controls how much already-transcribed audio remains in the next window.
 */
export const CONTEXT_LOOKBACK_S = 5.0;
export const FAST_TRIM_THRESHOLD_S = 10.0;

/**
 * Drain loop safety: cap ticks at Stop so a misbehaving model cannot stall
 * the flush forever.
 */
export const DRAIN_MAX_ITERATIONS = 20;

/**
 * Prompt length cap for `initial_prompt` (characters as a cheap proxy for
 * Whisper's ~224-token prompt window; the adapter further caps by token
 * count).
 */
export const MAX_PROMPT_CHARS = 800;
