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
