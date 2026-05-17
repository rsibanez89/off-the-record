// Public types for the in-house VAD subsystem.
//
// SOLID note: this file defines small, focused interfaces. Concrete classes
// live in `silero/`, `framer.ts`, `stateMachine.ts`, `resampler.ts`. The
// producer worker imports `VadEngine` and `createVadEngine`, never the
// concrete `SileroVad`. That lets us swap in TEN VAD or any other engine
// later by writing one new class plus one factory change.

/**
 * A single speech-probability sample emitted by a VadEngine for one frame.
 * Time is the absolute (chunk-startedAt) timeline used by the rest of the app.
 */
export interface FrameProbability {
  /** Speech probability in [0, 1]. */
  probability: number;
  /** Frame start time in seconds, absolute timeline. */
  startS: number;
  /** Frame end time in seconds, absolute timeline. */
  endS: number;
}

/**
 * A detected speech segment, produced by `VadStateMachine`. Times are in the
 * same absolute timeline as the input `FrameProbability`s.
 */
export interface SpeechSegment {
  startS: number;
  endS: number;
}

/**
 * The single VAD engine interface used by the producer worker. Keep it
 * narrow on purpose: initialize, process one frame, reset state, dispose.
 * Anything else (batch, event subscription, options surface) belongs on a
 * separate interface (see ISP note in `index.ts`).
 */
export interface VadEngine {
  /** Load model and any state. Idempotent. */
  initialize(): Promise<void>;
  /**
   * Run inference on one 16 kHz, 512-sample frame. Returns the speech
   * probability in [0, 1]. The engine threads its recurrent state across
   * calls internally.
   */
  process(frame: Float32Array): Promise<number>;
  /** Zero out recurrent state. Called between recording sessions. */
  reset(): void;
  /** Release any underlying resources. After dispose, the engine is dead. */
  dispose(): Promise<void>;
}

/**
 * Hysteresis state-machine configuration. Conventional Silero v5 defaults:
 * positive 0.3, negative 0.25, redemption 1400 ms, preSpeechPad 800 ms,
 * minSpeech 400 ms. Reasonable for Whisper gating; tune per recording
 * environment if needed.
 */
export interface VadStateMachineConfig {
  /** Frame above this is considered "speech" for entry. */
  positiveSpeechThreshold: number;
  /** Frame below this counts toward redemption (leaving speech). */
  negativeSpeechThreshold: number;
  /** How long sub-threshold frames must persist before ending a segment. */
  redemptionMs: number;
  /** How much pre-speech audio to prepend when a segment opens. */
  preSpeechPadMs: number;
  /** Segments shorter than this fire `onMisfire` instead of `onSpeechEnd`. */
  minSpeechMs: number;
  /** Frame duration in ms. At 16 kHz, 512-sample frames, this is 32 ms. */
  frameMs: number;
}

export const DEFAULT_STATE_MACHINE_CONFIG: VadStateMachineConfig = {
  positiveSpeechThreshold: 0.3,
  negativeSpeechThreshold: 0.25,
  redemptionMs: 1400,
  preSpeechPadMs: 800,
  minSpeechMs: 400,
  frameMs: 32,
};

/**
 * Indirection point for fetching the Silero ONNX model. Tests pass a fake
 * `ModelFetcher` so they never hit the network or disk. Production uses
 * `httpModelFetcher` which goes to `/models/silero_vad_v5.onnx`.
 *
 * DIP note: `SileroVad` depends on this interface, not on `fetch`.
 */
export interface ModelFetcher {
  fetch(url: string): Promise<ArrayBuffer>;
}
