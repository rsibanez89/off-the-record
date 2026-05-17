// Public surface of the VAD subsystem. The producer worker imports from
// this barrel, never from concrete files. That isolates the concrete engine
// behind the `VadEngine` interface (DIP) and lets us swap implementations
// without touching the worker.

export type {
  VadEngine,
  FrameProbability,
  SpeechSegment,
  VadStateMachineConfig,
  ModelFetcher,
} from './types';
export { DEFAULT_STATE_MACHINE_CONFIG } from './types';

export { Framer } from './framer';
export { LinearResampler } from './resampler';
export { VadStateMachine, type VadStateMachineHandlers } from './stateMachine';
export { NoopVadEngine } from './noopVad';
export { SileroVad, type SileroVadOptions } from './silero/sileroVad';
export { HttpModelFetcher, sha256Hex, verifyHash } from './silero/sileroModel';
export {
  SR as SILERO_SR,
  FRAME as SILERO_FRAME,
  CONTEXT as SILERO_CONTEXT,
  STATE_SHAPE as SILERO_STATE_SHAPE,
  DEFAULT_MODEL_URL as SILERO_DEFAULT_MODEL_URL,
  MODEL_SHA256 as SILERO_MODEL_SHA256,
} from './silero/constants';

import type { VadEngine } from './types';
import { SileroVad } from './silero/sileroVad';

/**
 * Factory used by the producer worker. The worker calls this once and never
 * sees the concrete class. To swap engines (e.g. to TEN VAD), replace the
 * body of this function with the new constructor; nothing else needs to
 * change.
 *
 * DIP: producer code depends on `VadEngine` and on `createVadEngine`, not
 * on `SileroVad`.
 */
export function createVadEngine(): VadEngine {
  return new SileroVad();
}
