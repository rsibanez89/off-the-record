// NoopVadEngine: a `VadEngine` that returns 0 for every frame. Useful for
// tests and as the safe default if the real engine fails to load.
//
// LSP: must be a drop-in replacement for any other VadEngine, including
// SileroVad. Tests that exercise the producer worker's "VAD-aware" path can
// inject this and verify behavior with VAD always saying "no speech".

import type { VadEngine } from './types';

export class NoopVadEngine implements VadEngine {
  async initialize(): Promise<void> {
    // No-op.
  }
  async process(_frame: Float32Array): Promise<number> {
    void _frame;
    return 0;
  }
  reset(): void {
    // No-op.
  }
  async dispose(): Promise<void> {
    // No-op.
  }
}
