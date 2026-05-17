import { describe, expect, it } from 'vitest';
import { NoopVadEngine } from './noopVad';
import { SILERO_FRAME } from './index';
import type { VadEngine } from './types';

describe('NoopVadEngine', () => {
  it('honours the VadEngine contract: returns a probability in [0, 1] for any input', async () => {
    // Substantive LSP check: any VadEngine, including this stub, must accept
    // a Float32Array frame and return a number in the closed unit interval,
    // including when fed adversarial signals (silence, large positive, large
    // negative, NaN). The producer worker relies on this invariant.
    const engine: VadEngine = new NoopVadEngine();
    await engine.initialize();
    const signals: Float32Array[] = [
      new Float32Array(SILERO_FRAME),                                   // all zeros (silence)
      new Float32Array(SILERO_FRAME).fill(1),                           // full-scale positive
      new Float32Array(SILERO_FRAME).fill(-1),                          // full-scale negative
      new Float32Array(SILERO_FRAME).map((_, i) => Math.sin(i / 8)),    // tone
      new Float32Array(SILERO_FRAME).map(() => Math.random() * 2 - 1),  // noise
      new Float32Array(SILERO_FRAME).fill(NaN),                         // adversarial
    ];
    for (const s of signals) {
      const p = await engine.process(s);
      expect(typeof p).toBe('number');
      expect(Number.isFinite(p)).toBe(true);
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }
  });

  it('reset / dispose are safe no-ops', async () => {
    const engine = new NoopVadEngine();
    await engine.initialize();
    engine.reset();
    await engine.dispose();
    // No assertions; just must not throw.
  });
});
