import { describe, expect, it } from 'vitest';
import { SileroVad, type SileroOrtSession, type SileroOrtTensor } from './sileroVad';
import { CONTEXT, FRAME, STATE_SIZE } from './constants';
import type { ModelFetcher } from '../types';

class FakeFetcher implements ModelFetcher {
  async fetch(_url: string): Promise<ArrayBuffer> {
    // Return a tiny non-empty buffer; the fake session ignores its contents.
    return new ArrayBuffer(4);
  }
}

interface RecordedRun {
  inputData: Float32Array;
  stateData: Float32Array;
  srData: BigInt64Array;
}

/**
 * Fake ORT session that:
 *   - records every input, state, and sr tensor it receives
 *   - increments an internal counter to make the returned state distinguishable
 *     per call (so the test can assert state is threaded through)
 *   - returns a deterministic probability based on the call index
 */
function makeFakeSession(): { session: SileroOrtSession; runs: RecordedRun[] } {
  const runs: RecordedRun[] = [];
  let callIndex = 0;
  const session: SileroOrtSession = {
    async run(feeds: Record<string, unknown>) {
      const input = feeds.input as { data: Float32Array };
      const state = feeds.state as { data: Float32Array };
      const sr = feeds.sr as { data: BigInt64Array };
      runs.push({
        inputData: new Float32Array(input.data),
        stateData: new Float32Array(state.data),
        srData: new BigInt64Array(sr.data),
      });
      // Produce a "new state" that encodes the call index, so the next call
      // receives a recognisably different state buffer.
      const newState = new Float32Array(STATE_SIZE);
      newState[0] = callIndex + 1;
      const prob = new Float32Array([0.1 + callIndex * 0.1]);
      callIndex++;
      const out: Record<string, SileroOrtTensor> = {
        stateN: { data: newState },
        output: { data: prob },
      };
      return out;
    },
  };
  return { session, runs };
}

describe('SileroVad', () => {
  it('runs inference with the correct input shape and sr', async () => {
    const { session, runs } = makeFakeSession();
    const vad = new SileroVad({
      fetcher: new FakeFetcher(),
      sessionFactory: async () => session,
      tensorFactory: (_type, data) => ({ data }),
    });
    await vad.initialize();
    const frame = new Float32Array(FRAME);
    for (let i = 0; i < FRAME; i++) frame[i] = (i % 100) / 100;
    const p = await vad.process(frame);
    expect(typeof p).toBe('number');
    expect(p).toBeCloseTo(0.1, 6);

    expect(runs.length).toBe(1);
    expect(runs[0].inputData.length).toBe(CONTEXT + FRAME);
    expect(runs[0].srData[0]).toBe(BigInt(16000));
    // First call: context is all zeros.
    for (let i = 0; i < CONTEXT; i++) expect(runs[0].inputData[i]).toBe(0);
    // Frame samples follow context.
    for (let i = 0; i < FRAME; i++) expect(runs[0].inputData[CONTEXT + i]).toBe(frame[i]);
  });

  it('threads state across consecutive calls', async () => {
    const { session, runs } = makeFakeSession();
    const vad = new SileroVad({
      fetcher: new FakeFetcher(),
      sessionFactory: async () => session,
      tensorFactory: (_type, data) => ({ data }),
    });
    await vad.initialize();
    const frame = new Float32Array(FRAME);
    await vad.process(frame);
    await vad.process(frame);
    await vad.process(frame);

    expect(runs.length).toBe(3);
    // First call: state is all zeros.
    expect(runs[0].stateData[0]).toBe(0);
    // Second call: state[0] should be 1 (output of call 1).
    expect(runs[1].stateData[0]).toBe(1);
    // Third call: state[0] should be 2 (output of call 2).
    expect(runs[2].stateData[0]).toBe(2);
  });

  it('carries the last CONTEXT samples of each frame as the next call context', async () => {
    const { session, runs } = makeFakeSession();
    const vad = new SileroVad({
      fetcher: new FakeFetcher(),
      sessionFactory: async () => session,
      tensorFactory: (_type, data) => ({ data }),
    });
    await vad.initialize();
    const frame1 = new Float32Array(FRAME);
    for (let i = 0; i < FRAME; i++) frame1[i] = i + 1; // distinct, non-zero
    const frame2 = new Float32Array(FRAME);
    for (let i = 0; i < FRAME; i++) frame2[i] = 1000 + i;
    await vad.process(frame1);
    await vad.process(frame2);

    expect(runs.length).toBe(2);
    // Second call's context must be the last CONTEXT samples of frame1.
    const expectedContext = frame1.subarray(FRAME - CONTEXT);
    for (let i = 0; i < CONTEXT; i++) {
      expect(runs[1].inputData[i]).toBe(expectedContext[i]);
    }
    // And the second frame's samples follow.
    for (let i = 0; i < FRAME; i++) {
      expect(runs[1].inputData[CONTEXT + i]).toBe(frame2[i]);
    }
  });

  it('reset zeros both state and context for the next call', async () => {
    const { session, runs } = makeFakeSession();
    const vad = new SileroVad({
      fetcher: new FakeFetcher(),
      sessionFactory: async () => session,
      tensorFactory: (_type, data) => ({ data }),
    });
    await vad.initialize();
    const frame = new Float32Array(FRAME);
    for (let i = 0; i < FRAME; i++) frame[i] = i + 1;
    await vad.process(frame);
    vad.reset();
    await vad.process(frame);
    expect(runs.length).toBe(2);
    // After reset, the second call's context should be all zeros again.
    for (let i = 0; i < CONTEXT; i++) expect(runs[1].inputData[i]).toBe(0);
    expect(runs[1].stateData[0]).toBe(0);
  });

  it('rejects process before initialize', async () => {
    const vad = new SileroVad({
      fetcher: new FakeFetcher(),
      sessionFactory: async () => makeFakeSession().session,
      tensorFactory: (_type, data) => ({ data }),
    });
    await expect(vad.process(new Float32Array(FRAME))).rejects.toThrow(/before initialize/);
  });

  it('rejects wrong-sized frames', async () => {
    const { session } = makeFakeSession();
    const vad = new SileroVad({
      fetcher: new FakeFetcher(),
      sessionFactory: async () => session,
      tensorFactory: (_type, data) => ({ data }),
    });
    await vad.initialize();
    await expect(vad.process(new Float32Array(100))).rejects.toThrow(/expected 512 samples/);
  });

  it('dispose makes process fail again', async () => {
    const { session } = makeFakeSession();
    const vad = new SileroVad({
      fetcher: new FakeFetcher(),
      sessionFactory: async () => session,
      tensorFactory: (_type, data) => ({ data }),
    });
    await vad.initialize();
    await vad.process(new Float32Array(FRAME));
    await vad.dispose();
    await expect(vad.process(new Float32Array(FRAME))).rejects.toThrow(/before initialize/);
  });
});
