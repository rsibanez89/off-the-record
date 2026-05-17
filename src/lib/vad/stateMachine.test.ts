import { describe, expect, it } from 'vitest';
import { VadStateMachine, type VadStateMachineHandlers } from './stateMachine';
import type { FrameProbability, SpeechSegment, VadStateMachineConfig } from './types';

interface CapturedHandlers extends VadStateMachineHandlers {
  starts: SpeechSegment[];
  ends: SpeechSegment[];
  misfires: SpeechSegment[];
}

function captureHandlers(): CapturedHandlers {
  const starts: SpeechSegment[] = [];
  const ends: SpeechSegment[] = [];
  const misfires: SpeechSegment[] = [];
  return {
    starts,
    ends,
    misfires,
    onSpeechStart: (s) => starts.push(s),
    onSpeechEnd: (s) => ends.push(s),
    onMisfire: (s) => misfires.push(s),
  };
}

function feed(sm: VadStateMachine, probs: number[], frameMs: number, startS = 0): void {
  let t = startS;
  for (const p of probs) {
    const f: FrameProbability = {
      probability: p,
      startS: t,
      endS: t + frameMs / 1000,
    };
    sm.ingest(f);
    t += frameMs / 1000;
  }
}

const TEST_CFG: Partial<VadStateMachineConfig> = {
  positiveSpeechThreshold: 0.5,
  negativeSpeechThreshold: 0.4,
  redemptionMs: 96, // 3 frames at 32 ms
  preSpeechPadMs: 64, // 2 frames at 32 ms
  minSpeechMs: 64, // 2 frames at 32 ms
  frameMs: 32,
};

describe('VadStateMachine', () => {
  it('fires onSpeechStart on positive threshold crossing', () => {
    const h = captureHandlers();
    const sm = new VadStateMachine(TEST_CFG, h);
    feed(sm, [0.1, 0.1, 0.8], 32, 0);
    expect(h.starts.length).toBe(1);
    expect(h.ends.length).toBe(0);
    expect(sm.getState()).toBe('speaking');
  });

  it('prepends pre-speech padding to the segment start', () => {
    const h = captureHandlers();
    const sm = new VadStateMachine(TEST_CFG, h);
    // 4 frames of silence, then one speech frame at t=0.128s.
    feed(sm, [0.1, 0.1, 0.1, 0.1, 0.9], 32, 0);
    expect(h.starts.length).toBe(1);
    // preSpeechPadMs=64ms = 2 frames; segment must start at t = 0.128 - 0.064 = 0.064s.
    expect(h.starts[0].startS).toBeCloseTo(0.064, 6);
  });

  it('fires onSpeechEnd after redemptionMs of sustained negative', () => {
    const h = captureHandlers();
    const sm = new VadStateMachine(TEST_CFG, h);
    // Open: 3 high frames so minSpeechMs is satisfied.
    feed(sm, [0.9, 0.9, 0.9], 32, 0);
    expect(h.starts.length).toBe(1);
    // 3 negative frames (redemptionMs=96ms exactly) closes the segment.
    feed(sm, [0.1, 0.1, 0.1], 32, 0.096);
    expect(h.ends.length).toBe(1);
    expect(h.misfires.length).toBe(0);
    expect(sm.getState()).toBe('silence');
  });

  it('does not close before redemption is reached', () => {
    const h = captureHandlers();
    const sm = new VadStateMachine(TEST_CFG, h);
    feed(sm, [0.9, 0.9, 0.9], 32, 0);
    // 2 negative frames, then a positive: redemption resets.
    feed(sm, [0.1, 0.1, 0.9], 32, 0.096);
    expect(h.ends.length).toBe(0);
    expect(sm.getState()).toBe('speaking');
  });

  it('treats short bursts as misfires', () => {
    const h = captureHandlers();
    const sm = new VadStateMachine(TEST_CFG, h);
    // Single speech frame (minSpeechMs=64ms=2 frames; one is below).
    feed(sm, [0.9], 32, 0);
    // Now sustained negative until redemption closes the segment.
    feed(sm, [0.1, 0.1, 0.1], 32, 0.032);
    expect(h.misfires.length).toBe(1);
    expect(h.ends.length).toBe(0);
  });

  it('ambiguous frames hold the segment open without advancing redemption', () => {
    const h = captureHandlers();
    const sm = new VadStateMachine(TEST_CFG, h);
    feed(sm, [0.9, 0.9, 0.9], 32, 0);
    // 4 frames in the ambiguous zone (>=0.4, <0.5). Redemption should not
    // accumulate, so the segment stays open. (redemptionLimit=3.)
    feed(sm, [0.45, 0.45, 0.45, 0.45], 32, 0.096);
    expect(h.ends.length).toBe(0);
    expect(sm.getState()).toBe('speaking');
  });

  it('flush closes an open segment', () => {
    const h = captureHandlers();
    const sm = new VadStateMachine(TEST_CFG, h);
    feed(sm, [0.9, 0.9, 0.9], 32, 0);
    sm.flush();
    expect(h.ends.length + h.misfires.length).toBe(1);
    expect(sm.getState()).toBe('silence');
  });

  it('reset clears all state without emitting', () => {
    const h = captureHandlers();
    const sm = new VadStateMachine(TEST_CFG, h);
    feed(sm, [0.9, 0.9, 0.9], 32, 0);
    sm.reset();
    expect(sm.getState()).toBe('silence');
    // No end was fired by reset.
    expect(h.ends.length).toBe(0);
    expect(h.misfires.length).toBe(0);
    // The pre-speech ring is empty.
    expect(sm.getPreSpeechBuffered()).toBe(0);
  });

  it('caps pre-speech padding buffer at preSpeechPadMs/frameMs', () => {
    const h = captureHandlers();
    const sm = new VadStateMachine(TEST_CFG, h);
    // 20 frames of silence: ring must stay at preSpeechCapacity=2.
    feed(sm, Array(20).fill(0.1), 32, 0);
    expect(sm.getPreSpeechBuffered()).toBe(2);
  });
});
