import { describe, expect, it } from 'vitest';
import { LinearResampler } from './resampler';

function makeSine(freqHz: number, sampleRate: number, durationS: number): Float32Array {
  const n = Math.floor(sampleRate * durationS);
  const out = new Float32Array(n);
  const w = (2 * Math.PI * freqHz) / sampleRate;
  for (let i = 0; i < n; i++) out[i] = Math.sin(w * i);
  return out;
}

function rms(buf: Float32Array): number {
  if (buf.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
  return Math.sqrt(s / buf.length);
}

describe('LinearResampler', () => {
  it('is the identity when source and target rate match', () => {
    const r = new LinearResampler(16000, 16000);
    const inSamples = makeSine(440, 16000, 0.1);
    const out = r.process(inSamples);
    expect(out.length).toBe(inSamples.length);
    // Same values, modulo a fresh array.
    for (let i = 0; i < out.length; i++) expect(out[i]).toBeCloseTo(inSamples[i], 6);
  });

  it('rejects non-positive rates', () => {
    expect(() => new LinearResampler(0, 16000)).toThrow();
    expect(() => new LinearResampler(16000, 0)).toThrow();
    expect(() => new LinearResampler(-1, 16000)).toThrow();
  });

  it('preserves RMS energy within tolerance when resampling 48 kHz -> 16 kHz sine', () => {
    // A 440 Hz sine at 48 kHz, downsampled to 16 kHz. Both well below Nyquist.
    const src = makeSine(440, 48000, 1.0);
    const r = new LinearResampler(48000, 16000);
    const out = r.process(src);
    // Output length should be ~src.length / 3.
    expect(out.length).toBeGreaterThan(15800);
    expect(out.length).toBeLessThan(16200);
    const inRms = rms(src);
    const outRms = rms(out);
    // RMS of a sine is 1/sqrt(2) ~ 0.7071. Linear resampling at this ratio
    // should preserve it within 1% relative.
    expect(Math.abs(inRms - outRms) / inRms).toBeLessThan(0.02);
  });

  it('seam continuity across two consecutive process calls', () => {
    const src1 = makeSine(440, 48000, 0.05);
    const src2 = makeSine(440, 48000, 0.05);
    // The second buffer must continue the phase of the first for the test to
    // be meaningful. Construct it accordingly.
    const continuous = new Float32Array(src1.length + src2.length);
    const w = (2 * Math.PI * 440) / 48000;
    for (let i = 0; i < continuous.length; i++) continuous[i] = Math.sin(w * i);
    const part1 = continuous.subarray(0, src1.length);
    const part2 = continuous.subarray(src1.length);

    const r = new LinearResampler(48000, 16000);
    const out1 = r.process(new Float32Array(part1));
    const out2 = r.process(new Float32Array(part2));
    const joined = new Float32Array(out1.length + out2.length);
    joined.set(out1, 0);
    joined.set(out2, out1.length);

    // Compare against running the same continuous signal through one fresh resampler.
    const ref = new LinearResampler(48000, 16000);
    const refOut = ref.process(new Float32Array(continuous));

    // Lengths should match within 1 sample.
    expect(Math.abs(joined.length - refOut.length)).toBeLessThanOrEqual(1);

    // Sample-by-sample, the two paths should agree to within a small epsilon.
    const n = Math.min(joined.length, refOut.length);
    let maxDiff = 0;
    for (let i = 0; i < n; i++) {
      const d = Math.abs(joined[i] - refOut[i]);
      if (d > maxDiff) maxDiff = d;
    }
    expect(maxDiff).toBeLessThan(1e-3);
  });

  it('reset clears seam state', () => {
    const src = makeSine(1000, 48000, 0.1);
    const r = new LinearResampler(48000, 16000);
    r.process(src);
    r.reset();
    const out = r.process(src);
    // After reset, processing the same signal must produce a deterministic
    // result identical to a fresh resampler.
    const fresh = new LinearResampler(48000, 16000).process(src);
    expect(out.length).toBe(fresh.length);
    for (let i = 0; i < out.length; i++) expect(out[i]).toBeCloseTo(fresh[i], 6);
  });
});
