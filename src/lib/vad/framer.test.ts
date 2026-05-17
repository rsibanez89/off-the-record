import { describe, expect, it } from 'vitest';
import { Framer } from './framer';

function ramp(n: number, start = 0): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = start + i;
  return out;
}

describe('Framer', () => {
  it('emits no frames for zero-length input', () => {
    const f = new Framer(512);
    expect(f.push(new Float32Array(0))).toEqual([]);
    expect(f.buffered()).toBe(0);
  });

  it('buffers until one full frame is available', () => {
    const f = new Framer(512);
    expect(f.push(ramp(100)).length).toBe(0);
    expect(f.buffered()).toBe(100);
    const out = f.push(ramp(412, 100));
    expect(out.length).toBe(1);
    expect(out[0].length).toBe(512);
    expect(f.buffered()).toBe(0);
    // First sample should be 0 (the start of the first push).
    expect(out[0][0]).toBe(0);
    // 100th sample should be 100 (the start of the second push).
    expect(out[0][100]).toBe(100);
  });

  it('emits exactly N full frames when fed N frames worth of input', () => {
    for (const n of [1, 2, 3]) {
      const f = new Framer(512);
      const frames = f.push(ramp(512 * n));
      expect(frames.length).toBe(n);
      expect(f.buffered()).toBe(0);
      for (let i = 0; i < n; i++) {
        expect(frames[i].length).toBe(512);
        expect(frames[i][0]).toBe(512 * i);
        expect(frames[i][511]).toBe(512 * i + 511);
      }
    }
  });

  it('emits multiple frames and buffers the remainder', () => {
    const f = new Framer(512);
    const out = f.push(ramp(1500));
    expect(out.length).toBe(2); // 1024 used, 476 buffered
    expect(f.buffered()).toBe(1500 - 1024);
    expect(out[1][0]).toBe(512);
  });

  it('preserves continuity across split inputs', () => {
    const f = new Framer(512);
    f.push(ramp(300));
    const out1 = f.push(ramp(212, 300)); // 300+212 = 512 exactly
    expect(out1.length).toBe(1);
    expect(out1[0][0]).toBe(0);
    expect(out1[0][511]).toBe(511);
    // Now feed exactly 512 more samples (continuing the ramp), expect another full frame.
    const out2 = f.push(ramp(512, 512));
    expect(out2.length).toBe(1);
    expect(out2[0][0]).toBe(512);
    expect(out2[0][511]).toBe(1023);
  });

  it('handles frameSize larger than the input', () => {
    const f = new Framer(1024);
    expect(f.push(ramp(100)).length).toBe(0);
    expect(f.push(ramp(100)).length).toBe(0);
    expect(f.buffered()).toBe(200);
    // Push enough to complete one frame.
    const out = f.push(ramp(824, 200));
    expect(out.length).toBe(1);
    expect(out[0].length).toBe(1024);
    expect(f.buffered()).toBe(0);
  });

  it('reset clears the buffer', () => {
    const f = new Framer(512);
    f.push(ramp(300));
    expect(f.buffered()).toBe(300);
    f.reset();
    expect(f.buffered()).toBe(0);
    const out = f.push(ramp(512));
    expect(out.length).toBe(1);
    expect(out[0][0]).toBe(0); // starts from the new ramp, not the old.
  });

  it('rejects non-positive frame sizes', () => {
    expect(() => new Framer(0)).toThrow();
    expect(() => new Framer(-1)).toThrow();
    expect(() => new Framer(1.5)).toThrow();
  });
});
