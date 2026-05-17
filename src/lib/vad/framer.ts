// Splits a continuous Float32Array stream into fixed-size frames. Pure, sync,
// no I/O. Used by the producer worker to feed Silero VAD with exactly the
// 512-sample windows it requires at 16 kHz.
//
// SRP: framing only. Resampling lives in `resampler.ts`, inference in
// `silero/sileroVad.ts`, hysteresis in `stateMachine.ts`.

export class Framer {
  private readonly frameSize: number;
  private buffer: Float32Array;
  /** Number of samples currently held in `buffer`. */
  private length = 0;

  constructor(frameSize: number) {
    if (!Number.isInteger(frameSize) || frameSize <= 0) {
      throw new Error(`Framer: frameSize must be a positive integer, got ${frameSize}`);
    }
    this.frameSize = frameSize;
    // Pre-allocate two frames so we can append without resizing in the common
    // case. If `push` ever feeds more than one frame's worth at once, the
    // buffer grows below.
    this.buffer = new Float32Array(frameSize * 2);
  }

  /**
   * Append samples; return zero or more full frames. Each returned frame is a
   * fresh `Float32Array` of length `frameSize`. Remaining samples stay
   * buffered for the next call.
   */
  push(samples: Float32Array): Float32Array[] {
    if (samples.length === 0) return [];
    this.ensureCapacity(this.length + samples.length);
    this.buffer.set(samples, this.length);
    this.length += samples.length;

    if (this.length < this.frameSize) return [];

    const out: Float32Array[] = [];
    let offset = 0;
    while (this.length - offset >= this.frameSize) {
      const frame = new Float32Array(this.frameSize);
      frame.set(this.buffer.subarray(offset, offset + this.frameSize));
      out.push(frame);
      offset += this.frameSize;
    }
    // Shift the remainder to the front of the buffer.
    const remainder = this.length - offset;
    if (remainder > 0) {
      this.buffer.copyWithin(0, offset, this.length);
    }
    this.length = remainder;
    return out;
  }

  /** Discard any buffered samples; do not emit them. */
  reset(): void {
    this.length = 0;
  }

  /** How many samples are currently buffered (waiting for the next frame). */
  buffered(): number {
    return this.length;
  }

  private ensureCapacity(needed: number): void {
    if (this.buffer.length >= needed) return;
    let next = this.buffer.length;
    while (next < needed) next *= 2;
    const grown = new Float32Array(next);
    grown.set(this.buffer.subarray(0, this.length));
    this.buffer = grown;
  }
}
