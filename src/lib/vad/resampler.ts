// Linear-interpolation resampler from a source sample rate to a target sample
// rate. Pure, sync. Treats successive `process` calls as one continuous
// stream by carrying `resamplePos` and `prevTail` across calls. Extracted
// from `producer.worker.ts` as part of the VAD work.
//
// Notes on accuracy: linear interpolation is the cheapest viable resampler.
// At 48 kHz to 16 kHz it introduces minor aliasing but the difference is
// inaudible at speech bandwidths and Whisper's mel feature extractor smooths
// over it. If higher quality is ever needed, swap in a Kaiser-windowed sinc
// filter behind this same interface.

export class LinearResampler {
  private readonly ratio: number;
  /**
   * Next emit position in the current frame's local coordinates. May be
   * negative, meaning "between the previous frame's last sample and this
   * frame's first sample".
   */
  private pos = 0;
  /** Last sample of the previous frame, used to interpolate across the seam. */
  private prevTail = 0;

  /**
   * @param sourceRate samples per second of the incoming stream
   * @param targetRate desired samples per second of the outgoing stream
   */
  constructor(sourceRate: number, targetRate: number) {
    if (sourceRate <= 0 || targetRate <= 0) {
      throw new Error(
        `LinearResampler: rates must be positive, got source=${sourceRate} target=${targetRate}`,
      );
    }
    this.ratio = sourceRate / targetRate;
  }

  /**
   * Resample one input frame. The returned `Float32Array` contains zero or
   * more output samples. Seam state is updated for the next call.
   */
  process(samples: Float32Array): Float32Array {
    if (samples.length === 0) return new Float32Array(0);

    if (this.ratio === 1) {
      // Identity passthrough. Copy so the caller cannot mutate our seam input.
      const out = new Float32Array(samples.length);
      out.set(samples);
      return out;
    }

    // Estimate output length to preallocate. We add 1 for safety so the
    // hot loop never has to grow the buffer.
    const estimate = Math.max(1, Math.ceil((samples.length - this.pos) / this.ratio) + 1);
    let out = new Float32Array(estimate);
    let oi = 0;

    while (true) {
      const i0 = Math.floor(this.pos);
      const i1 = i0 + 1;
      if (i1 >= samples.length) break;
      const frac = this.pos - i0;
      const a = i0 < 0 ? this.prevTail : samples[i0];
      const b = samples[i1];
      if (oi >= out.length) {
        // Defensive grow path. The estimate should always cover us; if it
        // does not we still produce correct output, just slower.
        const grown = new Float32Array(out.length * 2);
        grown.set(out);
        out = grown;
      }
      out[oi++] = a * (1 - frac) + b * frac;
      this.pos += this.ratio;
    }

    // Carry into the next frame's coordinate system.
    this.pos -= samples.length;
    this.prevTail = samples[samples.length - 1];

    return oi === out.length ? out : out.subarray(0, oi);
  }

  /** Reset seam state. Call between recording sessions. */
  reset(): void {
    this.pos = 0;
    this.prevTail = 0;
  }
}
