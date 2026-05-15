// Captures mono Float32 samples and posts them in fixed-size frames.
// Resampling to 16 kHz happens on the main side (OfflineAudioContext would block
// the worklet). We post raw context-rate samples; producer worker resamples.
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = [];
    this.bufferLength = 0;
    this.targetLength = 2048;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel = input[0];
    if (!channel || channel.length === 0) return true;

    // Copy because the underlying buffer is reused.
    const copy = new Float32Array(channel.length);
    copy.set(channel);
    this.buffer.push(copy);
    this.bufferLength += copy.length;

    while (this.bufferLength >= this.targetLength) {
      const out = new Float32Array(this.targetLength);
      let offset = 0;
      while (offset < this.targetLength) {
        const head = this.buffer[0];
        const need = this.targetLength - offset;
        if (head.length <= need) {
          out.set(head, offset);
          offset += head.length;
          this.buffer.shift();
        } else {
          out.set(head.subarray(0, need), offset);
          this.buffer[0] = head.subarray(need);
          offset += need;
        }
      }
      this.bufferLength -= this.targetLength;
      this.port.postMessage({ samples: out, sampleRate }, [out.buffer]);
    }

    return true;
  }
}

registerProcessor('capture-processor', CaptureProcessor);
