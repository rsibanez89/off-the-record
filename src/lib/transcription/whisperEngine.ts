// WhisperEngine owns the Whisper pipeline lifecycle: backend detection,
// instantiation, run, dispose. Both the live consumer and the batch worker
// consume it through the same interface so pipeline setup, error recovery,
// and (future) disposal cycles live in one place.
//
// SOLID:
//   Single Responsibility: pipeline lifecycle only. Algorithm logic stays in
//     `LiveTranscriptionLoop`; word-timestamp parsing stays in
//     `whisperAdapter.runWhisper`.
//   Open/Closed: future variants (e.g. a disposing-every-N-ticks engine for
//     WebGPU memory pressure) can subclass or wrap without touching workers.
//   Dependency Inversion: workers depend on this class, not on
//     transformers.js or ORT directly.

import { isMoonshine, type ModelId } from '../audio';
import {
  createPipeline,
  detectBackend,
  runWhisper,
  type Backend,
  type ProgressEntry,
  type WhisperPipeline,
  type WhisperRunOptions,
  type WhisperRunResult,
} from './whisperAdapter';

export class WhisperEngine {
  private pipelinePromise: Promise<WhisperPipeline> | null = null;
  private backend: Backend = 'wasm';
  private readonly modelId: ModelId;

  constructor(modelId: ModelId) {
    this.modelId = modelId;
  }

  /**
   * Detect the runtime backend, instantiate the pipeline, and hold it ready
   * for `run` and `getPipeline`. Idempotent: a second call replaces the
   * pipeline (used by the consumer worker on model switch).
   *
   * Moonshine override: even when WebGPU is available, we force the WASM
   * backend for Moonshine specifically. Empirical evidence (a user
   * recording of clean 9 s English mic audio) showed Moonshine on WebGPU
   * + q8 producing multilingual gibberish, while the same model on Node
   * CPU correctly transcribed every fixture in the matrix (synth 5 s,
   * jfk 11 s, apollo11 25 s). The strongest hypothesis is a WebGPU + q8
   * precision issue in transformers.js / ORT-web. WASM gives correct
   * output at the cost of speed; until upstream fixes WebGPU q8 for
   * Moonshine, the override is the safe default.
   */
  async load(onProgress: (p: ProgressEntry) => void): Promise<void> {
    const detected = await detectBackend();
    this.backend = isMoonshine(this.modelId) ? 'wasm' : detected;
    this.pipelinePromise = createPipeline(this.modelId, this.backend, onProgress);
    await this.pipelinePromise;
  }

  /** Run one Whisper inference. Throws if `load` has not been awaited. */
  async run(samples: Float32Array, opts: WhisperRunOptions): Promise<WhisperRunResult> {
    const pipeline = await this.getPipeline();
    return runWhisper(pipeline, samples, opts);
  }

  /**
   * Return the underlying pipeline (for callers that need to construct a
   * `LiveTranscriptionLoop` against it). Throws if not loaded.
   */
  async getPipeline(): Promise<WhisperPipeline> {
    if (!this.pipelinePromise) {
      throw new Error('WhisperEngine not loaded yet; call load() first');
    }
    return this.pipelinePromise;
  }

  getBackend(): Backend {
    return this.backend;
  }

  getModelId(): ModelId {
    return this.modelId;
  }

  /**
   * Drop the pipeline reference. Future variants can call the pipeline's
   * dispose hook here when WebGPU memory pressure requires periodic
   * reload (transformers.js issue #860). For now this is a hook for the
   * worker to release memory at session boundaries.
   */
  async dispose(): Promise<void> {
    this.pipelinePromise = null;
  }
}
