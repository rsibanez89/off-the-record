// SileroVad: VadEngine implementation backed by Silero v5 + onnxruntime-web.
// One ONNX inference per 512-sample frame at 16 kHz (32 ms). State is threaded
// across calls; context (64 samples) is prepended to each frame from the tail
// of the previous one.
//
// SRP: this class does ONE thing, ONNX inference for a single frame. It does
// not own framing, resampling, hysteresis, or model fetching.

import type { ModelFetcher, VadEngine } from '../types';
import { HttpModelFetcher, verifyHash } from './sileroModel';
import {
  CONTEXT,
  DEFAULT_MODEL_URL,
  FRAME,
  MODEL_SHA256,
  SR,
  STATE_SHAPE,
  STATE_SIZE,
} from './constants';

/**
 * Minimal shape we need from an ORT InferenceSession. Declared locally so
 * tests can fake it without importing onnxruntime-web. Matches the public
 * `ort.InferenceSession` runtime contract.
 */
export interface SileroOrtTensor {
  data: Float32Array | BigInt64Array;
}

export interface SileroOrtSession {
  run(feeds: Record<string, unknown>): Promise<Record<string, SileroOrtTensor>>;
}

export type SileroOrtSessionFactory = (modelBuffer: ArrayBuffer) => Promise<SileroOrtSession>;
export type SileroTensorFactory = (
  type: 'float32' | 'int64',
  data: Float32Array | BigInt64Array,
  dims: readonly number[],
) => unknown;

export interface SileroVadOptions {
  /** Where the ONNX file lives. Defaults to `/models/silero_vad_v5.onnx`. */
  modelUrl?: string;
  /** Indirection for fetching the model. Default: HttpModelFetcher. */
  fetcher?: ModelFetcher;
  /** Inject an ORT session factory (tests bypass real ORT). */
  sessionFactory?: SileroOrtSessionFactory;
  /** Inject a tensor factory (tests bypass real ORT). */
  tensorFactory?: SileroTensorFactory;
  /**
   * If true and the buffer is not verifiable (no ModelFetcher with a hash
   * step), still load. Default true: same-origin assets are trusted.
   */
  skipRuntimeHashCheck?: boolean;
  /** Expected SHA-256 (lowercase hex). Defaults to the pinned hash. */
  expectedSha256?: string;
}

/**
 * Default ORT session/tensor factories using onnxruntime-web. Pulled in only
 * on first real `initialize()` so tests do not load ORT at import time.
 */
async function defaultOrtFactories(): Promise<{
  sessionFactory: SileroOrtSessionFactory;
  tensorFactory: SileroTensorFactory;
}> {
  const ort = await import('onnxruntime-web');
  // ORT-Web loads its `.wasm` binaries at runtime via fetch. Without an
  // explicit `wasmPaths`, the loader resolves to a relative URL that Vite's
  // dev server happily 404s back to `index.html`, which then trips the
  // famous "expected magic word 00 61 73 6d, found 3c 21 64 6f" error
  // (3c 21 64 6f = `<!do`). We vendor the wasm artefacts into `public/ort/`
  // via `scripts/fetch-models.mjs` and pin the loader to that path here.
  // Same-origin, no CDN, idempotent across `npm run dev` and `npm run build`.
  try {
    ort.env.wasm.wasmPaths = '/ort/';
  } catch {
    // Older ORT shapes may freeze env.wasm; this is best-effort and
    // harmless if the field is already set or not writable.
  }
  const sessionFactory: SileroOrtSessionFactory = async (modelBuffer) => {
    // Match the existing Vite + COOP/COEP setup: WASM is the universal
    // fallback; WebGPU would also work but the per-frame cost is so small
    // for Silero (~0.05 ms on WASM SIMD) that we keep things simple.
    const session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['wasm'],
      logSeverityLevel: 3,
    });
    return session as unknown as SileroOrtSession;
  };
  const tensorFactory: SileroTensorFactory = (type, data, dims) =>
    new ort.Tensor(type, data as Float32Array | BigInt64Array, dims as number[]);
  return { sessionFactory, tensorFactory };
}

export class SileroVad implements VadEngine {
  private session: SileroOrtSession | null = null;
  private state: Float32Array = new Float32Array(STATE_SIZE);
  private ctx: Float32Array = new Float32Array(CONTEXT);
  private readonly modelUrl: string;
  private readonly fetcher: ModelFetcher;
  private readonly sessionFactoryOverride?: SileroOrtSessionFactory;
  private readonly tensorFactoryOverride?: SileroTensorFactory;
  private tensorFactory: SileroTensorFactory | null = null;
  private srTensorCached: unknown = null;
  private readonly expectedSha256?: string;

  constructor(opts: SileroVadOptions = {}) {
    this.modelUrl = opts.modelUrl ?? DEFAULT_MODEL_URL;
    this.fetcher = opts.fetcher ?? new HttpModelFetcher();
    this.sessionFactoryOverride = opts.sessionFactory;
    this.tensorFactoryOverride = opts.tensorFactory;
    this.expectedSha256 = opts.expectedSha256 ?? MODEL_SHA256;
    // skipRuntimeHashCheck reserved for future use; we trust same-origin assets.
    void opts.skipRuntimeHashCheck;
  }

  async initialize(): Promise<void> {
    if (this.session) return;
    let sessionFactory: SileroOrtSessionFactory;
    if (this.sessionFactoryOverride && this.tensorFactoryOverride) {
      sessionFactory = this.sessionFactoryOverride;
      this.tensorFactory = this.tensorFactoryOverride;
    } else {
      const fac = await defaultOrtFactories();
      sessionFactory = this.sessionFactoryOverride ?? fac.sessionFactory;
      this.tensorFactory = this.tensorFactoryOverride ?? fac.tensorFactory;
    }
    const buf = await this.fetcher.fetch(this.modelUrl);
    // Hash verification is best-effort: in browser contexts the asset is
    // same-origin and pinned by the build script; we recheck if expected is
    // set explicitly. Tests use a fake fetcher with no hash, so we skip
    // when injection is present.
    if (this.expectedSha256 && !this.sessionFactoryOverride) {
      try {
        await verifyHash(buf, this.expectedSha256);
      } catch (err) {
        // Re-throw with context. This is loud on purpose: a hash mismatch
        // means someone served a tampered model.
        throw new Error(`SileroVad model integrity check failed: ${(err as Error).message}`);
      }
    }
    this.session = await sessionFactory(buf);
    this.srTensorCached = this.tensorFactory('int64', BigInt64Array.from([BigInt(SR)]), [1]);
  }

  async process(frame: Float32Array): Promise<number> {
    if (!this.session || !this.tensorFactory) {
      throw new Error('SileroVad.process called before initialize');
    }
    if (frame.length !== FRAME) {
      throw new Error(`SileroVad.process: expected ${FRAME} samples, got ${frame.length}`);
    }
    const input = new Float32Array(CONTEXT + FRAME);
    input.set(this.ctx, 0);
    input.set(frame, CONTEXT);

    const feeds = {
      input: this.tensorFactory('float32', input, [1, CONTEXT + FRAME]),
      state: this.tensorFactory('float32', this.state, STATE_SHAPE),
      sr: this.srTensorCached,
    };

    const out = await this.session.run(feeds);
    // ORT returns new tensors. Copy state out (the underlying buffer might
    // be reused by ORT in some backends; copy defensively).
    const stateN = out.stateN ?? out.state;
    if (stateN && stateN.data instanceof Float32Array) {
      this.state = new Float32Array(stateN.data);
    }
    const output = out.output;
    const prob = output && output.data instanceof Float32Array ? output.data[0] : 0;

    // Keep the last CONTEXT samples of this frame as context for the next.
    this.ctx = new Float32Array(frame.subarray(FRAME - CONTEXT));

    return prob;
  }

  reset(): void {
    this.state = new Float32Array(STATE_SIZE);
    this.ctx = new Float32Array(CONTEXT);
  }

  async dispose(): Promise<void> {
    // onnxruntime-web sessions are GC'd when references are dropped. No
    // explicit dispose call is needed for the wasm EP; if WebGPU is added,
    // call session.release() here.
    this.session = null;
    this.srTensorCached = null;
    this.tensorFactory = null;
    this.state = new Float32Array(STATE_SIZE);
    this.ctx = new Float32Array(CONTEXT);
  }
}
