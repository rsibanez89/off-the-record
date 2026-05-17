// Minimal local ambient declaration for `onnxruntime-web`. The package ships
// types at `node_modules/onnxruntime-web/types.d.ts` but its package.json
// `exports` map does not surface them under bundler resolution, so TS cannot
// see them. We only consume `InferenceSession` and `Tensor` (and only their
// surface used by SileroVad), so declare those minimally here. The full
// runtime is loaded at runtime by `import('onnxruntime-web')`.

declare module 'onnxruntime-web' {
  export class Tensor {
    constructor(
      type: 'float32' | 'int64' | 'float16' | 'int32' | 'uint8' | string,
      data: Float32Array | BigInt64Array | Int32Array | Uint8Array,
      dims: number[],
    );
    readonly data: Float32Array | BigInt64Array | Int32Array | Uint8Array;
    readonly dims: number[];
    readonly type: string;
  }

  export interface InferenceSessionOptions {
    executionProviders?: string[];
    logSeverityLevel?: number;
    // Additional ORT session options are accepted but not declared here.
    [k: string]: unknown;
  }

  export class InferenceSession {
    static create(
      modelBuffer: ArrayBuffer | Uint8Array,
      options?: InferenceSessionOptions,
    ): Promise<InferenceSession>;
    run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>;
    release?(): Promise<void>;
  }

  // Subset of the global ORT environment object. We only set `wasmPaths`
  // here (see `sileroVad.ts`); the real shape carries many more fields,
  // but TS does not need to see them.
  export const env: {
    wasm: {
      wasmPaths?: string;
      numThreads?: number;
      simd?: boolean;
      proxy?: boolean;
      [k: string]: unknown;
    };
    [k: string]: unknown;
  };
}
