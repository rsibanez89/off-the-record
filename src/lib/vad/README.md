# `src/lib/vad/`, In-house Silero Voice Activity Detection

> Browser native, offline VAD. Wraps the **Silero VAD v5** ONNX model (MIT, `snakers4/silero-vad`) via `onnxruntime-web`.

## Why

Whisper hallucinates on silence ("Thanks for watching", "Please subscribe", `[BLANK_AUDIO]`, etc.). The previous gate was a fixed-threshold RMS check in `heuristics.ts::isSilent`. RMS treats soft speech as silence and loud noise as speech.

The new gate is **Silero VAD v5**: a 2.2 MB ONNX model that emits a per-frame speech probability. It is trained on 100+ languages, runs in sub-millisecond on WASM SIMD, and is the de-facto open VAD for 2026 browser pipelines.

What we get:

- Fewer Whisper hallucinations.
- Lower compute, since silent windows skip the Whisper pass entirely.
- More aggressive anchor advancement during long silences.
- A foundation for future endpointing and segmentation work.

## Why in-house

For a security-conscious offline app:

- We already pay for the heavy parts. `onnxruntime-web` ships as a transitive dep of `@huggingface/transformers`, and an `AudioWorklet` capture pipeline already exists. A wrapper would save us roughly 200 lines.
- We pin the model hash at build time, serve every byte same-origin, own the audio hot path end to end.
- Smaller surface area in the supply chain, one less single-maintainer dependency in the recording path.

License posture: Silero v5 model weights are MIT. `onnxruntime-web` is MIT. Nothing in this folder carries a copyleft obligation.

## Layout

```
src/lib/vad/
  types.ts                # VadEngine, FrameProbability, SpeechSegment, ModelFetcher
  index.ts                # public barrel + createVadEngine() factory
  framer.ts               # Float32Array stream -> fixed 512-sample frames
  resampler.ts            # 48 kHz -> 16 kHz linear resampler (extracted from producer)
  stateMachine.ts         # frame probabilities -> onSpeechStart / onSpeechEnd / onMisfire
  noopVad.ts              # NoopVadEngine, returns 0 always. For tests.
  silero/
    constants.ts          # SR=16000, FRAME=512, CONTEXT=64, STATE_SHAPE, MODEL_SHA256
    sileroModel.ts        # HttpModelFetcher + sha256 verification
    sileroVad.ts          # VadEngine implementation, one ONNX inference per frame
  __tests__/              # vitest pure-module tests
  *.test.ts               # collocated tests (framer, resampler, stateMachine, noopVad, sileroVad)
```

Companion files outside this folder:

- `scripts/fetch-models.mjs`: build-time provisioner for the Silero ONNX weights. See "Build-time assets" below.
- `vite.config.ts::ortRuntime`: the Vite plugin that serves the `onnxruntime-web` runtime under `/ort/`. See "Runtime serving" below.
- `src/types/onnxruntime-web.d.ts`: minimal ambient type override that surfaces `InferenceSession`, `Tensor`, and `env.wasm.wasmPaths` (the upstream package's `exports` map hides its `.d.ts` from bundler resolution).
- `src/lib/config.ts`: defines `VAD_SILENCE_THRESHOLD` (default 0.3).
- `src/lib/db.ts`: Dexie bumped v3 to v4 to add `speechProbability?: number` to `AudioChunk`.
- `src/workers/producer.worker.ts`: runs the engine on every 512-sample frame, writes `max(speechProbability)` per 1 second chunk.
- `src/workers/consumer.worker.ts`: prefers the VAD verdict over RMS in the silence gate.

## Build-time assets

`scripts/fetch-models.mjs` (wired to `predev` and `prebuild` in `package.json`) downloads **Silero v5 ONNX weights** from `huggingface.co/onnx-community/silero-vad` into `public/models/silero_vad_v5.onnx`. SHA-256 is pinned in the script; mismatch is a hard failure that deletes the bad file and exits non-zero. The file is gitignored (reproducible from the pinned hash).

## Runtime serving

ORT Web loads its `.wasm` binaries at runtime via `fetch`, and the threaded variant `import()`s its `.mjs` loader. Without explicit configuration, the loader resolves to URLs Vite has never heard of, the dev server's SPA fallback serves `index.html`, ORT tries to instantiate it as WebAssembly, and you hit `expected magic word 00 61 73 6d, found 3c 21 64 6f` (`<!do`, the start of `<!doctype html>`).

The fix has two parts.

**`vite.config.ts::ortRuntime`** serves `/ort/*` directly from `node_modules/onnxruntime-web/dist/`. In dev, a middleware streams the file with the right `Content-Type`. In build, `closeBundle` copies the `.wasm` and `.mjs` files into `dist/ort/` so the production bundle is self-contained. Same-origin, no CDN, deterministic across `npm run dev` and `npm run build`.

Why not `public/ort/`? Files in `public/` cannot be dynamically `import()`-ed by Vite (it intercepts module requests for transformation and refuses to serve `public/` files as modules). ORT's threaded variant uses `import()` to load its `.mjs` loader, so vendoring to `public/` only works for the binaries, not the loaders. Sourcing from `node_modules` via middleware works for both.

**`silero/sileroVad.ts::defaultOrtFactories()`** points ORT at that path:

```ts
const ort = await import('onnxruntime-web');
ort.env.wasm.wasmPaths = '/ort/';
```

Setting `wasmPaths` is global to ORT in this realm; because `@huggingface/transformers` resolves to the same `onnxruntime-web` install, this also unifies the WASM load path for Whisper inference.

## Public API

Import from the barrel, not from concrete files. The producer and consumer workers must depend on `VadEngine`, never on `SileroVad` directly. This is the Dependency Inversion seam.

```ts
import {
  // interfaces
  type VadEngine,
  type FrameProbability,
  type SpeechSegment,
  type VadStateMachineConfig,
  type ModelFetcher,

  // defaults
  DEFAULT_STATE_MACHINE_CONFIG,

  // building blocks (pure, no I/O)
  Framer,
  LinearResampler,
  VadStateMachine,

  // engine implementations
  SileroVad,
  NoopVadEngine,

  // factory (the one place that picks the concrete engine)
  createVadEngine,

  // model fetch + integrity
  HttpModelFetcher,
  sha256Hex,
  verifyHash,

  // Silero runtime contract
  SILERO_SR,
  SILERO_FRAME,
  SILERO_CONTEXT,
  SILERO_STATE_SHAPE,
  SILERO_DEFAULT_MODEL_URL,
  SILERO_MODEL_SHA256,
} from '../lib/vad';
```

### `VadEngine`

Narrow on purpose. Four methods only.

```ts
interface VadEngine {
  initialize(): Promise<void>;          // idempotent
  process(frame: Float32Array): Promise<number>;  // 512 samples at 16 kHz -> P(speech)
  reset(): void;                        // between recordings
  dispose(): Promise<void>;             // release session
}
```

### `VadStateMachine`

Pure hysteresis. Consumes per-frame `FrameProbability`, fires callbacks. Defaults: positive threshold 0.3, negative 0.25, redemption 1400 ms, pre-speech padding 800 ms, minimum speech 400 ms, frame 32 ms.

```ts
const sm = new VadStateMachine(
  { positiveSpeechThreshold: 0.3, redemptionMs: 1400 },
  {
    onSpeechStart: (seg) => { /* segment opened */ },
    onSpeechEnd:   (seg) => { /* segment closed cleanly */ },
    onMisfire:     (seg) => { /* segment shorter than minSpeechMs */ },
  },
);
sm.ingest({ probability: 0.42, startS: 1.024, endS: 1.056 });
```

The state machine is **wired and tested** but not yet subscribed to in the consumer worker. The current consumer uses the per-chunk aggregate probability only. Subscribing for aggressive trim during long silences is a planned follow-up (see "Open follow-ups" below).

### `createVadEngine()`

The single place where the concrete engine is chosen. Producer worker imports this, never `new SileroVad()`.

```ts
// src/workers/producer.worker.ts
import { createVadEngine } from '../lib/vad';

const vad = createVadEngine();
await vad.initialize();
const prob = await vad.process(frame);  // returns 0..1
```

Swapping to TEN VAD later is a **one-line factory change** and a new class. Nothing in the worker needs to move.

## How it plugs into the pipeline

```mermaid
flowchart LR
    mic(["mic"]) --> worklet["AudioWorklet, 48 kHz frames"]
    worklet --> producer["Producer worker"]
    producer --> resamp["LinearResampler, 48k -> 16k"]
    resamp --> framer["Framer, 512-sample frames"]
    framer --> silero["SileroVad.process, P(speech)"]
    silero --> maxprob["max() per 1 s chunk"]
    maxprob --> idb[("IDB chunks + audioArchive<br/>speechProbability"]]
    idb --> consumer["Consumer worker"]
    consumer --> gate{"vadVerdict < 0.3 ?"}
    gate -- yes --> skipwhisper["skip Whisper, force-commit, advance anchor"]
    gate -- no --> whisper["run Whisper + LocalAgreement-2"]
```

Key points:

- The producer is the only worker that runs VAD inference. The consumer reads the verdict from IDB.
- The VAD load happens lazily on first record. Until it loads (a few hundred ms), chunks get `speechProbability: undefined` and the consumer falls back to the original RMS heuristic. The recording does not block on VAD load.
- The model is fetched at build time by `scripts/fetch-models.mjs` and served from `public/models/silero_vad_v5.onnx`. No CDN hit at runtime.

## SOLID layout, by file

Each file has one reason to change. Reviewers should be able to point at the file responsible for any one concern.

| Principle | Where it lives |
|---|---|
| **SRP**: ONNX inference for one frame | `silero/sileroVad.ts` |
| **SRP**: framing | `framer.ts` |
| **SRP**: resampling | `resampler.ts` |
| **SRP**: hysteresis | `stateMachine.ts` |
| **SRP**: model fetch + integrity | `silero/sileroModel.ts` |
| **OCP / DIP**: producer depends on `VadEngine` + `createVadEngine()`, not on `SileroVad` | `types.ts`, `index.ts` |
| **LSP**: `NoopVadEngine` is a true drop-in for any `VadEngine` consumer; used in tests | `noopVad.ts` |
| **ISP**: batch / non-realtime / event subscription are NOT on `VadEngine` | `types.ts` |
| **DIP**: `SileroVad` depends on injectable `ModelFetcher`, `sessionFactory`, `tensorFactory` interfaces | `silero/sileroVad.ts` |

The injection seams in `SileroVad` (model fetcher, ORT session factory, tensor factory) exist for tests: the unit tests do not load `onnxruntime-web` or read the 2.2 MB model file. They pass fakes.

## Model integrity

- **Source URL**: `https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx`.
- **License**: MIT, upstream is `snakers4/silero-vad`.
- **Pinned SHA-256**: `a4a068cd6cf1ea8355b84327595838ca748ec29a25bc91fc82e6c299ccdc5808`.
- **Build script** `scripts/fetch-models.mjs`:
  - Skips when the local file already matches the pinned hash.
  - Refuses to write a file with the wrong hash, removes the bad download, exits non-zero.
- The hash is exported from `silero/constants.ts` as `MODEL_SHA256` and rechecked at runtime if available. The file is served same-origin, so we treat it as trusted post-build.

## Configuration

`src/lib/config.ts`:

```ts
export const VAD_SILENCE_THRESHOLD = 0.3;
```

The consumer worker treats `chunk.speechProbability < VAD_SILENCE_THRESHOLD` as silence. Aligned with Silero v5's default positive threshold so the same number gates both "frame is speech" and "chunk has speech in it".

`stateMachine.ts` exports `DEFAULT_STATE_MACHINE_CONFIG` with conventional Silero v5 hysteresis defaults: positive threshold 0.3, negative 0.25, redemption 1400 ms, pre-speech pad 800 ms, minimum speech 400 ms. Override per recording environment via the `VadStateMachine` constructor.

## Tests

```
npm test
```

5 test files, 31 passing tests:

- `framer.test.ts` (8): single full frame, exact-N frames, multi-frame + remainder, split-input continuity, frame larger than input, reset, non-positive frame size rejected.
- `resampler.test.ts` (5): identity ratio, non-positive rate rejected, RMS energy preserved (48 kHz to 16 kHz sine within delta), seam continuity across calls, reset.
- `stateMachine.test.ts` (9): positive crossing fires `onSpeechStart`, pre-speech padding prepended, `onSpeechEnd` after redemption, no close before redemption, short bursts as misfires, ambiguous frames hold steady, flush, reset, pre-speech ring capped.
- `noopVad.test.ts` (2): substantive `VadEngine` contract over adversarial signals (silence, full-scale, tone, noise, NaN), dispose / reset safe.
- `silero/sileroVad.test.ts` (7): with mocked ORT session: input shape and `sr` correct, state threaded, context carried across calls, reset zeros state and context, process before initialize rejected, wrong frame size rejected, dispose makes process fail again.

The tests do not touch `onnxruntime-web` or the real ONNX file. They run in a Node environment in milliseconds.

## Engine swap example

To replace Silero with a hypothetical TEN VAD:

1. Add `silero/`-style folder: `ten/` with `tenVad.ts` implementing `VadEngine`.
2. Change one line in `index.ts`:

   ```ts
   export function createVadEngine(): VadEngine {
   -  return new SileroVad();
   +  return new TenVad();
   }
   ```

3. Update `scripts/fetch-models.mjs` to fetch the TEN VAD model with its own pinned hash.

Nothing in `producer.worker.ts`, `consumer.worker.ts`, or `db.ts` changes. That is OCP earning its keep.

## Open follow-ups

These were intentionally deferred when the in-house VAD landed. None block shipping.

- **Subscribe the consumer to `onSpeechStart` / `onSpeechEnd`** so the anchor can trim aggressively during long silences. The state machine is wired and tested; only the subscriber is missing.
- **Surface `VAD_SILENCE_THRESHOLD`** in the dev panel for live tuning.
- **WebGPU EP for Silero**. Currently uses WASM only. Silero is sub-ms per frame on WASM SIMD; WebGPU setup cost would not pay off until we batch frames.
- **Sub-second remainder on Stop**. The producer still drops the trailing partial chunk on stop. Pad with zeros and emit one final chunk so we do not lose up to 1 s of audio at end of recording. Tracked in the workspace improvements backlog as 3.6.

## References

- Workspace improvements backlog: section 5.1 (proposal), section 4.x (tests), section 3.x (memory).
- `docs/state-of-the-art.md` section 6: the broader VAD landscape (Silero, TEN, WebRTC, MarbleNet, pyannote).
- Silero VAD upstream: `https://github.com/snakers4/silero-vad`.
- ONNX Runtime Web docs (`env.wasm.wasmPaths`): `https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html`.
