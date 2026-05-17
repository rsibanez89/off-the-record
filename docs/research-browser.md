# Browser-Based / Edge / WebGPU Offline Speech Recognition. Runtime Research

Scope: the **runtime / inference layer** for the browser and edge, not the ASR models themselves. Findings feed an offline browser app (Off The Record) built on transformers.js v3 + WebGPU + LocalAgreement-2.

Snapshot date: **2026-05**.

---

## TL;DR

- The default-and-best modern stack for in-browser Whisper is **transformers.js v3, ONNX Runtime Web, WebGPU**. There is no serious competitor that combines model selection breadth, GPU acceleration, and idiomatic JS.
- **WebGPU is officially Baseline across all major browsers** as of 2026 (Chrome 113+, Edge 113+, Safari 26 on macOS Tahoe / iOS 26 / iPadOS 26 / visionOS 26, Firefox 141 on Windows / 145 on macOS ARM). It is no longer behind flags in evergreen desktop releases.
- For Whisper specifically, **WebGPU is not unconditionally faster** than WASM. On Apple Silicon, recent reports show WASM beating WebGPU for the tiny/base encoder/decoder by 1.5x to 5x depending on quantization. WebGPU wins decisively on discrete NVIDIA GPUs and on large models / long decode runs. **Benchmark on your target hardware.**
- The **whisper-large-v3-turbo** model (4 decoder layers vs 32) is the practical sweet spot for browser deployment: roughly 800 MB on `q4` weights, encoder is the bottleneck, decoder benefits dramatically from quantization without much WER loss.
- **fp16 / q4f16 are still risky on WebGPU** in 2026. Overflow / NaN bugs in ONNX Runtime Web's WebGPU fp16 path bite encoder/decoder models. The proven recipe is `encoder: fp32, decoder: q4` for desktop Chrome, and `encoder: fp32, decoder: q8` if you need WASM fallback.
- **Moonshine-web** (useful-sensors) is a smaller, lower-latency alternative for live captioning (27M / 61M params, around 5x faster than equivalent Whisper on short utterances), shipped via ONNX Runtime Web. Worth comparing for the LocalAgreement-2 streaming use case.
- **Cross-origin isolation (COOP/COEP)** is required to get WASM threads + SharedArrayBuffer. Without it you fall back to a single-threaded WASM build and lose around 3.4x on CPU inference.
- **AudioWorklet is the only sane microphone capture path** in 2026. ScriptProcessorNode is deprecated. MediaRecorder is fine for "record then transcribe" but not real-time. The AudioContext almost always runs at 48 kHz; you must resample to 16 kHz for Whisper.
- **Cache API (or OPFS) for model weights**, not IndexedDB. Per-origin quotas in 2026 are generous: Chrome up to 60% of disk per origin, Firefox the smaller of 10% disk / 10 GiB.

---

## 1. Runtimes

### 1.1 transformers.js v3 (Hugging Face)

The reference JS port of the Hugging Face `transformers` Python library. v3 (Oct 2024) added WebGPU via an ORT Web collaboration. As of v3.7 / v3.8 (mid-2025, latest series in 2026) it ships **120+ architectures**, **1200+ pre-converted ONNX models** on the Hub, and is the de-facto runtime for browser ASR.

- Package: `@huggingface/transformers` (formerly `@xenova/transformers`).
- Backends: ONNX Runtime Web (WASM, WebGPU, WebNN-experimental). No native WebGL backend for ML.
- Whisper-relevant API surface:
  - `pipeline("automatic-speech-recognition", model_id, { device, dtype })`
  - Per-module `dtype` for encoder/decoder models: `{ encoder_model, decoder_model_merged, embed_tokens, ... }`
  - `ModelRegistry.get_available_dtypes(repo)`. Probe which quantizations exist before loading.
  - `return_timestamps: "word"` for word-level timing; long-form chunking via `chunk_length_s`.
- Whisper specifics:
  - Audio in: **16 kHz mono Float32Array**, planar, normalized to [-1, 1].
  - Default `chunk_length_s` is 30 (Whisper's native window).
  - `TextStreamer` lets the worker emit tokens as they are generated.
  - Long-form chunking has known timestamp bugs (issues [#1357](https://github.com/huggingface/transformers.js/issues/1357), [#1358](https://github.com/huggingface/transformers.js/issues/1358)).

**Key known issue. WebGPU memory leak ([#860](https://github.com/huggingface/transformers.js/issues/860))**: tensors are not always disposed after a transcription pipeline run. GPU memory grows monotonically over repeated transcriptions, eventually causing a "device lost" event and tab crash. Mitigation: explicit `model.dispose()` between runs, hold a single Singleton pipeline instance (don't recreate per chunk), and reload the page periodically for long-running sessions. The root architectural fix (separating static encoder + dynamic decoder KV cache, à la Ratchet) is open.

### 1.2 ONNX Runtime Web

The underlying inference runtime that transformers.js calls into. Two distribution flavors:

- `onnxruntime-web` (default). WASM EP, around a few MB of WASM + JS.
- `onnxruntime-web/webgpu`. Adds the WebGPU EP via JSEP (JS Execution Provider). Selected by `executionProviders: ['webgpu']`.

Important environment flags (`ort.env`):

| Flag | Default | What it does |
|---|---|---|
| `env.wasm.numThreads` | auto (min(hw/2, 4)) | Worker thread count. Requires SAB + COOP/COEP. Set to 1 to force single-thread. |
| `env.wasm.simd` | auto-detected | Fixed-width WASM SIMD. v1.19+ dropped non-SIMD builds. |
| `env.wasm.proxy` | false | Run inference in a Worker. **Incompatible with WebGPU EP** and CSP-restricted pages. |
| `env.wasm.wasmPaths` | CDN | Override the WASM binary URL (e.g., to self-host). |
| `env.webgpu.adapter` |  | Pre-supply a GPU adapter (e.g., to pick `high-performance`). |
| `env.webgpu.powerPreference` |  | `"high-performance"` vs `"low-power"`. |
| `env.webgpu.forceFallbackAdapter` | false | Force a fallback (software) WebGPU adapter. |
| `env.webgpu.profiling` |  | WebGPU op-level profiling. |
| Session: `enableGraphCapture` | false | WebGPU graph capture for static-shape models (huge win for fixed-shape encoder). |
| Session: `preferredOutputLocation` | cpu | `"gpu-buffer"` keeps outputs on GPU. essential for chained inference. |
| Session: `externalData` |  | Required for **models > 2 GB**, which exceed the protobuf limit. |
| Session: `freeDimensionOverrides` |  | Pin dynamic dims (e.g., sequence length) for kernel specialization. |

**Operator coverage / fp16**: WebGPU EP now has Flash-Attention, qMoE, Split-K MatMul, GridSample. **The WebGPU fp16 path still has correctness bugs** ([ORT #26732](https://github.com/microsoft/onnxruntime/issues/26732), [#26367](https://github.com/microsoft/onnxruntime/issues/26367)). `fp16` and `q4f16` produce NaNs / overflow on some models including Whisper decoder and Gemma 3. Treat fp16 on WebGPU as "test, don't trust" in 2026.

**Model size > 2 GB**: protobuf's hard 2 GB limit means whisper-large-v3 fp32 (around 3 GB) must use ONNX external data format. ORT Web supports this. pass `externalData: [{ data: blobURL, path: "model.onnx_data" }]` in session options. ORT Web allocates WASM memory > 2 GB via `new WebAssembly.Memory({ shared: true })`. Without external data + WASM64 / cross-origin isolation, you cannot load large-v3 fp32 in a browser.

### 1.3 Other runtimes (context)

| Project | Status (2026) | Browser fit |
|---|---|---|
| **WebLLM / MLC** | Mature for LLMs, not ASR. WebGPU-only. | Not relevant for transcription. |
| **TensorFlow.js** | Still maintained, but the ML ecosystem has moved to ONNX/transformers.js for non-vision tasks. | Avoid for Whisper. |
| **whisper.cpp WASM** | First-class WASM builds, microphone demo at `ggml.ai/whisper.cpp/stream.wasm/`. | Solid fallback, CPU-only, no WebGPU. |
| **sherpa-onnx WASM** | Streaming Zipformer + others, CPU WASM only, no WebGPU. | Useful if you switch to Zipformer/Paraformer; very tiny WERs in Mandarin. |
| **vosk-browser** | Kaldi WASM; last released 2022; effectively unmaintained. | Avoid for new builds. |
| **Picovoice Cheetah / Leopard Web SDK** | Commercial, closed-source, on-device WASM. Cheetah is streaming. Free tier exists. | Strong real-time alternative if licensing terms work. |
| **moonshine-web (useful-sensors)** | ONNX Runtime Web + transformers.js. Tiny (27M) / Base (61M). v2 has sliding-window streaming. | Best alternative ASR for live captioning. |
| **whisper.rn** | React Native binding of whisper.cpp via JSI. | Mobile-only, but a useful comparison point for offline ASR cost. |

---

## 2. Backends

### 2.1 WebGPU

**Browser availability (May 2026):**

| Browser | Status | Notes |
|---|---|---|
| Chrome 113+ (desktop) | Stable, on by default | Windows, macOS, Linux unflagged 2024 |
| Chrome 121+ (Android) | Stable, on by default | Android 12+ on Adreno/Mali GPUs |
| Edge 113+ | Stable, on by default | Chromium-based |
| Safari 26 | Stable, on by default | macOS Tahoe 26, iOS 26, iPadOS 26, visionOS 26. Released Sept 2025. |
| Firefox 141+ (Windows) | Stable | July 2025 |
| Firefox 145+ (macOS ARM) | Stable | Late 2025 |
| Firefox (Linux / Android / Intel Mac) | Behind flag | Mozilla targets shipping through 2026 |

**WebGPU features that matter for Whisper:**

- `"shader-f16"` GPU feature. Enables WGSL `f16` type and ORT's `q4f16` / `fp16` kernels. Available on most desktop / mobile GPUs in 2026. Without it you fall back to fp32 shaders, around 2x slower and 2x memory.
- Buffer mapping with `mapAsync` for zero-copy uploads of audio mels.
- IO binding via `Tensor.fromGpuBuffer()` keeps encoder output on GPU between encoder and decoder calls. critical to avoid per-token PCIe / unified-memory round trips.
- Graph capture (`enableGraphCapture: true`) for the encoder (static shape, 30 s window). Decoder cannot graph-capture because seq length changes per step.

**Known WebGPU issues in 2026 (still alive):**

- fp16 / q4f16 produce NaN on some encoder/decoder models. Workaround: stay on fp32 encoder + q4 (integer) decoder.
- Device-loss after extended runs (memory pressure or driver timeouts). Listen for `device.lost` and recreate the session.
- Buffer pool growth: ORT does not aggressively reclaim transient buffers. Call `session.release()` + `await device.queue.onSubmittedWorkDone()` between long jobs.
- macOS Safari 26 has stricter timeout-detection-and-recovery than Chrome. long encoder runs on large-v3 can be killed.

### 2.2 WASM (WebAssembly)

The fallback for everyone without WebGPU and the **faster path on Apple Silicon for small Whisper models** today.

Requirements for the fast path:

| Feature | How to enable | Why it matters |
|---|---|---|
| WASM SIMD (fixed-width) | Default in ORT Web v1.19+; required (non-SIMD builds dropped) | Around 2x over scalar |
| WASM threads (Wasm-MT) | Build with `--enable_wasm_threads`; runtime requires `crossOriginIsolated === true` | Around 1.5x to 2x on top of SIMD |
| Cross-origin isolation | Send `Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy: require-corp` (or `credentialless`) | Unlocks SharedArrayBuffer, which unlocks Wasm threads |
| `env.wasm.numThreads` tuning | Set to `Math.min(hardwareConcurrency - 1, 8)` | Diminishing returns past 4 to 8 |

Microsoft reports MobileNet V2 sees **3.4x speedup** with SIMD + threads vs scalar single-thread WASM. Whisper-tiny encoder behaves similarly.

### 2.3 WebNN

Web Neural Network API. Promising for NPU offload (Apple Neural Engine, Qualcomm Hexagon, Intel NPU), but in 2026 it is **still gated behind flags** in Chrome / Edge on most platforms. The ORT Web `webnn` EP exists (`onnxruntime-web/experimental`) but operator coverage is incomplete for Whisper's attention kernels. Not a production path yet; revisit in late 2026 / early 2027.

---

## 3. Models in this runtime context

Useful for sizing infra; full model-side analysis lives in another doc.

| Model | Params | Encoder size (fp32) | Decoder size (fp32) | q4 total | Notes |
|---|---|---|---|---|---|
| whisper-tiny / .en | 39M | around 36 MB | around 185 MB | around 50 to 80 MB | Browser-friendly; fits free quota easily |
| whisper-base | 74M | around 75 MB | around 280 MB | around 100 to 150 MB | The default for `realtime-whisper-webgpu` demo |
| whisper-small | 244M | around 250 MB | around 700 MB | around 280 MB | Slower than realtime on WASM CPU |
| whisper-large-v3-turbo | 809M | around 620 MB enc | around 280 MB dec (only 4 layers) | around 500 to 800 MB | Best browser sweet spot in 2026 |
| whisper-large-v3 | 1.55B | around 1.2 GB | around 1.8 GB | around 1.0 GB | Requires ONNX external data on Web |
| moonshine-tiny | 27M | around 30 MB combined |  | around 15 MB | 5x faster than whisper-tiny for short utterances |
| moonshine-base | 61M | around 65 MB combined |  | around 30 MB | Real-time on commodity laptop CPU |

The whisper-large-v3-turbo decoder reduction (32 to 4 layers) is exactly the architectural change that makes it browser-viable: the encoder is the **fixed cost** for a 30 s window (one pass per chunk), and the decoder runs per token, so fewer decoder layers means roughly linear decode speedup. q4 quantization of the decoder is mostly safe (encoder is far more sensitive). The recommended Off-The-Record default is:

```js
const transcriber = await pipeline(
  "automatic-speech-recognition",
  "onnx-community/whisper-large-v3-turbo",
  {
    device: "webgpu",
    dtype: {
      encoder_model: "fp32",          // q4 encoder degrades WER noticeably
      decoder_model_merged: "q4",     // safe and around 4x smaller than fp32
    },
  },
);
```

For Apple Silicon laptops where WASM beats WebGPU at this scale, omit `device: "webgpu"` (or fall back if `!navigator.gpu`) and set `dtype: { encoder_model: "fp32", decoder_model_merged: "q4" }`. WASM SIMD+threads + integer q4 is the practical winning combo.

---

## 4. Audio capture stack

### 4.1 Capture primitives

| API | Use it for | Avoid because |
|---|---|---|
| `getUserMedia({ audio: true })` | Acquire a `MediaStream` |  |
| `MediaRecorder` | "record file, then transcribe" workflow (encodes to Opus/webm) | High latency, lossy, not raw PCM |
| `ScriptProcessorNode` | Never | **Deprecated**. Main-thread. Drops samples under UI load. |
| `AudioWorklet` (`AudioWorkletNode`) | Real-time live transcription | The right tool. Runs on the audio thread. |
| Web Audio `AnalyserNode` | VU meters, VAD energy gating | Doesn't deliver raw samples in a stream |

The canonical real-time capture graph for Whisper is:

```
MediaStream, MediaStreamAudioSourceNode, AudioWorkletNode, postMessage, Worker (ASR)
```

### 4.2 Sample-rate handling

`AudioContext.sampleRate` is **48 000 Hz on almost every modern desktop / iOS device** and is fixed once any node is created. Whisper demands **16 kHz mono Float32**. Two paths:

1. **Construct `new AudioContext({ sampleRate: 16000 })`**. works on Chromium, ignored on Safari, flaky if any element on the page is already playing 48 kHz. Don't rely on it as the only path.
2. **Resample inside the AudioWorklet**. The Worklet's `process()` callback fires with 128-sample blocks (around 2.67 ms at 48 kHz). Buffer them, decimate 3:1 (48 to 16 kHz). Linear interpolation is acceptable in practice (Whisper's mel filterbank smooths over it); for higher fidelity use `libsamplerate-js` (WASM port of libsamplerate, supports SINC_BEST_QUALITY / SINC_MEDIUM_QUALITY / LINEAR converters).

### 4.3 Inter-thread transfer

- `postMessage(arrayBuffer, [arrayBuffer])` with the buffer in the transfer list moves ownership without copying. Use this for periodic 1 to 5 s audio chunks to the ASR worker.
- For very tight loops (every 128 samples) use a **lock-free ring buffer in SharedArrayBuffer**. `Atomics.load` / `Atomics.store` on Int32 head/tail indices, Float32 payload. Mozilla bug 1754400 shows `postMessage` with very large payloads crashes Firefox; SAB sidesteps it.
- Worklets cannot import scripts dynamically; bundle them inline via `URL.createObjectURL(new Blob([code]))`.

### 4.4 LocalAgreement-2 pipeline notes

LocalAgreement-2 (Macháček et al., 2023, [arxiv 2307.14743](https://arxiv.org/abs/2307.14743)) confirms tokens as "stable" when two consecutive overlapping Whisper passes agree on the prefix. The browser implementation:

- Maintain a rolling audio buffer of around 30 s.
- Every `T` seconds (typical T = 1 s), run Whisper on the buffer.
- Diff the new transcription against the previous one (token-level, longest common prefix).
- "Confirm" tokens whose substring matches the previous run; show the new tail as "tentative" in a dimmed style.
- When the buffer reaches 30 s, slice off the confirmed prefix and continue with the unconfirmed remainder.

Decoder KV cache reuse across overlapping passes is **not currently exposed** through transformers.js. every pass starts from scratch. This is the single biggest performance lever still on the table; until the runtime exposes stateful decoder caches (Ratchet-style), you pay the full decode cost per LocalAgreement step. Mitigate by using a smaller model (turbo / moonshine) and a longer T (1.5 to 2 s).

---

## 5. Storage of model weights

### 5.1 Quotas (2026)

| Browser | Per-origin quota |
|---|---|
| Chrome / Edge | Up to 60% of disk per origin, 80% per browser; eviction in best-effort buckets when low on space |
| Firefox | min(10% of disk, 10 GiB) per group; best-effort eviction |
| Safari (iOS/macOS) | Around 60% of disk, with 7-day eviction for non-installed PWAs since iOS 17 |

For Off The Record: persistent storage requires `navigator.storage.persist()`, which prompts the user on Firefox/Safari and is automatically granted in Chrome if the site is installed as a PWA or marked as bookmarked.

### 5.2 API choice for weights

| API | Pros | Cons | Verdict for ASR models |
|---|---|---|---|
| **Cache API** | Native Request/Response objects; no serialization; native HTTP semantics; works with `fetch`; persistent | Slightly fiddly for in-memory blobs (need synthetic Response) | **Recommended by Chrome's storage team** for AI model caching |
| **OPFS** | Highest raw read/write throughput; multi-GB single files; in-place writes | Worker-only for sync handles; serialization needed if data isn't already file-shaped | Best for large single-file models (`model.onnx_data`); around 4x faster reads than IndexedDB |
| **IndexedDB** | Mature, universal | Serializes on read AND write; slow for large blobs; per-record size limits in some implementations | Avoid for weights; fine for transcription metadata |

For transformers.js the easy path: configure the runtime to fetch model files via the Cache API (the library already does HTTP fetches against the Hub, so a Service Worker `cache.match()` plus `cache.put(request, response.clone())` is sufficient). For very large models (turbo q4 around 500 to 800 MB; large-v3 around 1 GB+), OPFS is worth the migration.

### 5.3 Service Worker considerations

- Service Workers can host transformers.js, but be aware of issue [#787](https://github.com/huggingface/transformers.js/issues/787): WebGPU and the WASM threaded build are unavailable inside a classic Service Worker context (no SharedArrayBuffer, no GPU). Run inference in a **dedicated Worker** controlled by the page, and use the Service Worker only for caching the model bytes.

---

## 6. Cross-origin isolation requirements

To run threaded WASM (which gives around 2x over single-thread on most CPUs) the page must be cross-origin isolated:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

(Or `credentialless` instead of `require-corp` if you embed third-party resources without CORP headers, supported in Chromium since 2022.)

Consequences:
- All third-party assets (images, fonts, scripts, iframes) must send `Cross-Origin-Resource-Policy: cross-origin` or `Access-Control-Allow-Origin` (so a `crossorigin` fetch succeeds).
- Many ad networks and embeds will fail silently. pick your domain isolation strategy carefully.
- `SharedArrayBuffer` is only constructable when `crossOriginIsolated === true`.
- WebGPU itself does **not** require COI, but threaded WASM does. If you accept single-thread WASM as the fallback, you can skip COI and keep embeds working.

A practical pattern: serve the transcription view from a subdomain (`record.example.com`) with strict COI, and keep the marketing site (`example.com`) un-isolated.

---

## 7. Browser support matrix (May 2026)

| Feature | Chrome | Edge | Safari | Firefox |
|---|---|---|---|---|
| WebGPU (desktop) | 113+ yes | 113+ yes | 26 yes (Tahoe) | 141+ Win, 145+ macOS ARM; Linux behind flag |
| WebGPU (mobile) | Android 12+ yes |  | iOS 26 yes | Android: flag |
| WebGPU `shader-f16` | yes on most GPUs | yes | yes Apple Silicon | yes |
| WASM SIMD | yes | yes | yes | yes |
| WASM threads | yes (needs COI) | yes (needs COI) | yes 15+ (needs COI) | yes (needs COI) |
| SharedArrayBuffer | yes (needs COI) | yes | yes | yes |
| AudioWorklet | 66+ yes | 79+ yes | 14.1+ yes | 76+ yes |
| OPFS | 86+ yes | 86+ yes | 15.2+ yes | 111+ yes |
| Cache API | yes | yes | yes | yes |
| WebNN | Behind flag | Behind flag | no | no |
| Service Worker | yes | yes | yes | yes |
| MediaRecorder Opus | yes | yes | 14.1+ yes | yes |
| `navigator.storage.persist()` | Auto-grant for installed PWA | Auto-grant | Prompts | Prompts |

---

## 8. Performance benchmarks (collected from public reports, 2024 to 2026)

All numbers below are real-time-factor (RTF) or wall-clock for transcribing **60 s of audio** unless stated. Hardware varies, use as relative indicators, not absolutes.

### 8.1 transformers.js on Apple M2 Mac mini (issue #894, Oct 2024, ORT 1.19)

| Config | WebGPU | WASM |
|---|---|---|
| whisper-base encoder fp32 + decoder q4 | 9.5 s | **5.9 s** |
| whisper-base encoder fp32 + decoder fp32 | 9.6 s | **4.9 s** |
| whisper-base encoder q8 + decoder q8 | 27 s | **5.2 s** |

Interpretation: On Apple Silicon at this scale **WASM beats WebGPU**. The encoder is small enough that GPU dispatch overhead dominates; q8 on WebGPU shows pathological behavior due to per-op overhead.

### 8.2 transformers.js whisper-tiny / base WebGPU on discrete NVIDIA

- Tiny: around 7 s per minute of audio (around 8.6x real-time).
- Base: around 20 s per minute of audio (around 3x real-time).
- Small: around 90 s per minute of audio (slower than real-time).

### 8.3 whisper.cpp WASM in the browser (ggml.ai bench)

- Tiny / base: 2x to 3x real-time on a modern desktop CPU (60 s in 20 to 30 s).
- Firefox cannot load >256 MB WASM modules, Chrome required for small/medium.

### 8.4 Moonshine ONNX Runtime Web demo

- Mean inference time 0.14 s for a short utterance on commodity CPU; reported **27x real-time factor**.
- Around 75 ms latency claimed for the streaming variant. Best-in-class for live captioning today.

### 8.5 WebGPU vs WASM for LLMs (context for "WebGPU's strengths")

- TinyLlama-1.1B generating 128 tokens: WebGPU 25 to 40 TPS on discrete NVIDIA, 15 to 25 TPS on Apple M2 integrated; WASM 2 to 6 TPS. WebGPU wins **as soon as the model is large enough or the workload is autoregressive enough** that GPU dispatch overhead amortizes.

**Takeaway for Whisper**: WebGPU pays off above the small variant (244M params). For tiny/base on Apple Silicon, ship WASM. Detect at runtime and pick.

---

## 9. Pitfalls (real, observed)

1. **WebGPU memory leak** in transformers.js Whisper pipeline (#860). Long sessions OOM the GPU. Hold a single Singleton, dispose between long jobs, listen for `device.lost`.
2. **fp16 / q4f16 on WebGPU** produces NaN / overflow on Whisper decoder, Gemma 3, nanochat. Use fp32 encoder + integer-quantized decoder.
3. **AudioContext starts at 48 kHz** even if you ask for 16 kHz, especially on Safari and Firefox or when other audio is playing on the page. Always resample defensively in the worklet.
4. **`postMessage` with large Float32Array** (multi-MB) is a perf cliff in Firefox (bug 1754400). Transfer the buffer (second arg of `postMessage`) or use SharedArrayBuffer.
5. **Cross-origin isolation breaks third-party embeds** (ads, social widgets). Plan the isolation boundary.
6. **`AudioWorklet.process()` runs on the audio thread**, never block it. No big buffers, no model inference there. Buffer + transfer to a Worker.
7. **`navigator.storage.persist()` is required** to avoid eviction of multi-hundred-MB models. Detect storage pressure with `navigator.storage.estimate()` and warn the user.
8. **ScriptProcessorNode** still works in 2026 but drops samples whenever the main thread is busy (e.g., model loading, UI re-renders). Don't use it.
9. **Service Workers can't use WebGPU or SAB**, host the model in a dedicated Worker.
10. **Whisper hallucinates on silence**. Front-end the worker with a VAD (Silero ONNX is around 2 MB and runs <1 ms on CPU; transformers.js ships it via `onnx-community/silero-vad`).
11. **Whisper word-timestamps in long-form chunking are buggy** in current transformers.js. see issues #1357, #1358. If you need word timestamps, force `chunk_length_s` to the model's training window and avoid stride overlap.
12. **The Web Audio API mixes microphone audio with output audio at 48 kHz** if any element on the page is playing audio. Construct the `AudioContext` early, before any `<audio>` / `<video>` element initializes.
13. **`device.lost` events are silent** unless you subscribe, always `device.lost.then(handler)` immediately after `requestDevice()`.
14. **`enableGraphCapture: true`** crashes on dynamic-shape kernels, apply it only to the encoder InferenceSession, never the decoder.
15. **Loading 1+ GB models on slow networks** without progress reporting feels broken. Use `fetch` + `ReadableStream` and emit progress events to a UI ring.

---

## 10. Best practices for Off The Record's stack

Stack contract: **transformers.js v3 + WebGPU + LocalAgreement-2 + AudioWorklet capture.**

### Recommended config

```ts
// Detect capabilities
const hasWebGPU = !!navigator.gpu;
const isAppleSilicon = /Mac/.test(navigator.platform) && navigator.hardwareConcurrency >= 8;

const device: "webgpu" | "wasm" =
  hasWebGPU && !isAppleSilicon ? "webgpu" : "wasm";

const dtype = {
  encoder_model: "fp32",          // never q4/fp16 on the encoder
  decoder_model_merged: "q4",     // safe; could try q4f16 on non-Apple WebGPU
};

ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency - 1, 8);
ort.env.wasm.simd = true;
ort.env.webgpu.powerPreference = "high-performance";

const transcriber = await pipeline(
  "automatic-speech-recognition",
  "onnx-community/whisper-large-v3-turbo",
  { device, dtype }
);
```

### Architecture

- **One** dedicated `Worker` hosts transformers.js. Keep it warm with a Singleton pipeline.
- **One** AudioWorklet captures from the mic, resamples 48 to 16 kHz, posts 1 s Float32Array chunks (via Transferable) to the ASR Worker.
- **Service Worker** caches model files via Cache API, serves `precache + stale-while-revalidate`.
- Page is cross-origin isolated (`COOP: same-origin`, `COEP: require-corp`).
- VAD (Silero) gates ASR, skip transcription on silent windows.
- LocalAgreement-2 confirmer lives on the **main thread** (cheap diff, runs once per second).
- On WebGPU error / `device.lost`, fall back to WASM transparently and re-run the failed chunk.
- On every Nth chunk (around 60), call `model.dispose()` + recreate the pipeline to defeat the GPU memory leak.

### Model selection logic

```
if WebGPU + discrete or Apple Silicon with >16GB:
    whisper-large-v3-turbo (q4 decoder)
elif WebGPU + integrated, low memory:
    whisper-base (q4 decoder)
elif WASM-only, Apple Silicon:
    whisper-base or moonshine-base
elif slow CPU / mobile / no WebGPU:
    moonshine-tiny (or whisper-tiny if multilingual required)
```

### Verification / testing

- Test in **Chrome 124+**, **Firefox 145+**, **Safari 26 macOS**, and **Chrome Android 12+**.
- Profile with `ort.env.webgpu.profiling = "default"` and `performance.measureUserAgentSpecificMemory()`.
- Re-benchmark when ORT Web ships fp16/q4f16 fixes, q4f16 will likely become the default for both encoder and decoder once stable.

---

## 11. Source URLs

### transformers.js
- v3 launch blog: https://huggingface.co/blog/transformersjs-v3
- WebGPU guide: https://huggingface.co/docs/transformers.js/guides/webgpu
- dtypes guide: https://huggingface.co/docs/transformers.js/guides/dtypes
- GitHub: https://github.com/huggingface/transformers.js
- Releases (v3.0 to v3.8): https://github.com/huggingface/transformers.js/releases
- realtime-whisper-webgpu example: https://github.com/huggingface/transformers.js-examples/tree/main/realtime-whisper-webgpu
- WebGPU memory leak (#860): https://github.com/huggingface/transformers.js/issues/860
- WebGPU vs WASM Whisper perf (#894): https://github.com/huggingface/transformers.js/issues/894
- Service Worker limitation (#787): https://github.com/huggingface/transformers.js/issues/787
- Long-form timestamp bugs (#1357, #1358): https://github.com/huggingface/transformers.js/issues/1357
- Whisper Webgpu demo: https://huggingface.co/spaces/Xenova/whisper-webgpu
- Real-time Whisper WebGPU demo: https://huggingface.co/spaces/Xenova/realtime-whisper-webgpu
- whisper-web repo: https://github.com/xenova/whisper-web

### ONNX Runtime Web
- WebGPU EP docs: https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html
- Env flags / session options: https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html
- Build for web: https://onnxruntime.ai/docs/build/web.html
- WebNN EP: https://onnxruntime.ai/docs/tutorials/web/ep-webnn.html
- Large models / external data: https://onnxruntime.ai/docs/tutorials/web/large-models.html
- Releases: https://github.com/microsoft/onnxruntime/releases
- WebGPU EP merge PR (#14579): https://github.com/microsoft/onnxruntime/pull/14579
- fp16 / q4f16 overflow bug (#26732): https://github.com/microsoft/onnxruntime/issues/26732
- WebGPU fp16 NaN bug (#26367): https://github.com/microsoft/onnxruntime/issues/26367

### Models on Hugging Face
- whisper-large-v3-turbo (ONNX): https://huggingface.co/onnx-community/whisper-large-v3-turbo
- whisper-base (ONNX): https://huggingface.co/onnx-community/whisper-base
- Moonshine: https://github.com/moonshine-ai/moonshine
- Moonshine streaming: https://huggingface.co/UsefulSensors/moonshine-streaming-tiny
- Moonshine demo: https://github.com/usefulsensors/moonshine/blob/main/demo/README.md

### WebGPU / WebNN / browser status
- WebGPU caniuse: https://caniuse.com/webgpu
- WebGPU implementation status: https://github.com/gpuweb/gpuweb/wiki/Implementation-Status
- WebGPU baseline announcement: https://web.dev/blog/webgpu-supported-major-browsers
- WebNN overview (MS): https://learn.microsoft.com/en-us/windows/ai/directml/webnn-overview
- WebNN compatibility: https://webnn.io/en/api-reference/browser-compatibility/api
- Chrome WebGPU on Android (Chrome 121): https://developer.chrome.com/blog/new-in-webgpu-121

### Other runtimes
- whisper.cpp WASM stream demo: https://ggml.ai/whisper.cpp/stream.wasm/
- whisper.cpp WASM bench: https://ggml.ai/whisper.cpp/bench.wasm/
- whisper.cpp repo: https://github.com/ggml-org/whisper.cpp
- sherpa-onnx repo: https://github.com/k2-fsa/sherpa-onnx
- sherpa-onnx WASM docs: https://deepwiki.com/k2-fsa/sherpa-onnx/3.8-webassembly-support
- vosk-browser: https://www.npmjs.com/package/vosk-browser
- Picovoice Leopard: https://picovoice.ai/docs/leopard/
- Picovoice Cheetah: https://picovoice.ai/docs/cheetah/
- WebLLM: https://github.com/mlc-ai/web-llm
- whisper.rn (React Native): https://github.com/mybigday/whisper.rn

### Audio / storage / isolation
- Cross-origin isolation guide: https://web.dev/articles/cross-origin-isolation-guide
- COOP/COEP: https://web.dev/articles/coop-coep
- SharedArrayBuffer (MDN): https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer
- OPFS (MDN): https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system
- OPFS deep dive (web.dev): https://web.dev/articles/origin-private-file-system
- Cache AI models in browser (Chrome docs): https://developer.chrome.com/docs/ai/cache-models
- Storage quotas (MDN): https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria
- AudioWorklet (Chrome blog): https://developer.chrome.com/blog/audio-worklet
- libsamplerate-js: https://github.com/aolsenjazz/libsamplerate-js
- wave-resampler: https://github.com/rochars/wave-resampler

### LocalAgreement / streaming Whisper
- Macháček et al. "Turning Whisper into Real-Time Transcription System": https://arxiv.org/abs/2307.14743
- whisper_streaming reference impl: https://github.com/ufal/whisper_streaming
- WhisperLiveKit: https://github.com/QuentinFuxa/WhisperLiveKit
