# Whisper Ecosystem, State of the Art (Offline / On-Device STT)

Research compiled for the **Off The Record** project (browser-based offline transcription, transformers.js + WebGPU + LocalAgreement-2 streaming).

Scope: **Whisper and its derivatives only**. Excludes Chinese models, NVIDIA NeMo/Parakeet, Meta MMS, etc.; those are handled by sibling research docs.

---

## 1. Models, OpenAI Whisper family

OpenAI Whisper is a Transformer encoder/decoder seq2seq model trained on **680,000 hours** of weakly-supervised audio (v1/v2). Input is split into 30 second chunks, converted to log-Mel spectrograms (80 bins for v1/v2, **128 bins for v3**), passed through the encoder, and autoregressively decoded. The decoder emits both content tokens and special tokens for language ID, voice activity, translation, and timestamps.

Paper: *Robust Speech Recognition via Large-Scale Weak Supervision* (Radford et al., 2022), https://cdn.openai.com/papers/whisper.pdf

| Model | Params | Encoder layers | Decoder layers | Mel bins | Multilingual | English-only `.en` | Disk (FP16) | Relative speed |
|---|---|---|---|---|---|---|---|---|
| tiny | 39 M | 4 | 4 | 80 | yes | yes | 75 MB | ~10x |
| base | 74 M | 6 | 6 | 80 | yes | yes | 142 MB | ~7x |
| small | 244 M | 12 | 12 | 80 | yes | yes | 466 MB | ~4x |
| medium | 769 M | 24 | 24 | 80 | yes | yes | 1.5 GB | ~2x |
| large-v1 | 1550 M | 32 | 32 | 80 | yes | no | 2.9 GB | 1x |
| large-v2 | 1550 M | 32 | 32 | 80 | yes | no | 2.9 GB | 1x |
| **large-v3** | 1550 M | 32 | 32 | **128** | yes (+ Cantonese) | no | 2.9 GB | 1x |
| **large-v3-turbo** | **809 M** | 32 | **4** | 128 | yes (99 langs) | no | 1.6 GB | ~8x |

### Key version notes

- **large-v2** (Dec 2022): same arch as v1, trained 2.5x longer with regularization.
- **large-v3** (Nov 2023): 128 Mel bins (vs 80), new Cantonese language token, trained on 1M hr weak-labeled + 4M hr pseudo-labeled (with v2 teacher) = 5M hr. ~10 to 20% WER reduction vs v2.
- **large-v3-turbo** (Oct 2024): pruned large-v3 with decoder layers reduced **32 to 4**. Params 1550M to 809M (~1.78x smaller, 2 to 5x faster). Two extra fine-tune epochs on the same v3 multilingual data, excluding translation. WER comparable to large-v2 on most languages; larger degradation on Thai, Cantonese. Translation quality not preserved.

### Repo / artifacts

- **GitHub**: https://github.com/openai/whisper, 99.5k stars, MIT, last release v20250625 (June 2025), still maintained for occasional updates.
- **HF org**: https://huggingface.co/openai
  - `openai/whisper-tiny`, `whisper-tiny.en`, `whisper-base`, `whisper-base.en`, `whisper-small`, `whisper-small.en`, `whisper-medium`, `whisper-medium.en`, `whisper-large`, `whisper-large-v2`, `whisper-large-v3`, `whisper-large-v3-turbo`.

### Published WER (large-v3-turbo, Open ASR Leaderboard)

| Dataset | WER (%) |
|---|---|
| LibriSpeech clean | 2.10 |
| LibriSpeech other | 4.24 |
| Earnings22 | 11.63 |
| GigaSpeech | 10.14 |
| SpGISpeech | 2.97 |
| AMI | 16.13 |
| TED-LIUM | ~3.9 |
| **Mean (Open ASR LB)** | **7.83** |
| RTFx | 200.19 |

Whisper large-v3 LibriSpeech test-clean WER ~ 2.7%; medium.en test-clean ~ 3.0%, test-other ~ 7.5%; tiny 10 to 15%.

---

## 2. Inference engines

### 2.1 Faster-Whisper (CTranslate2)

- **Repo**: https://github.com/SYSTRAN/faster-whisper, 22.9k stars, MIT, latest v1.2.1 (Oct 2025), actively maintained.
- Reimplementation of Whisper on **CTranslate2** (custom Transformer inference engine in C++ with INT8/INT16/FP16/BF16 kernels).
- **Quantization**: `float32`, `float16`, `int8_float16`, `int8` (CPU + GPU). INT8 CPU runs at ~1/4 original time; FP16 GPU achieves ~5 to 6x speedup with no accuracy loss.
- **Batched inference** via `BatchedInferencePipeline` (since v1.0).
- Benchmarks: large-v2 on RTX 3070 Ti: FP16 1m03s, INT8 batch=8 16s (vs openai/whisper 2m23s). Large-v3 FP16 RTF 2.883, INT8 4.594; large-v3-turbo RTF ~1.92.
- Supports **distil-whisper checkpoints** natively.
- Used as backend by WhisperX, WhisperLive, whisper_streaming, stable-ts.

### 2.2 whisper.cpp (Georgi Gerganov / ggml-org)

- **Repo**: https://github.com/ggml-org/whisper.cpp, 49.7k stars, MIT, latest v1.8.4 (March 2026), very actively maintained.
- Pure C/C++ inference using **ggml** tensor library; no Python, no Torch.
- **Backends** (configurable at compile time):
  - Apple Silicon: ARM NEON, Accelerate (vDSP), **Metal**, **Core ML** (encoder on Apple Neural Engine, >3x faster than CPU-only)
  - x86: AVX/AVX2/AVX-512
  - GPU: **CUDA** (cuBLAS), **Vulkan**, **OpenVINO**, **Moore Threads MUSA**
  - CPU BLAS: OpenBLAS
  - **WebAssembly**: SIMD-required, ~2 to 3x real-time for tiny/base on modern CPU
  - NPU: Ascend CANN
  - POWER: VSX
- **Quantization** (GGUF-style, integer types):
  - `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0` (k-quants such as `Q4_K`, `Q5_K` are llama.cpp-specific and are not standard for whisper.cpp; whisper.cpp uses the older non-K quant types).
  - Convert FP16 to 4/5/8-bit integer weights with the `quantize` tool.
- **Distil-Whisper support**: initial integration since 2023 (issue #1423), pre-converted GGML at `distil-whisper/distil-large-v3-ggml` and `distil-large-v3.5-ggml`. Chunk-based transcription strategy not fully implemented, so quality is sub-optimal on long-form vs the original PyTorch path.
- **Streaming demo**: `examples/stream` (microphone, half-second sampling); `examples/whisper.wasm` for browser; `stream.wasm` for browser real-time.
- WASM live demos: https://whisper.ggerganov.com/ and https://ggml.ai/whisper.cpp/stream.wasm/
- HF model hub: https://huggingface.co/ggerganov/whisper.cpp (all sizes by all quantization levels).

### 2.3 WhisperJAX (Sanchit Gandhi)

- **Repo**: https://github.com/sanchit-gandhi/whisper-jax, Apache-2.0, ~4.5k stars.
- JAX implementation, JIT-compiled, designed for **TPU**.
- For 1 hour of audio: OpenAI Whisper PyTorch on A100 = 1001 s; HF Transformers PyTorch A100 = 126.1 s; Whisper-JAX A100 = 75.3 s; **Whisper-JAX TPU v4-8 = 13.8 s**, so ~70x speedup over reference.
- Batched chunked inference. Released April 2023; maintenance now sporadic (Sanchit moved to other projects).

### 2.4 insanely-fast-whisper (Vaibhav Srivastav)

- **Repo**: https://github.com/Vaibhavs10/insanely-fast-whisper, 12.9k stars, Apache-2.0.
- CLI wrapping HF Transformers + Optimum + **Flash Attention 2** + BetterTransformer.
- Benchmark on A100-80GB: 150 minutes of audio transcribed in ~98 s with large-v3 + FP16 + batch=24 + FA2. Distil-large-v2 + FA2 = ~78 s.
- Supports CUDA and **MPS** (Mac); FA2 requires modern NVIDIA GPUs, else falls back to BetterTransformer.

### 2.5 WhisperKit / Argmax OSS Swift (Argmax Inc.)

- **Repo**: https://github.com/argmaxinc/WhisperKit (now consolidated into https://github.com/argmaxinc/argmax-oss-swift), 6.1k stars, MIT.
- Native Swift + **CoreML**, optimized for Apple **Neural Engine (ANE)**. Encoder fully on ANE.
- HF: https://huggingface.co/argmaxinc/whisperkit-coreml (large-v3, large-v3-turbo, distil variants prebuilt).
- ICML 2025 paper: *WhisperKit: On-device Real-time ASR with Billion-Scale Transformers*, https://arxiv.org/abs/2507.10860, reports 2.2% WER (LibriSpeech) with large-v3-turbo at real-time streaming latency on iPhone/Mac.
- Platforms: macOS 14+, iOS 16+. Companion SpeakerKit (pyannote v4 ANE port) and TTSKit (Qwen-TTS).
- Caveat: large encoders OOM on 4 GB iPhones; runs fine on M-series Mac with at least 16 GB unified memory.

### 2.6 MLX-Whisper (Apple MLX)

- **Repo**: https://github.com/ml-explore/mlx-examples (whisper subdir), MIT.
- PyPI `mlx-whisper`. Uses Apple's MLX array framework (GPU on Apple Silicon, unified memory).
- 30 to 40% faster than vanilla PyTorch on M-series; on a 10-min file M1 Pro took 216 s vs RTX 4090 186 s (Mac ~16% slower than 4090).
- Used as backend in `whisper_streaming` and `stable-ts`.

### 2.7 candle-whisper (HF Candle, Rust)

- **Repo**: https://github.com/huggingface/candle (example: `candle-examples/examples/whisper/`), Apache-2.0 / MIT, 17k+ stars.
- Minimalist Rust ML framework, megabyte-sized binaries, supports CPU/CUDA/Metal and **WebAssembly**.
- HF demo space: https://huggingface.co/spaces/lmz/candle-whisper (Whisper running entirely client-side in browser via Candle WASM).

### 2.8 transformers.js Whisper (Xenova / HF)

- **Repo**: https://github.com/huggingface/transformers.js (formerly xenova/transformers.js), Apache-2.0.
- **ONNX Runtime Web** backend with **WASM** (default) and **WebGPU** support (since v3, Oct 2024).
- Models published at https://huggingface.co/Xenova/whisper-* (tiny, tiny.en, base, base.en, small, small.en, medium, large-v3, large-v3-turbo, distil-* variants).
- ONNX quantization variants per model: `fp32`, `fp16`, `q8`/`int8`/`uint8`, `q4`, `q4f16`, `bnb4`. Encoder is sensitive to aggressive quantization, so `q4f16` (4-bit weights, fp16 compute) is a typical sweet spot.
- Choose dtype via `dtype: 'fp16'` (WebGPU default), `q8` (WASM default), `q4` (smallest).
- **Demos**:
  - https://huggingface.co/spaces/Xenova/whisper-web, vanilla, file upload
  - https://huggingface.co/spaces/Xenova/realtime-whisper-webgpu, real-time microphone, fully on-device, multilingual
- Repo: https://github.com/xenova/whisper-web (template for in-browser).

---

## 3. Streaming variants

### 3.1 whisper_streaming (UFAL, Machacek & Bojar 2023)

- **Repo**: https://github.com/ufal/whisper_streaming, 3.6k stars, MIT.
- **Paper**: *Turning Whisper into Real-Time Transcription System*, arXiv https://arxiv.org/abs/2307.14743 (Machacek, Dabre, Bojar; IJCNLP-AACL 2023 system demo).
- **LocalAgreement-n** algorithm: only commit a prefix to output when *n* consecutive update windows agree on it (longest common prefix on the unconfirmed tail of two subsequent decodes). LocalAgreement-2 is the practical choice.
- Self-adaptive latency; **3.3 s** end-to-end latency on unsegmented long-form test sets.
- Backends: faster-whisper (recommended), whisper-timestamped, OpenAI Whisper API, MLX-Whisper.
- Includes VAC (Voice Activity Controller), buffer-trimming strategies (sentence/segment), simulation modes, TCP server.
- **Critical for Off The Record**: this is the canonical LocalAgreement-2 reference implementation.

### 3.2 SimulStreaming (2025 successor)

- **Repo**: https://github.com/ufal/SimulStreaming, MIT.
- Merges ideas from Simul-Whisper + Whisper-Streaming. Removed "Whisper" from the name because it now also supports LLM-based MT.
- Uses **AlignAtt** policy (encoder/decoder attention determines when audio buffer is "safe" to decode further) instead of LocalAgreement-2 polling, so it is **~5x faster** than the original whisper_streaming.
- IWSLT 2025 Simultaneous Speech Translation Shared Task winning entry. Paper: https://arxiv.org/abs/2506.17077
- Lightning-SimulWhisper (Apple Silicon MLX/CoreML port): https://github.com/altalt-org/Lightning-SimulWhisper, claims ~15x perf increase.

### 3.3 WhisperLive (Collabora)

- **Repo**: https://github.com/collabora/WhisperLive, 4k+ stars, MIT.
- Server/client architecture, **WebSocket** streaming, JSON segment output.
- Backends: **faster-whisper**, **NVIDIA TensorRT-LLM**, **OpenVINO**.
- Features: VAD, speaker diarization, hotwords, word-level timestamps + confidence, RTSP/HLS input, OpenAI-compatible REST.
- Clients: Chrome extension, Firefox extension, iOS native, Python.

### 3.4 WhisperLiveKit (Quentin Fuxa)

- **Repo**: https://github.com/QuentinFuxa/WhisperLiveKit, MIT.
- Combines whisper_streaming + SimulStreaming + diarization into a single Python package. PyPI `whisperlivekit`.

### 3.5 CarelessWhisper / WhisperRT

- **Paper**: *CarelessWhisper: Turning Whisper into a Causal Streaming Model* (2025), https://arxiv.org/abs/2508.12301
- LoRA-finetunes Whisper's bidirectional encoder into a causal/streaming encoder, eliminating the need for full 30-s windows for low-latency operation.

### 3.6 Adapting Whisper for Streaming via Two-Pass Decoding (2025)

- **Paper**: https://arxiv.org/abs/2506.12154, alternative streaming approach using CTC/Attention two-pass.

---

## 4. Specialized forks / derivatives

### 4.1 Distil-Whisper (HF)

- **Repo**: https://github.com/huggingface/distil-whisper, 4.1k stars, MIT.
- **Paper**: Gandhi, von Platen, Rush (Nov 2023), *Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling*, https://arxiv.org/abs/2311.00430
- Student keeps full 32-layer encoder, slashes decoder to **2 layers**. Distilled via pseudo-labels filtered by WER heuristic on 22k hrs (v3) / 98k hrs (v3.5) public audio.
- Designed as **draft model for speculative decoding** with large-v3, giving exact-output 2x speedup.

| Model | Params | Decoder | Speedup vs large-v3 | Short WER (OOD) | Long WER | HF ID |
|---|---|---|---|---|---|---|
| distil-small.en | 166 M | 2 | ~6x | 12.1 | 12.8 | `distil-whisper/distil-small.en` |
| distil-medium.en | 394 M | 2 | ~6x | within 1% of medium.en | n/a | `distil-whisper/distil-medium.en` |
| distil-large-v2 | 756 M | 2 | 5.8x | within 1% of large-v2 | n/a | `distil-whisper/distil-large-v2` |
| distil-large-v3 | 756 M | 2 | 6.3x | 9.7 | 10.8 | `distil-whisper/distil-large-v3` |
| **distil-large-v3.5** | 756 M | 2 | ~1.5x faster than turbo | **7.08** | 11.39 (OOD), 4.63 (ID) | `distil-whisper/distil-large-v3.5` |

- v3.5 (March 2025): 4x more data (98k hr incl. YODAS YouTube), "patient" teacher (80 epochs vs 11), aggressive SpecAugment, BPE dropout. **OOD short-form WER 7.08% beats v3 (7.53%)**. Released in multiple formats: PyTorch, ONNX, CTranslate2 (`distil-large-v3.5-ct2`), GGML (`distil-large-v3.5-ggml`), original-OpenAI format.

### 4.2 WhisperX (Max Bain, Oxford VGG)

- **Repo**: https://github.com/m-bain/whisperX, 21.9k stars, **BSD-2-Clause**, last release v3.8.5 (April 2026).
- Pipeline: faster-whisper transcription, then wav2vec2 **forced phoneme alignment** for word-level timestamps (about plus/minus 50 ms vs Whisper's plus/minus 500 ms), then **pyannote.audio** speaker diarization.
- Batched inference enables **~70x real-time** with large-v2 on 8 GB GPU.
- VAD-based pre-segmentation reduces hallucination on long-form.
- Per-language wav2vec2 alignment models auto-downloaded.

### 4.3 whisper-timestamped (Linto AI)

- **Repo**: https://github.com/linto-ai/whisper-timestamped, MIT.
- Word-level timestamps via **Dynamic Time Warping** on cross-attention weights (per Jong Wook Kim's notebook). No extra inference passes when not using beam search.
- Per-word **confidence scores**.
- Used as a streaming backend by whisper_streaming.

### 4.4 CrisperWhisper (Nyra Health)

- **Repo**: https://github.com/nyrahealth/CrisperWhisper, 951 stars, **CC BY-NC 4.0** (non-commercial).
- **Paper**: arXiv https://arxiv.org/abs/2408.16589 (INTERSPEECH 2024).
- Innovations:
  1. **Verbatim transcription**, captures fillers ("uh", "um"), stutters, false starts (vs Whisper's "intended-transcription" style).
  2. **Precise word timestamps** via tokenizer-adjusted DTW on cross-attention scores + custom attention loss.
  3. **Hallucination mitigation**, 1% noise-only training samples with empty transcripts.
- WER 6.66% mean across 9 datasets, beats Whisper Large-v3 (7.7%). AMI 8.72 vs 16.01; TED-LIUM 3.35 vs 3.9. Common Voice segmentation F1 0.80 vs 0.48.
- Checkpoints: `nyrahealth/CrisperWhisper` (transformers), `nyrahealth/faster_CrisperWhisper` (ct2).
- **Licensing caveat**: not usable in commercial product without a separate license, important for AMC Hero / Off The Record commercial deployment.

### 4.5 stable-ts (jianfch)

- **Repo**: https://github.com/jianfch/stable-ts, 2.2k stars, MIT.
- Wraps OpenAI Whisper, faster-whisper, HF Transformers, MLX-Whisper.
- Features: **silence suppression** for timestamp boundaries, `.refine()` iterative timestamp tightening via probability monitoring, plain-text to audio alignment, output to SRT/VTT/ASS/TSV/JSON. Word-level + segment-level.

### 4.6 Other notable forks

| Project | Repo | Notes |
|---|---|---|
| pyannote-whisper | https://github.com/yinruiqing/pyannote-whisper | Diarization wrapper |
| nb-whisperX | https://github.com/NbAiLab/nb.whisperX | Norwegian fine-tune of whisperX |
| OWSM (Open Whisper-style) | https://github.com/espnet/espnet | Open replication via ESPnet |
| VoiceStreamAI | https://github.com/alesaccoia/VoiceStreamAI | WS-based real-time wrapper |
| docker-whisper-live | https://github.com/hwdsl2/docker-whisper-live | Containerized WhisperLive |

---

## 5. Quantization / optimization landscape

| Format | Tooling | Engine | Notes |
|---|---|---|---|
| FP32 | Reference | All | Original PyTorch weights |
| FP16 | All | PyTorch, CT2, GGML, MLX, ONNX | Standard ~2x speedup, no accuracy loss |
| BF16 | CT2, MLX | GPU only | Equivalent to FP16 for inference |
| INT8 dynamic | CT2, ONNX, GGML | CPU + GPU | ~3 to 4x CPU speedup, minimal WER hit |
| `int8_float16` | CTranslate2 | GPU | Weights INT8, activations FP16, best speed/accuracy on GPU |
| `Q8_0` | whisper.cpp/ggml | Any backend | 8-bit integer, near-lossless |
| `Q5_0`, `Q5_1` | whisper.cpp | Any | 5-bit, small WER hit |
| `Q4_0`, `Q4_1` | whisper.cpp | Any | 4-bit, noticeable WER hit on small models, fine on large |
| `q4f16` | transformers.js / ONNX | WebGPU | 4-bit weights, FP16 compute, recommended for browser |
| `q4` (bnb4) | transformers.js / ONNX | WASM/WebGPU | 4-bit; encoder is sensitive, may need q8/fp16 encoder + q4 decoder |
| Flash Attention 2 | PyTorch + flash-attn | CUDA | 1.5 to 2x decoder speedup, requires Ampere or newer |
| BetterTransformer | PyTorch nn.Module | CUDA / MPS | Fused attention, ~30% speedup |
| Speculative decoding | HF Transformers | Any | Use distil-large-v3(.5) as draft for large-v3, exact-output 2x speedup |
| **Pruning (Turbo)** | OpenAI | Native | 32 to 4 decoder layers, 1.78x smaller, 2 to 5x faster |

### Speculative decoding (Whisper)

- HF blog: *Speculative Decoding for 2x Faster Whisper Inference*, https://huggingface.co/blog/whisper-speculative-decoding
- Use `distil-large-v3` as `assistant_model` to `large-v3`; if draft top-k matches teacher, accept; else fall back. Mathematically guarantees identical output to teacher.
- Combine with INT8/FP16 for compound speedup.

### Quantization study

- *Quantization for OpenAI's Whisper Models* (arXiv 2503.09905, 2025), comparative analysis of INT8/INT4 across CTranslate2, llama.cpp, ONNX backends.

---

## 6. Benchmarks summary table

Mean WER on the Open ASR Leaderboard (lower is better), RTFx (higher is faster):

| Model | Backend | Mean WER (%) | RTFx | Notes |
|---|---|---|---|---|
| whisper-large-v3 | PyTorch FP16 | ~7.4 | ~30 | Reference |
| whisper-large-v3-turbo | PyTorch FP16 | 7.83 | 200 | OpenAI's own |
| whisper-large-v3-turbo | faster-whisper INT8 | ~8.0 | ~400+ | CT2 INT8 GPU |
| distil-large-v3 | PyTorch FP16 | ~9.7 (short) | ~190 | HF |
| distil-large-v3.5 | PyTorch FP16 | 7.08 (OOD short) | ~300 | HF, beats turbo on long-form RTFx by 1.5x |
| CrisperWhisper | PyTorch FP16 | 6.66 | ~30 | AMI 8.72 (best in class) |
| WhisperKit (large-v3-turbo) | CoreML / ANE | ~7.8 | varies | On-device iPhone 15 Pro / M-series |
| Whisper-JAX large-v2 | JAX TPU v4-8 | ~9 | ~250 (TPU) | 13.8 s per 1 hr audio |
| whisper.cpp large-v3 Q5_0 | Metal M2 Max | ~7.5 | ~10 (M2 Max) | Quality very close to FP16 |
| whisper.cpp base.en WASM | WebAssembly SIMD | ~12 | 2 to 3 | Browser, CPU only |
| transformers.js large-v3-turbo | WebGPU q4f16 | ~8 | varies | Browser, M2/M3 |

Browser-specific targets (Off The Record context):

| Stack | Model | Quant | Approx in-browser RTFx (M3 Pro) |
|---|---|---|---|
| transformers.js v3 WebGPU | whisper-base | q4f16 | 8 to 15x |
| transformers.js v3 WebGPU | whisper-small | q4f16 | 3 to 6x |
| transformers.js v3 WebGPU | distil-large-v3-turbo | q4f16 | 1.5 to 3x |
| transformers.js v3 WASM SIMD | whisper-base | q8 | 1 to 2x |
| whisper.cpp WASM SIMD | base.en | Q5_0 | 2 to 3x |
| whisper.cpp WASM SIMD | tiny.en | Q5_0 | 4 to 6x |

---

## 7. Browser / WebGPU compatibility matrix

| Project | Browser? | WebGPU | WASM | Notes |
|---|---|---|---|---|
| transformers.js | yes | yes (v3+) | yes (default) | Recommended for browser apps; ONNX Runtime Web backend |
| whisper.cpp wasm | yes | no | yes (SIMD required) | Native browser via Emscripten, CPU-only |
| candle (HF, Rust) | yes | partial | yes | WASM target; WebGPU early-stage |
| whisper-web (Xenova) | yes | yes | yes | Template repo using transformers.js |
| WhisperJAX | no | n/a | n/a | TPU/GPU server only |
| WhisperKit | no (native Apple) | n/a | n/a | iOS / macOS only |
| MLX-Whisper | no | n/a | n/a | Apple Silicon native |
| faster-whisper | no | n/a | n/a | Server CTranslate2 only |
| WhisperLive | yes (client) | no | no | Browser is just a WS client; server-side inference |

For **Off The Record** specifically (browser, offline, WebGPU): **transformers.js v3 + Xenova ONNX checkpoints** is the dominant production-ready stack. whisper.cpp WASM is a viable CPU fallback for non-WebGPU browsers.

---

## 8. Maintenance status snapshot (May 2026)

| Project | Stars | Last release | Active? |
|---|---|---|---|
| openai/whisper | 99.5k | v20250625 | low-key maintenance |
| whisper.cpp | 49.7k | v1.8.4 (Mar 2026) | very active |
| faster-whisper | 22.9k | v1.2.1 (Oct 2025) | very active |
| whisperX | 21.9k | v3.8.5 (Apr 2026) | active |
| insanely-fast-whisper | 12.9k | sporadic | community |
| WhisperKit / argmax-oss-swift | 6.1k | v1.0.0 (May 2026) | very active |
| distil-whisper | 4.1k | v3.5 (Mar 2025) | active |
| WhisperLive | 4k+ | rolling | active |
| whisper_streaming | 3.6k | superseded by SimulStreaming | maintenance only |
| SimulStreaming | new | 2025 | active |
| stable-ts | 2.2k | rolling | active |
| CrisperWhisper | 951 | 2024 | low activity |
| transformers.js | 13k+ | v3.x (rolling) | very active |
| candle | 17k+ | rolling | very active |

---

## 9. Notable papers (arXiv)

| Title | Authors | Year | arXiv |
|---|---|---|---|
| Robust Speech Recognition via Large-Scale Weak Supervision (Whisper) | Radford et al. (OpenAI) | 2022 | [pdf](https://cdn.openai.com/papers/whisper.pdf) |
| Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling | Gandhi, von Platen, Rush | 2023 | [2311.00430](https://arxiv.org/abs/2311.00430) |
| Turning Whisper into Real-Time Transcription System (whisper_streaming, LocalAgreement) | Machacek, Dabre, Bojar | 2023 | [2307.14743](https://arxiv.org/abs/2307.14743) |
| CrisperWhisper: Accurate Timestamps on Verbatim Speech Transcriptions | Nyra Health | 2024 | [2408.16589](https://arxiv.org/abs/2408.16589) |
| WhisperKit: On-device Real-time ASR with Billion-Scale Transformers | Argmax Inc. | 2025 (ICML) | [2507.10860](https://arxiv.org/abs/2507.10860) |
| Simultaneous Translation with Offline Speech and LLM Models (SimulStreaming / IWSLT 2025) | CUNI / Machacek et al. | 2025 | [2506.17077](https://arxiv.org/abs/2506.17077) |
| CarelessWhisper / WhisperRT: Turning Whisper into a Causal Streaming Model | n/a | 2025 | [2508.12301](https://arxiv.org/abs/2508.12301) |
| Adapting Whisper for Streaming Speech Recognition via Two-Pass Decoding | n/a | 2025 | [2506.12154](https://arxiv.org/abs/2506.12154) |
| Quantization for OpenAI's Whisper Models: A Comparative Analysis | n/a | 2025 | [2503.09905](https://arxiv.org/html/2503.09905v1) |
| uDistil-Whisper: Label-Free Data Filtering for Knowledge Distillation | n/a | 2024 | [2407.01257](https://arxiv.org/html/2407.01257v1) |

---

## 10. Recommendations for Off The Record

Distilled from this survey, tailored to a browser-based offline transcription app with WebGPU + LocalAgreement-2 streaming:

1. **Primary stack**: transformers.js v3 + ONNX Runtime Web + WebGPU. Use Xenova/* checkpoints. Default to `q4f16` encoder + decoder for size, fall back to `fp16` if encoder quality regresses.
2. **Default model**: `Xenova/whisper-base` or `whisper-small` for live streaming; offer `whisper-large-v3-turbo` or `distil-large-v3.5` as a premium "Quality" mode (when WebGPU is available and enough VRAM, roughly 1.5 to 2 GB).
3. **Distil-large-v3.5 is the new default candidate** for "best quality on consumer hardware". It is 1.5x faster than Turbo on long-form, has better short-form WER (7.08% OOD), and has ONNX checkpoints already (`distil-whisper/distil-large-v3.5-ONNX`).
4. **Streaming**: implement LocalAgreement-2 per Machacek 2023; ufal/whisper_streaming is the reference. The newer AlignAtt (SimulStreaming) is faster but requires modifying decoder attention, which is harder in the browser ONNX path.
5. **CrisperWhisper is tempting for verbatim accuracy** (clinical/medical) but **CC BY-NC blocks commercial use**. Skip unless a license is acquired.
6. **Fallback for non-WebGPU**: whisper.cpp WASM with `tiny.en` Q5_0 or `base.en` Q5_0 gives 2 to 6x real-time on CPU.
7. **Avoid for the browser**: faster-whisper, WhisperJAX, WhisperKit, MLX-Whisper, insanely-fast-whisper, WhisperLive. All assume server or native platform.
8. **Diarization**: WhisperX path (pyannote) is server-side only currently; browser-side diarization is still an open problem (small experiments exist with ONNX pyannote ports, but immature). Defer.

---

## 11. Source URLs (consolidated)

### Repos
- OpenAI Whisper, https://github.com/openai/whisper
- whisper.cpp, https://github.com/ggml-org/whisper.cpp
- faster-whisper, https://github.com/SYSTRAN/faster-whisper
- whisperX, https://github.com/m-bain/whisperX
- distil-whisper, https://github.com/huggingface/distil-whisper
- whisper-timestamped, https://github.com/linto-ai/whisper-timestamped
- insanely-fast-whisper, https://github.com/Vaibhavs10/insanely-fast-whisper
- whisper_streaming, https://github.com/ufal/whisper_streaming
- SimulStreaming, https://github.com/ufal/SimulStreaming
- Lightning-SimulWhisper, https://github.com/altalt-org/Lightning-SimulWhisper
- WhisperLive, https://github.com/collabora/WhisperLive
- WhisperLiveKit, https://github.com/QuentinFuxa/WhisperLiveKit
- whisper-web, https://github.com/xenova/whisper-web
- transformers.js, https://github.com/huggingface/transformers.js
- WhisperKit / argmax-oss-swift, https://github.com/argmaxinc/argmax-oss-swift
- WhisperJAX, https://github.com/sanchit-gandhi/whisper-jax
- stable-ts, https://github.com/jianfch/stable-ts
- CrisperWhisper, https://github.com/nyrahealth/CrisperWhisper
- MLX-Whisper (examples), https://github.com/ml-explore/mlx-examples/tree/main/whisper
- candle, https://github.com/huggingface/candle
- mac-whisper-speedtest, https://github.com/anvanvan/mac-whisper-speedtest

### HuggingFace
- openai/whisper-*, https://huggingface.co/openai
- openai/whisper-large-v3-turbo, https://huggingface.co/openai/whisper-large-v3-turbo
- ggerganov/whisper.cpp, https://huggingface.co/ggerganov/whisper.cpp
- distil-whisper/*, https://huggingface.co/distil-whisper
- distil-whisper/distil-large-v3.5, https://huggingface.co/distil-whisper/distil-large-v3.5
- distil-whisper/distil-large-v3.5-ONNX, https://huggingface.co/distil-whisper/distil-large-v3.5-ONNX
- distil-whisper/distil-large-v3.5-ggml, https://huggingface.co/distil-whisper/distil-large-v3.5-ggml
- distil-whisper/distil-large-v3.5-ct2, https://huggingface.co/distil-whisper/distil-large-v3.5-ct2
- Xenova/whisper-*, https://huggingface.co/Xenova
- argmaxinc/whisperkit-coreml, https://huggingface.co/argmaxinc/whisperkit-coreml
- nyrahealth/CrisperWhisper, https://huggingface.co/nyrahealth/CrisperWhisper

### Live browser demos
- whisper.cpp WASM, https://whisper.ggerganov.com/
- whisper.cpp stream.wasm, https://ggml.ai/whisper.cpp/stream.wasm/
- Xenova Whisper Web, https://huggingface.co/spaces/Xenova/whisper-web
- Xenova Real-time WebGPU, https://huggingface.co/spaces/Xenova/realtime-whisper-webgpu
- candle-whisper space, https://huggingface.co/spaces/lmz/candle-whisper
- WhisperJAX space, https://huggingface.co/spaces/sanchit-gandhi/whisper-jax

### Papers
- Whisper original, https://cdn.openai.com/papers/whisper.pdf
- Distil-Whisper, https://arxiv.org/abs/2311.00430
- whisper_streaming (LocalAgreement), https://arxiv.org/abs/2307.14743
- CrisperWhisper, https://arxiv.org/abs/2408.16589
- WhisperKit, https://arxiv.org/abs/2507.10860
- SimulStreaming IWSLT 2025, https://arxiv.org/abs/2506.17077
- CarelessWhisper, https://arxiv.org/abs/2508.12301
- Whisper Quantization analysis, https://arxiv.org/abs/2503.09905
- HF speculative decoding blog, https://huggingface.co/blog/whisper-speculative-decoding

### Other references
- OpenAI Whisper announcement, https://openai.com/index/whisper/
- HF Transformers.js v3 announcement, https://huggingface.co/blog/transformersjs-v3
- Modal: Choosing Whisper variants, https://modal.com/blog/choosing-whisper-variants
- Demystifying Whisper Turbo (Amgad Hasan), https://amgadhasan.substack.com/p/demystifying-openais-new-whisper
- VoicePing offline benchmark (16 models), https://voiceping.net/en/blog/research-offline-speech-transcription-benchmark/

---

*Compiled 2026-05-16. Source breadth: 25+ GitHub repos, 10+ arXiv papers, HF model cards, blog posts, and benchmark suites.*
