# State of the Art in Offline Speech to Text Transcription

> Synthesised narrative for an engineer building a browser side, offline transcription app.
> Last updated 2026-05-16. Read `AGENT_INSTRUCTIONS.md` for how this doc was produced.
> Working notes with full citations live in `research-whisper.md`, `research-chinese.md`, `research-western.md`, `research-browser.md`, `research-streaming.md`, `research-papers.md`. Flat repo catalogue: `repositories.md`. Annotated bibliography: `papers.md`.

## TL;DR for the impatient

For an offline transcription web app like Off The Record:

1. **Default stack today**: `transformers.js v3` plus `ONNX Runtime Web` plus `WebGPU`. No serious competitor matches its model breadth and JS ergonomics in 2026.
2. **Default model (English or multilingual)**: `whisper-large-v3-turbo` with `{ encoder: "fp32", decoder: "q4" }`. Best browser sweet spot today. Total around 500 to 800 MB.
3. **Lighter live model**: `distil-large-v3.5` (multilingual, 1.5x faster than turbo on long form, OOD WER 7.08%) or `Moonshine v2 base` (61M params, MIT, English only, around 13x faster than Whisper small).
4. **Default streaming algorithm**: `LocalAgreement-2` (Machacek et al. 2023). Already used by Off The Record. Watch `Simul-Whisper` plus `AlignAtt` as the natural successor when sub second latency matters.
5. **Default VAD**: `Silero VAD v5` via `@ricky0123/vad-web`. Around 1 MB ONNX, sub millisecond per frame.
6. **For Chinese, Cantonese, Japanese, Korean coverage**: `sherpa-onnx WASM` plus `SenseVoice Small` or `Streaming Zipformer`. The only fully tested offline CJK path in a browser.
7. **Current open accuracy ceiling for English (offline, not in a browser)**: `NVIDIA Canary-Qwen-2.5B` at 5.63% mean WER on the Open ASR Leaderboard, `IBM Granite-Speech-3.3-8B` at 5.74%, `Parakeet-TDT-0.6B-v2` at around 6.05%. All CC-BY-4.0 or Apache-2.0. Not browser viable yet.
8. **Mind the licenses**: many top tier open models are CC-BY-NC (Meta MMS, Seamless, Rev Reverb, CrisperWhisper). Read licenses before shipping.

## 1. The eras of modern ASR

ASR has gone through five stylistic eras since the deep learning turn:

1. **Hybrid HMM + GMM, then HMM + DNN**, roughly 2010 to 2015. Kaldi is the canonical toolkit. Still alive in `Vosk` and old enterprise deployments. Mediocre accuracy by modern standards. Tiny WASM footprint, hence still shipped at edge.
2. **End to end CTC and LAS**, 2015 to 2019. Deep Speech 2 (Baidu) and Listen-Attend-Spell (Google) prove that a single neural network can replace the entire hand engineered pipeline.
3. **Conformer plus RNN-T**, 2019 to 2022. Combining convolutions and self attention (Conformer) plus a transducer head gives mobile friendly, low latency streaming ASR. This is what powered Google Pixel keyboard, Amazon Transcribe Live, Apple Dictation.
4. **Whisper era**, 2022 to 2024. Encoder decoder Transformer trained on 680k hours of weakly supervised audio. Whisper rewrites public expectations: multilingual, multitask, robust to noise, zero shot on most benchmarks. Everyone in the open world ports it: `whisper.cpp`, `faster-whisper`, `WhisperX`, `Distil-Whisper`, `WhisperKit`, `transformers.js`.
5. **Post Whisper, 2024 to 2026**. Three parallel directions:
   - **Bigger and more accurate**: `NVIDIA Canary-Qwen-2.5B`, `IBM Granite-Speech-3.3-8B`, `Phi-4-Multimodal`, `Mistral Voxtral`. Audio aware LLMs eating ASR.
   - **Smaller and faster**: `Moonshine v2`, `Distil-Whisper v3.5`, `Whisper Turbo`, `WhisperKit`. Edge first.
   - **Truly streaming**: `Kyutai STT`, `SeamlessStreaming`, `Simul-Whisper`, `CarelessWhisper`, `Voxtral-Mini-4B-Realtime`. End to end models built for real time from the start.

The bet Off The Record makes is on era 4 (Whisper) with era 5 streaming algorithms (LocalAgreement-2) bolted on. That bet still pays off in May 2026 because the era 5 models that are smaller and faster than Whisper (Moonshine) are not yet better than Whisper Turbo on quality, and the era 5 models that are better than Whisper (Canary, Granite) are too large for a browser. The sweet spot for in browser deployment remains a quantised Whisper Turbo variant.

## 2. Architectures that matter

### 2.1 Encoder decoder Transformer (Whisper, Voxtral, Phi-4)

- Whisper style. Encoder consumes 30 second log mel spectrograms (80 bins for v1 and v2, 128 bins for v3), decoder autoregressively emits tokens with special tokens for language, task, and timestamps.
- Pros: simple to train at scale, no alignment supervision needed, zero shot across languages and tasks.
- Cons: bidirectional encoder is not natively streaming. Fixed 30 second windows. Hallucinates on silence.
- All Whisper derivatives inherit these tradeoffs. `large-v3-turbo` prunes decoder from 32 to 4 layers, giving 2x to 5x speedup with comparable WER on most languages.

### 2.2 Non autoregressive (Paraformer, SenseVoice)

- Predict target length first (CIF, continuous integrate and fire), decode all tokens in parallel, optionally refine with a second pass.
- 10x to 15x faster inference than autoregressive seq2seq at comparable WER on Mandarin.
- `Paraformer-large` (Alibaba DAMO) is the canonical example. `SenseVoice Small` (Alibaba FunAudioLLM) extends the pattern to multilingual ASR plus emotion plus audio event detection in a single forward pass.

### 2.3 RNN-T, TDT, U2++ (streaming first)

- **RNN-Transducer**: encoder over audio, prediction network over previous tokens, joiner combines them, monotonic alignment. Native streaming.
- **TDT** (Token and Duration Transducer, NVIDIA, 2023): generalises RNN-T by predicting both a token and a frame skip duration per step. Up to 2.82x faster than RNN-T at equal or better WER. Used in `Parakeet-TDT`.
- **U2 and U2++** (WeNet, 2021): single Conformer encoder, two heads. CTC streaming for live UI, attention decoder rescoring for the final transcript. Mirrors Off The Record's dual panel architecture at the model level.

### 2.4 FastConformer, Zipformer (modern encoders)

- **FastConformer** (NVIDIA, 2023): 8x stride subsampling plus depthwise separable convolutions plus simplified blocks. Around 2x faster than Conformer. Substrate for every NeMo ASR model.
- **Zipformer** (k2-fsa / Daniel Povey, ICLR 2024): U-Net like multi rate encoder, BiasNorm replaces LayerNorm, SwooshR/L activations, ScaledAdam optimiser. SOTA on LibriSpeech and AISHELL-1 at smaller parameter counts than Conformer.

### 2.5 Audio LLMs (Voxtral, Phi-4-Multimodal, Qwen2-Audio, Step-Audio 2, Kimi-Audio)

- Audio encoder bolted onto a chat LLM. Transcription is one task; the model also summarises, answers questions, follows instructions about the audio.
- Strong on the Open ASR Leaderboard: `Canary-Qwen-2.5B` (5.63% WER), `Granite-Speech-3.3-8B` (5.74%), `Phi-4-Multimodal` (6.02%). All beat Whisper-large-v3.
- Browser viability: poor. These are 2B to 24B parameter models. Server side only today.

### 2.6 Delayed Streams Modeling (Kyutai)

- Decoder only LM operating on time aligned audio plus text streams with fixed inter stream delays. Delaying text relative to audio gives ASR; delaying audio relative to text gives TTS.
- `Kyutai STT-1B-en_fr` runs at 0.5 second delay; `Kyutai STT-2.6B-en` at 2.5 second delay, both CC-BY-4.0.
- One H100 runs 400 concurrent real time streams of the 1B model. No ONNX export yet, so not browser deployable.

## 3. The current English accuracy leaderboard

Snapshot from the Hugging Face Open ASR Leaderboard, May 2026. Mean WER averaged across LibriSpeech (clean and other), AMI, Earnings-22, GigaSpeech, SPGISpeech, TED-LIUM, VoxPopuli.

| Rank | Model | Params | Mean WER | RTFx | License | Streams? | Browser? |
|---|---|---|---|---|---|---|---|
| 1 | NVIDIA Canary-Qwen-2.5B | 2.5B | 5.63 | 418 | CC-BY-4.0 | No | No |
| 2 | IBM Granite-Speech-3.3-8B | 8.4B | 5.74 | 145 | Apache-2.0 | No | No |
| 3 | Microsoft Phi-4-Multimodal | 5.6B | 6.02 | n/a | MIT | No | No |
| 4 | NVIDIA Parakeet-TDT-0.6B-v2 | 0.6B | 6.05 | 3,380 | CC-BY-4.0 | Yes (chunked) | Via ONNX |
| 5 | NVIDIA Parakeet-TDT-0.6B-v3 | 0.6B | 6.34 | 749 | CC-BY-4.0 | Yes (chunked) | Via ONNX |
| 6 | NVIDIA Canary-1B-Flash | 0.9B | 6.35 | 1,045 | CC-BY-4.0 | No | No |
| 6 | NVIDIA Parakeet-CTC-1.1B | 1.1B | 6.68 | 2,793 (top) | CC-BY-4.0 | Yes | Via ONNX |
| 7 | Kyutai STT-2.6B-en | 2.6B | 6.4 (streaming) | 88 | CC-BY-4.0 | Yes (2.5 s) | Not yet |
| 8 | Mistral Voxtral-Mini-3B | 3B | 7.05 | 109 | Apache-2.0 | Realtime variant | No |
| 9 | OpenAI Whisper-large-v3-turbo | 0.8B | 7.83 | 200 | MIT | No (bolt on) | Yes |
| 10 | OpenAI Whisper-large-v3 | 1.55B | 7.4 | 68 | MIT | No | Yes (heavy) |
| 11 | Distil-Whisper large-v3.5 | 0.76B | 7.08 (OOD) | 300 | MIT | No | Yes |
| 12 | UsefulSensors Moonshine v2 base | 61M | 9.99 | 566 | MIT | Yes (ergodic) | Yes (native) |
| 13 | Meta MMS-1B-all | 1B | 22.54 | 231 | CC-BY-NC-4.0 | No | Possible |
| 14 | Vosk small-en-us | ~50M | ~15 | streaming | Apache-2.0 | Yes | Yes (WASM) |

Observations:

- The top 3 are all "audio plus LLM" hybrids. They drag along a 2B+ language model.
- The top **browser viable** model on raw accuracy remains `Distil-Whisper-large-v3.5` (7.08% OOD short form WER, MIT, ONNX checkpoints already published).
- The top **explicit browser model** is `Moonshine v2 base`, which trades around 3 points of WER for around 5x speed and a 13x smaller download.
- `Whisper-large-v3-turbo` sits in the middle of both axes. It is the practical default.

## 4. The current Mandarin accuracy leaderboard

Aggregated CER on AISHELL-1 and AISHELL-2:

| Model | Params | AISHELL-1 | AISHELL-2 | WenetSpeech meeting | License |
|---|---|---|---|---|---|
| Qwen3-ASR-LLM | 1.7B | 0.57 (AED) / 0.64 | 2.15 | 4.32 | Apache-2.0 |
| Seed-ASR 2.0 (closed, ByteDance) | >12B | 1.52 | 2.77 | 4.74 | proprietary |
| FireRedASR-LLM | 8.3B | ~1.4 | ~2.5 | ~4.6 | Apache-2.0 |
| FireRedASR-AED | 1.1B | ~1.6 | ~2.7 | ~5.0 | Apache-2.0 |
| Paraformer-large | 220M | 1.95 | 2.85 | 6.97 | FunASR custom |
| Streaming Zipformer (zh-en) | ~70M | ~4.5 | ~5.0 | n/a | Apache-2.0 |
| WeNet U2++ Conformer | ~120M | ~4.6 | ~5.5 | ~16 | Apache-2.0 |
| SenseVoice Small | ~250M | beats Whisper-large | beats Whisper-large | n/a | model-license |

Browser viable subset, narrowest to widest CJK coverage:

- `Streaming Zipformer (zh-en)`, Apache-2.0, around 30 to 60 MB INT8, browser ready via `sherpa-onnx` WASM. Latency winner.
- `Paraformer-large`, custom license, around 120 MB INT8 via `sherpa-onnx`. Accuracy winner among small models.
- `SenseVoice Small`, custom license, around 120 MB INT8 via `sherpa-onnx`. Multilingual winner: Mandarin, Cantonese, Japanese, Korean, English, plus emotion plus audio event tags in a single pass.
- `Qwen3-ASR` (Apache-2.0): top accuracy at 1.7B params, but no quantised browser build exists yet.

## 5. Streaming algorithms

### 5.1 LocalAgreement-2

The single most important algorithm for "Whisper as a live transcriber". Off The Record's live panel implements it.

- Re run Whisper on a growing audio buffer.
- Compare the new hypothesis against the previous one.
- Commit the longest common prefix. Anything past the agreement point is tentative and may change.
- Trim the audio buffer at sentence boundaries in the committed text. Keep `CONTEXT_LOOKBACK_S` of already transcribed audio to avoid losing acoustic context.

Why it works: token uncertainty concentrates at chunk boundaries; an extra chunk of audio stabilises the boundary; two consecutive agreements are a cheap proxy for "this prefix will not change".

Reference: Machacek, Dabre, Bojar 2023, "Turning Whisper into Real-Time Transcription System", https://arxiv.org/abs/2307.14743. Reference implementation: `ufal/whisper_streaming`. Off The Record's `src/lib/transcription/hypothesisBuffer.ts` is the TypeScript port.

Reported numbers, Whisper-large, ESIC English: WER 8.1% streaming vs 7.9% offline, 3.62 s end to end latency, 1.91 s computationally unaware latency.

### 5.2 AlignAtt and Simul-Whisper

A more aggressive policy. Uses Whisper's cross attention argmax to decide per token whether to emit now or wait for more audio. If the model is still attending to the most recent encoder frames, hold the token. Simul-Whisper applies AlignAtt plus an integrate and fire truncation detector for split words. Only 1.46% absolute WER degradation at 1 second chunk size, beating LocalAgreement-2 at short chunks.

Reference: Papi et al. 2023 (https://arxiv.org/abs/2305.11408); Wang et al. 2024 (https://arxiv.org/abs/2406.10052).

When to switch to AlignAtt over LocalAgreement-2: when sub second commit latency is critical. The wiring is more involved because the policy needs cross attention weights at decode time. Transformers.js exposes them for `_timestamped` exports.

### 5.3 SimulStreaming (2025)

Successor to whisper_streaming from the same authors. Uses AlignAtt instead of LocalAgreement, around 5x faster. Winning entry of IWSLT 2025 Simultaneous Speech Translation Shared Task. Paper: https://arxiv.org/abs/2506.17077.

### 5.4 CarelessWhisper

Retrains Whisper with causal attention via LoRA, eliminating the need for full 30 second windows for low latency. Avoids the re-decoding overhead of LocalAgreement entirely. Paper: https://arxiv.org/abs/2508.12301. Promising research direction; not yet shipping in a real toolkit.

### 5.5 Two pass and U2++

The U2++ pattern (one model, two latency modes) is the architectural blueprint for Off The Record's UI (live LocalAgreement panel plus batch one shot panel). It is normally done with a single CTC + attention model; Off The Record does the analogous thing by running two Whisper passes with different strategies.

### 5.6 Confidence based commit

An alternative to agreement: commit a token once its log probability exceeds a threshold. Cheaper (one inference per chunk instead of two) but noisier near chunk boundaries: the exact place where LocalAgreement provides the most value. Production deployments tend to combine the two.

### 5.7 Speculative decoding

Pair a small Distil-Whisper draft model with full Whisper-large-v3. The draft proposes k tokens, the target verifies in one forward pass, accept the longest matching prefix. Mathematically guaranteed to produce identical output to the target. Around 2x faster on a GPU with enough memory for both models.

Browser viability: holding both models in WebGPU is feasible on desktop with 8 GB plus VRAM but pinched on consumer hardware. Off The Record currently picks one model per panel. Watch this once browsers commonly expose larger GPU allocations.

## 6. Voice Activity Detection

VAD has two roles in a streaming Whisper pipeline: gate Whisper away from silence (which Whisper hallucinates on), and decide when an utterance ends.

| VAD | Size | Latency | Browser | Notes |
|---|---|---|---|---|
| **Silero VAD v5** | ~1 MB ONNX | sub-ms / 30 ms frame | `@ricky0123/vad-web` | Default. Apache-2.0. Multilingual. 8 kHz and 16 kHz. |
| **WebRTC VAD** | ~50 KB | sub-ms | native in browsers | Legacy. GMM based. Noticeably worse than Silero on noisy audio. |
| **TEN VAD** | small | sub-ms, faster offset | WASM build | Lower offset latency than Silero. Worth evaluating for endpointing critical UIs. |
| **NVIDIA Frame-VAD MarbleNet** | ~91k params | ms scale | not browser | Server pipelines. |
| **pyannote VAD** | 17M params | server scale | not browser | Building block of pyannote diarization. |

For Off The Record specifically, `@ricky0123/vad-web` is the obvious drop in. It already ships Silero v5 ONNX, an AudioWorklet, and ONNX Runtime Web wiring.

## 7. Diarization

Open source diarization is server side today. There is no clean browser story.

- **pyannote.audio 3.1**: the de facto open toolkit. Powerset segmentation plus speaker embedding plus agglomerative clustering. Handles up to 3 concurrent speakers per window.
- **WhisperX diarization**: faster-whisper plus wav2vec2 forced alignment plus pyannote plus token assignment. Word level speaker labels accurate to around 50 ms. Server side.
- **Reverb Diarization (Rev.ai)**: WavLM based embeddings, fine tuned on 26k hours. Strong on conversational audio. **Non commercial license.**
- **NeMo MSDD plus TitaNet**: NVIDIA stack. Production grade and fast on NVIDIA GPUs.
- **EEND** (End-to-End Neural Diarization): research line that frames diarization as multi label frame classification. Handles overlapping speech natively. `LS-EEND` (2024) is a streaming variant.

Practical recommendation for a browser app: run diarization post hoc at end of recording, not live. Use the `audioArchive` table as input. There is no maintained ONNX export of the full pyannote 3.1 pipeline today.

## 8. Forced alignment and word timestamps

Two main paths for per word timestamps:

1. **Whisper DTW on cross attention** (Jong Wook Kim's method). Decode normally, extract cross attention from a small subset of heads chosen during training to be most aligned, run Dynamic Time Warping to assign each token a `(start, end)` time. This is what `transformers.js` does for `_timestamped` ONNX exports. Accuracy around plus or minus 200 ms, worse near chunk boundaries. **This is what LocalAgreement-2 in Off The Record depends on.**
2. **Wav2Vec2 CTC forced alignment** (WhisperX). Separate model, separate pass. Run a phoneme CTC model on the same audio, Viterbi alignment over CTC posteriors constrained by the Whisper transcript. Accuracy around plus or minus 50 ms. Cost: an extra acoustic forward pass plus a language specific Wav2Vec2 model (around 315 MB for `wav2vec2-large-xlsr`).

For Off The Record's browser context, DTW from cross attention is the only currently viable path. Wav2Vec2 alignment would add 300 MB to the model download and another forward pass to the live loop.

A third option: `MahmoudAshraf97/ctc-forced-aligner` wraps Meta's MMS multilingual CTC model for one shot alignment across 1000+ languages. ONNX export available, SRT and WebVTT output. Worth evaluating if multilingual alignment becomes critical.

## 9. Hallucination mitigation

Whisper hallucinates on:

- Long silence ("Thank you.", "Bye.", "Please subscribe."). Common in YouTube training data.
- Constant low energy noise (ventilation, music).
- Repeated speech that triggers loop emission ("the the the the").

Levers:

- **VAD front gate** is the single most effective mitigation. Drop any chunk classified as silence before it reaches Whisper.
- **`no_speech_threshold` and `logprob_threshold`**: reject segments whose `<|nospeech|>` token probability exceeds threshold and whose average log prob is below threshold.
- **`compression_ratio_threshold`** (default 2.4): if the UTF-8 byte length of text divided by its zlib compressed length exceeds threshold, the segment is repetitive and triggers temperature fallback.
- **Temperature fallback**: if a segment fails the compression or logprob check, re-decode with progressively higher temperatures. Breaks repetition loops.
- **`condition_on_previous_text=False`** stops bad context from poisoning later segments at the cost of consistency.
- **Per word filtering**: Off The Record's `heuristics.ts::isHallucinationWord` filters known hallucination tokens at the word level after LocalAgreement-2.
- **Calm-Whisper** (https://arxiv.org/abs/2505.12969): a research direction that adds head level dropout to cross attention heads most responsible for non speech hallucinations. Needs retraining.

Off The Record's existing design (silence gate before Whisper; hallucination deferral on non silent audio) already covers most of the practical hallucination surface.

## 10. Browser runtime stack

### 10.1 Runtime layer

- **transformers.js v3** (Hugging Face): the JS port. v3 added WebGPU. v3.7+ ships 120+ architectures, 1200+ ONNX models. Whisper specific API: `pipeline("automatic-speech-recognition", model, { device, dtype })`, per module `dtype` for encoder and decoder, `return_timestamps: "word"` for word level timing.
- **ONNX Runtime Web** is the underlying runtime. Two flavors: `onnxruntime-web` (WASM EP) and `onnxruntime-web/webgpu` (WebGPU EP via JSEP). Has known issues with fp16 / q4f16 producing NaN on some encoder and decoder models in 2026; safest path is fp32 encoder plus integer q4 decoder.
- **WebLLM** and **MLC** are LLM focused, not ASR.
- **whisper.cpp WASM** is a solid CPU only fallback. Live demos at https://ggml.ai/whisper.cpp/stream.wasm/.
- **sherpa-onnx WASM** is the right runtime if you want Zipformer, Paraformer, SenseVoice, or TeleSpeech in the browser. CPU WASM only, no WebGPU.

### 10.2 Backend support matrix (May 2026)

| Feature | Chrome | Edge | Safari | Firefox |
|---|---|---|---|---|
| WebGPU desktop | 113+ | 113+ | 26 (Tahoe) | 141+ Win, 145+ macOS ARM, Linux behind flag |
| WebGPU mobile | Android 12+ | n/a | iOS 26 | flag |
| WebGPU shader-f16 | most GPUs | yes | Apple Silicon | yes |
| WASM SIMD | yes | yes | yes | yes |
| WASM threads | yes (needs COI) | yes | 15+ | yes |
| SharedArrayBuffer | yes (needs COI) | yes | yes | yes |
| AudioWorklet | 66+ | 79+ | 14.1+ | 76+ |
| OPFS | 86+ | 86+ | 15.2+ | 111+ |
| Cache API | yes | yes | yes | yes |
| WebNN | flag | flag | no | no |

### 10.3 Counter intuitive performance note

**WebGPU is not unconditionally faster than WASM for Whisper.** On Apple Silicon, WASM beats WebGPU by 1.5x to 5x on `whisper-base` (encoder fp32 plus decoder q4) per transformers.js issue #894. The encoder is small enough that GPU dispatch overhead dominates. WebGPU wins decisively on discrete NVIDIA and on models at least the size of `whisper-small`.

Strategy: detect at runtime, pick per device. The published Off The Record pattern is `device: webgpu` when WebGPU exists and the machine is not Apple Silicon; otherwise fall back to WASM with `numThreads = min(hardwareConcurrency - 1, 8)`.

### 10.4 Quantization recipe for Whisper Turbo in the browser

```ts
const transcriber = await pipeline(
  "automatic-speech-recognition",
  "onnx-community/whisper-large-v3-turbo",
  {
    device: "webgpu",
    dtype: {
      encoder_model: "fp32",          // q4 encoder degrades WER noticeably
      decoder_model_merged: "q4",     // safe and about 4x smaller than fp32
    },
  },
);
```

Why: q4 quantization of the encoder destroys mel filterbank fidelity. The decoder is far more tolerant. The Turbo architecture's 4 layer decoder amplifies this: there is little decoder weight to begin with, q4 saves around 4x size with negligible WER loss.

### 10.5 Audio capture

- `getUserMedia` plus `MediaStreamAudioSourceNode` plus `AudioWorkletNode` is the only correct real time pipeline. `ScriptProcessorNode` is deprecated. `MediaRecorder` is for record then transcribe, not real time.
- `AudioContext.sampleRate` is **48 kHz on almost every modern device** and immutable once any node is created. Whisper needs 16 kHz mono Float32. Resample in the AudioWorklet. Linear interpolation is adequate for Whisper mel filterbanks; `libsamplerate-js` is available if higher fidelity is needed.
- Inter thread transfer: `postMessage(buf, [buf])` with the buffer in the transferable list moves ownership without copy. For very tight loops, lock free ring buffer in SharedArrayBuffer.

### 10.6 Storage

- **Model weights**: Cache API or OPFS. **Not IndexedDB.** Quotas in 2026 are generous: Chrome up to 60% of disk per origin, Firefox the smaller of 10% disk and 10 GiB. Persistent storage needs `navigator.storage.persist()`.
- **Audio chunks and transcript**: IndexedDB via Dexie. Off The Record's existing schema is the right shape.

### 10.7 Cross origin isolation

- Required for `SharedArrayBuffer` and threaded WASM (around 2x speedup over single thread).
- Set `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` (or `credentialless`).
- Consequence: third party embeds (ads, social widgets, fonts without CORP) fail. Strategy is usually to host the transcription view on a subdomain and keep the marketing site un isolated.

### 10.8 Common pitfalls

1. **WebGPU memory leak in transformers.js** (issue #860). GPU memory grows monotonically over repeated transcriptions. Mitigation: hold a single Singleton pipeline, call `model.dispose()` between long sessions, listen for `device.lost`, reload periodically.
2. **fp16 and q4f16 on WebGPU produce NaN on some models** (ORT issues #26732, #26367). Use fp32 encoder plus integer q4 decoder.
3. **AudioContext defaults to 48 kHz even when you ask for 16 kHz**, especially on Safari and Firefox. Always resample defensively.
4. **`postMessage` with multi MB Float32Array is a perf cliff in Firefox** (bug 1754400). Transfer the buffer or use SharedArrayBuffer.
5. **AudioWorklet `process()` runs on the audio thread**: never block it. Buffer and transfer to a Worker.
6. **`navigator.storage.persist()` is required** to avoid eviction of multi hundred MB model caches.
7. **Service Workers cannot use WebGPU or threaded WASM**. Host the model in a dedicated Worker, use the Service Worker only for caching model bytes.
8. **Long form word timestamps in transformers.js have known bugs** (issues #1357, #1358). Force `chunk_length_s` to the model training window and avoid stride overlap if you need word timing.

## 11. Proposed algorithms and architectures for offline browser ASR

Synthesising the survey, here is a menu of designs ordered from "what Off The Record already does" to "research directions to track".

### 11.1 Current Off The Record architecture (recommended default)

- Whisper Turbo (or distil-large-v3.5) via transformers.js plus WebGPU.
- AudioWorklet capture, resample to 16 kHz, 1 second chunks.
- Dual table IndexedDB: `chunks` (evictable) plus `audioArchive` (kept for batch).
- Three workers: producer (capture), live consumer (LocalAgreement-2), batch (one shot on Stop).
- Decoder dtype q4, encoder dtype fp32 (fp16 only for Turbo on WebGPU).
- Conservative anchor advancement: trim only at sentence boundaries or at `FAST_TRIM_THRESHOLD_S`. Always keep `CONTEXT_LOOKBACK_S` of context.
- Silence gate before Whisper; hallucination deferral on non silent garbage.

This is the production grade pattern. Keep it.

### 11.2 Drop in upgrades

- **Add Silero VAD frontend** via `@ricky0123/vad-web`. Around 1 MB extra. Cuts a class of hallucinations and reduces compute by gating Whisper away from silence.
- **Switch live model to `distil-large-v3.5`** when WebGPU is available and the user accepts a one time around 500 MB download. 1.5x faster than Turbo on long form, OOD short form WER 7.08% beats Turbo's 7.83%.
- **Switch live model to `Moonshine v2 base`** for the lowest latency English only path. Around 120 MB download, 13x faster than Whisper small on Apple M3, native transformers.js support.
- **Add a Singleton pipeline plus `model.dispose()` cycle** to defeat the WebGPU memory leak in long sessions.
- **Add WebGPU vs WASM detection at runtime**. WASM beats WebGPU for Whisper base on Apple Silicon.

### 11.3 Streaming algorithm upgrades

- **Hold the line on LocalAgreement-2** for now. The signal is robust; the dual panel side by side comparison gives the user a visible accuracy benchmark.
- **AlignAtt and Simul-Whisper** are the natural next step when sub second commit latency is needed. Requires routing cross attention into the consumer worker. Already possible because `_timestamped` exports expose those tensors.
- **CarelessWhisper** if a once off LoRA finetune to causal Whisper becomes practical. Eliminates the re-decoding cost of LocalAgreement entirely.
- **Speculative decoding** when 8 GB plus WebGPU memory budgets are common. Distil-large-v3 as draft plus Whisper-large-v3 as target gives 2x speedup with identical output.

### 11.4 Multilingual upgrades

- **CJK and broad multilingual**: ship a second engine, `sherpa-onnx` WASM, alongside transformers.js. Default Mandarin / Cantonese / Japanese / Korean / English model: `SenseVoice Small`. Fallback streaming model: `Streaming Zipformer (zh-en)`. Both via `sherpa-onnx` npm package.
- **Korean specific**: `k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16`, Apache-2.0.
- **Japanese specific**: `kotoba-whisper v2.0`, Apache-2.0, 6.3x faster than Whisper-large-v3 with comparable Japanese WER. Or rely on SenseVoice's Japanese.
- **Watch `Qwen3-ASR` (Apache-2.0, 1.7B, 52 languages)** for a Whisper replacement once a quantised browser build appears.

### 11.5 Future track items

- **Audio LLMs in the browser**: `Phi-4-Multimodal` and `Voxtral-Mini-3B` are MIT and Apache-2.0 respectively. Not yet small enough to ship in a tab, but the 3B class is on the horizon as WebGPU memory budgets grow.
- **Kyutai STT browser port**: `kyutai/stt-1b-en_fr` is genuinely streaming (0.5 s delay) and CC-BY-4.0. No ONNX export exists yet. If one appears, this becomes a serious LocalAgreement-2 competitor.
- **WebNN as a third backend**: still flag gated in 2026. Once it ships, NPU offload (Apple Neural Engine, Qualcomm Hexagon, Intel NPU) becomes a real path. Revisit late 2026.
- **Diarization in the browser**: today, defer to end of recording. Watch for an ONNX export of pyannote 3.1 or a maintained `diart` port.
- **Forced alignment in the browser** via MMS `ctc-forced-aligner`. Adds around 300 MB but gives 50 ms word timestamps, much better than DTW.

## 12. Licensing field guide

License is a deployment blocker, not an academic detail. Categorised by what is shippable:

**Apache-2.0 and MIT (safe for commercial deployment)**:

- OpenAI Whisper (MIT), all derivatives that inherit Whisper's license.
- transformers.js, whisper.cpp, faster-whisper, distil-whisper, whisper_streaming, WhisperLive, sherpa-onnx, k2, icefall, WeNet, PaddleSpeech, kotoba-whisper, FireRedASR.
- NVIDIA Canary and Parakeet (CC-BY-4.0, commercial allowed with attribution).
- IBM Granite-Speech (Apache-2.0).
- Microsoft Phi-4-Multimodal (MIT), WavLM (MIT), SpeechT5 (MIT).
- Mistral Voxtral (Apache-2.0).
- Kyutai Moshi and STT (CC-BY-4.0).
- UsefulSensors Moonshine (MIT).
- Step-Audio 2 mini (Apache-2.0).
- Qwen2-Audio, Qwen2.5-Omni, Qwen3-Omni, Qwen3-ASR (Apache-2.0).
- GLM-4-Voice code (Apache-2.0; weights have an additional model license that is generally permissive).
- Silero VAD (Apache-2.0).
- Vosk and Kaldi (Apache-2.0).

**Custom restrictive (read carefully before shipping)**:

- FunASR Model License: Paraformer-zh, SenseVoice, CAM++, ERes2Net. Attribution required. Commercial use is not explicitly forbidden but legal review is advised.
- Tongyi Qianwen License (older Qwen): research only without explicit commercial application.
- Moonshot Kimi-Audio: custom, generally permissive, verify HF card.

**Non commercial (blocker for paid products)**:

- Meta MMS: CC-BY-NC-4.0.
- Meta SeamlessM4T-v2, SeamlessExpressive, SeamlessStreaming: CC-BY-NC-4.0 plus custom.
- Rev Reverb-ASR and Reverb-Diarization: custom non commercial. Contact `licensing@rev.com` for commercial use.
- CrisperWhisper: CC-BY-NC-4.0. Tempting for verbatim accuracy in medical contexts; needs a separate license.
- TeleSpeech-ASR (China Telecom): Tele-AI agreement, commercial use requires written approval.

**Closed (reference only, cannot ship)**:

- AssemblyAI Universal-2, Deepgram Nova-3, Speechmatics Ursa 2, Google USM, Google Gemini, iFlyTek Spark, Tencent Hunyuan-ASR, ByteDance Seed-ASR. Useful as accuracy ceilings on benchmarks.

## 13. Recommendation matrix

| Use case | Browser viable today | Primary stack | Notes |
|---|---|---|---|
| English live transcription, lowest latency | yes | transformers.js + Moonshine v2 base + LocalAgreement-2 (or skip LA-2 since Moonshine is already short window) | MIT, 120 MB, around 13x Whisper small speed |
| English live transcription, best accuracy | yes | transformers.js + distil-large-v3.5 q4 + LocalAgreement-2 | MIT, ~500 MB, 7.08% OOD short form WER |
| English live transcription, default | yes | transformers.js + whisper-large-v3-turbo q4 + LocalAgreement-2 | Current Off The Record default. MIT. |
| Multilingual, English plus EU | yes | transformers.js + whisper-large-v3-turbo q4 + LocalAgreement-2 | Same as default. 99 langs but quality varies. |
| Multilingual including Mandarin, Cantonese, Japanese, Korean | yes | sherpa-onnx WASM + SenseVoice Small | License audit needed. Single forward pass for transcript + LID + emotion + event. |
| Mandarin only, low latency | yes | sherpa-onnx WASM + Streaming Zipformer (zh-en) | Apache-2.0, ~30-60 MB INT8. |
| Mandarin only, best accuracy | server only | Qwen3-ASR-1.7B | Apache-2.0, no quantised browser build yet. |
| English live, beat Whisper accuracy | server only | NVIDIA Parakeet-TDT-0.6B-v2 via ONNX | CC-BY-4.0, ~6% mean WER, 3380 RTFx on Open ASR LB. |
| English best in class accuracy | server only | NVIDIA Canary-Qwen-2.5B or IBM Granite-Speech-3.3-8B | CC-BY-4.0 / Apache-2.0, 5.63% / 5.74% mean WER. |
| Multilingual server side, best WER | server only | NVIDIA Canary-1B-v2 or Parakeet-TDT-0.6B-v3 | CC-BY-4.0, 25 EU langs, Granary trained. |
| Verbatim medical transcription | server only | CrisperWhisper | **CC-BY-NC-4.0 license blocker for commercial use.** |
| Forced alignment, browser | yes (heavy) | transformers.js + MMS ctc-forced-aligner ONNX | 1000+ languages, 50 ms timestamps, ~300 MB. |
| VAD, browser | yes | `@ricky0123/vad-web` (Silero v5) | Apache-2.0, ~1 MB. Default everywhere. |
| Diarization | server only today | pyannote 3.1 or WhisperX or Reverb-Diarization | No clean browser story. Defer to end of recording for browser apps. |

## 14. Open questions and things to verify before shipping

1. **SenseVoice model license commercial clarification.** Read the actual MODEL_LICENSE text in `modelscope/FunASR`. Legal review before shipping.
2. **Quantised SenseVoice in WASM.** Verify int8 ONNX loads and runs at less than 1x RTF on a mid range laptop and on iOS Safari (which currently lacks WASM SIMD threads in some configurations).
3. **WebGPU EP for ONNX-runtime-web with sherpa-onnx.** Not turn key. Investigate WebGPU operator coverage for the Zipformer transducer graph.
4. **TeleSpeech commercial path.** If dialect coverage matters strategically, negotiate with Tele-AI; otherwise drop.
5. **Qwen3-ASR ONNX.** As of May 2026 not exported. Track `QwenLM/Qwen3-ASR`; could replace SenseVoice as the default multilingual ASR if and when a quantised browser runnable build appears.
6. **Streaming Paraformer in the browser.** `paraformer-zh-streaming` exists but is less well trodden than Streaming Zipformer in sherpa-onnx. Benchmark before choosing.
7. **Decoder KV cache reuse across LocalAgreement-2 passes** is not exposed through transformers.js today. The single biggest performance lever still on the table for streaming Whisper in a browser. Track upstream.
8. **Whisper-large-v3 fp32 in the browser** requires ONNX external data format and WebAssembly memory bigger than 2 GB. Workable but the easier path is to ship Turbo or distil-large-v3.5.
9. **WebGPU memory leak in transformers.js** (issue #860). Implement a Singleton plus dispose cycle. Reload page every N transcriptions for long lived sessions.

## 15. Changelog

- 2026-05-16. Initial synthesis. Six parallel research agents, 3,463 lines of working notes, 200+ cited sources.
