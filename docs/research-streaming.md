# Streaming Transcription Research

Reference document for algorithms, libraries, and supporting techniques used when turning an offline ASR model (Whisper, Conformer, Zipformer) into a real-time browser transcription pipeline. Scope: algorithms and supporting libraries, not speech models themselves.

The Off The Record project uses **LocalAgreement-2** over Whisper running in transformers.js with WebGPU. This doc covers that algorithm in depth and surveys the broader landscape: alternative commit policies, streaming architectures, VAD, diarization, alignment, hallucination handling, latency metrics, and speculative decoding.

---

## 1. Streaming commit protocols

### 1.1 LocalAgreement-n and LocalAgreement-2

The core idea behind Whisper-Streaming. Originally introduced by **CUNI-KIT at IWSLT 2022** as a streaming policy that can convert any offline sequence-to-sequence model into a simultaneous streaming model without retraining. Adopted by Macháček, Dabre, Bojar (2023, arXiv:2307.14743) for Whisper.

**Mechanism**: every time a new audio chunk arrives, the system re-decodes a growing audio buffer and produces a new full hypothesis. The system **commits only the longest common prefix** between the current hypothesis and the previous hypothesis (n=2). For LocalAgreement-n, the longest common prefix across n consecutive hypotheses is committed. Anything past the agreed prefix is treated as tentative and may change with new audio.

**Why it works**: at chunk boundaries the model is uncertain, but as more audio arrives those boundary tokens stabilize. Agreement across two consecutive decodes is a cheap proxy for "this prefix will not change". n=2 is the standard tradeoff (one extra inference of latency, big reduction in flicker). n=3 commits more conservatively, latency goes up.

**Buffer trimming**: after each commit, audio still in the window is kept. The reference implementation trims the audio buffer at sentence-ending punctuation in the committed portion (language-aware sentence splitter), or by Whisper segment boundaries when the buffer grows beyond a threshold (default 15 s). The trimmed-out audio is gone forever; the model never re-decodes it. The last roughly 200 committed words are passed as `initial_prompt` to keep context across the boundary.

**Reported numbers** (ESIC corpus, English, MinChunkSize=1.0 s, Whisper-large):

- WER 8.1% streaming vs 7.9% offline.
- Average latency 3.62 s; computationally-unaware latency 1.91 s.
- German: WER 9.4% streaming vs 9.2% offline at 4.37 s latency.
- Czech: WER 12.9% streaming vs 12.3% offline at 4.76 s latency.

**Pseudocode** (paraphrased from `whisper_online.py::HypothesisBuffer`):

```text
state: buffer (previous hypothesis tail), commited (final output),
       last_commited_time, last_commited_word

on new audio chunk of size MinChunkSize:
    1. transcribe(audio_buffer, init_prompt=last_200_words_of_commited)
    2. new_words = words from transcription with t > last_commited_time - 0.1
    3. de-duplicate against commited tail by n-gram match (n=1..5)
       (handles cases where the new decode re-emits words already committed)
    4. flush:
         commit = []
         while new_words and buffer and new_words[0].text == buffer[0].text:
             commit.append(new_words.pop(0))
             buffer.pop(0)
         buffer = new_words  # the tail past the agreement point
         commited.extend(commit)
    5. if len(audio_buffer) > buffer_trimming_sec:
         trim audio at last sentence-ending punctuation in commit,
         or at last Whisper segment boundary
    6. emit commit to UI as new "committed" tokens; buffer is "tentative"
```

The `flush()` operation is the actual LocalAgreement-2 step: token-by-token longest common prefix between previous `buffer` and the current `new`. The buffer is the tail of the previous hypothesis past the previous commit point, which is what makes the comparison work.

**Implementations**:

- [`ufal/whisper_streaming`](https://github.com/ufal/whisper_streaming): reference Python implementation. Supports `whisper`, `faster-whisper`, MLX, OpenAI API backends. WebSocket server included (`whisper_online_server.py`).
- [`ufal/SimulStreaming`](https://github.com/ufal/SimulStreaming): successor with newer policies including AlignAtt-based decoding.
- [`collabora/WhisperLive`](https://github.com/collabora/WhisperLive): Python server, WebSocket, Silero VAD frontend.
- [`codesdancing/whisper_streaming_web`](https://github.com/codesdancing/whisper_streaming_web): a web wrapper.
- Off The Record `src/lib/transcription/hypothesisBuffer.ts`: TypeScript port, browser, runs in a Web Worker.

**Browser portability**: pure algorithm, no I/O, trivial to port. The hard part is running Whisper itself in the browser (transformers.js + ONNX Runtime Web + WebGPU). The buffer logic is roughly 100 lines of TypeScript.

### 1.2 AlignAtt and EDAtt

**AlignAtt** (Papi et al., Interspeech 2023, arXiv:2305.11408): a policy that uses **cross-attention** between the decoder output token and the encoder source frames to decide whether to emit the next token now or wait for more audio. If the decoder attends to encoder frames close to the end of the current audio chunk, it implies the model is "looking at the future" and the token should be held back. AlignAtt simplifies the earlier **EDAtt** policy by removing a hyperparameter (no manual attention threshold). It works with offline-trained models without fine-tuning. Reports +2 BLEU and 0.5 to 0.8 s latency reduction over state of the art on MuST-C.

**Simul-Whisper** (Wang et al., Interspeech 2024, arXiv:2406.10052): applies AlignAtt to Whisper. Adds an **integrate-and-fire truncation detector** that catches when a chunk boundary splits a word. Reports only 1.46% absolute WER degradation at 1 s chunk size, beating LocalAgreement-2 in chunk-bounded regimes.

**Browser portability**: requires access to cross-attention weights at decode time. Transformers.js exposes them for `_timestamped` Whisper exports (the same tensors used for DTW word timestamps). Doable, more wiring than LocalAgreement-2.

### 1.3 Hold-n and wait-k

Older simultaneous-translation policies. **Wait-k** waits for k source tokens before emitting any target. **Hold-n** holds the last n tokens of every hypothesis as tentative and only commits the rest. LocalAgreement-n superseded these for streaming ASR because they don't use the cross-decode signal: they just discard the tail unconditionally, which throws away information.

### 1.4 Confidence-based commit (alternative to agreement)

Instead of running two decodes and comparing, commit a token once its **log-probability** is above a threshold. Whisper exposes per-token logprobs; word-level confidence can be computed as `min` or `last-token` of constituent token softmax probabilities. The "Adopting Whisper for Confidence Estimation" paper (arXiv:2502.13446) trains a calibrator on top.

Tradeoffs vs LocalAgreement:

- One inference per chunk instead of two: lower compute.
- Per-token logprob is noisy on word boundaries near chunk edges. The same boundary-instability that LocalAgreement fixes is the exact place where logprob is least reliable.
- Calibration is brittle across languages, accents, noise.

Most production streaming Whisper deployments combine the two: LocalAgreement decides the agreement boundary, logprob/no_speech filters spurious commits.

### 1.5 Attention-based commit

For Whisper specifically, you can also commit at points where the **encoder cross-attention concentrates strongly on one frame** (high entropy collapse). This is the same signal AlignAtt uses, generalized as a commit gate.

---

## 2. Streaming architectures

### 2.1 RNN-Transducer (RNN-T)

The standard streaming architecture before Whisper. An encoder over acoustic features, a prediction network over previous tokens, a joiner that combines them. Native streaming because the encoder can be causal or chunk-causal. Emits one token (or blank) per encoder step. Used in Google's mobile ASR, Amazon Transcribe Live, Apple dictation.

- **CHAT** (Chunk-wise Attention Transducer, arXiv:2602.24245): cross-attention inside each fixed chunk. Strictly monotonic at chunk granularity. Reports up to 6.3% relative WER reduction over plain RNN-T, 46% lower peak training memory.
- **Alignment-restricted RNN-T**: constrains emission to be close to forced alignments during training; reduces emission latency.

### 2.2 Conformer and Zipformer (chunk-causal encoders)

**Conformer** combines convolutions and self-attention. Streaming variants use **chunk-causal masking**: self-attention attends to the current chunk plus a fixed lookback, no lookahead. Latency is bounded by chunk size plus lookahead.

**Zipformer** (k2-fsa, used in `icefall` and `sherpa-onnx`) is a redesigned Conformer with multi-scale downsampling. Streaming Zipformer in sherpa-onnx is the most-deployed open streaming transducer today: ONNX exports, runs on CPU and embedded.

- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx): C++ runtime with bindings for 12 languages including JS via WASM. Browser-portable for small models.
- [Zipformer streaming pretrained models](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html).

### 2.3 U2 and U2++ (WeNet, two-pass)

A **unified streaming and non-streaming** model. Single shared encoder. First pass: CTC prefix beam search runs in streaming mode and emits partial hypotheses. Second pass: attention decoder rescores the top-N CTC hypotheses at end of utterance (or every K seconds) for accuracy. U2++ adds a right-to-left attention decoder and bidirectional rescoring; +10% relative WER over U2. The shared encoder uses limited right context for low streaming latency.

- arXiv:2106.05642 (U2++), arXiv:2102.01547 (WeNet original).
- [wenet-e2e/wenet](https://github.com/wenet-e2e/wenet).

This pattern is useful for browser apps: cheap CTC streaming for live UI, attention rescoring for the final transcript. It mirrors Off The Record's dual-panel live and batch architecture, at a lower level.

### 2.4 MoChA: Monotonic Chunkwise Attention

**Monotonic Chunkwise Attention** (Chiu and Raffel, ICLR 2018, arXiv:1712.05382): adapts soft attention to streaming by emitting a hard monotonic pointer, then computing soft attention over a small chunk to the left of the pointer. Trainable with standard backprop via expectation, decodes in linear time. Multi-head variant (MTH-MoChA) and CTC-synchronous training extensions. Important historically; less used today since RNN-T won the production race.

### 2.5 Chunk attention schemes

Vocabulary used in streaming Conformer and Zipformer literature:

- **Chunk-causal**: attention masked so frame i only sees frames in the current chunk plus all past chunks.
- **Look-ahead**: small fixed future context allowed (e.g. 4 frames). Better accuracy, plus look-ahead times hop ms latency.
- **Lookback (memory)**: bounded number of past chunks attended to; reduces compute on long audio.
- **Dynamic chunk training**: train with random chunk sizes so one model serves multiple latency targets.

### 2.6 CarelessWhisper and WhisperRT

**CarelessWhisper** (arXiv:2508.12301): turns Whisper itself into a causal streaming model by retraining with masked future audio. Avoids the re-decoding overhead of LocalAgreement. Not yet widely adopted; competes with Simul-Whisper.

---

## 3. Voice Activity Detection (VAD)

VAD has two roles in a streaming Whisper pipeline:

1. **Frontend gate**: don't feed silence to Whisper. Whisper hallucinates on silence ("Thanks for watching!", "Please subscribe.") and on low-energy noise.
2. **Endpointing**: declare "speaker stopped" so a UI can finalize an utterance or trigger downstream actions.

### 3.1 Silero VAD (v4, v5)

State of the art for general-purpose VAD. Tiny model (around 1 MB ONNX), high accuracy, multilingual.

- v4: 16 kHz only, 30 ms frames.
- v5: introduced **8 kHz support** alongside 16 kHz, smaller compute, somewhat shorter response latency.
- Outputs a per-frame speech probability in [0, 1]. Recommended threshold 0.5; sticky thresholds (`min_speech_duration_ms`, `min_silence_duration_ms`) to avoid flapping.
- Repo: [snakers4/silero-vad](https://github.com/snakers4/silero-vad). Apache 2.0 license as of late 2024 (after a contested license change).

**Browser**:

- [@ricky0123/vad-web](https://www.npmjs.com/package/@ricky0123/vad) wraps `silero_vad_v5.onnx` via ONNX Runtime Web with an AudioWorklet. Drop-in for browser use. Distributes the ONNX model and the WASM runtime via npm or CDN.
- Sample rate conversion to 16 kHz is done in the worklet.

### 3.2 WebRTC VAD (legacy)

Google's GMM-based VAD shipped in WebRTC. Six log-energy subbands (80 to 250, 250 to 500, 500 to 1000, 1000 to 2000, 2000 to 3000, 3000 to 4000 Hz). Accepts 16-bit mono PCM at 8, 16, 32, or 48 kHz; 10, 20, or 30 ms frames.

Four aggressiveness modes (0=quality, 3=very aggressive). Mode 3 has high false-rejection on quiet speech. Mode 1 is the usual default for transcription.

Pros: tiny, deterministic, no model weights, no GPU. Cons: noticeably worse than Silero on non-clean audio.

Browser: works directly because the WebRTC stack already ships with browsers; or compile [py-webrtcvad](https://github.com/wiseman/py-webrtcvad) C bits to WASM.

### 3.3 TEN VAD

Newer entrant from the TEN Framework. Frame-level, optimized for voice-agent turn detection. Claims:

- Better precision than both WebRTC VAD and Silero VAD.
- Lower memory than Silero.
- Faster speech-to-nonspeech transition (Silero has a multi-hundred ms lag on offset).
- Cross-platform C with WASM build, 16 kHz, 10 or 16 ms hops.

Repo: [TEN-framework/ten-vad](https://github.com/ten-framework/ten-vad), HF model card. Worth evaluating if endpointing latency is critical (interactive agents, push-to-talk UIs).

### 3.4 NVIDIA MarbleNet

End-to-end neural VAD from NeMo. Deep residual 1D time-channel separable convolutions, batch-norm, ReLU, dropout. **Frame-VAD MarbleNet** outputs per-frame speech probability at 20 ms resolution; segment-VAD MarbleNet uses overlapping segments. Frame-VAD uses 8x less GPU memory than segment-VAD.

- Models: `vad_multilingual_marblenet`, `vad_telephony_marblenet`, `Frame_VAD_Multilingual_MarbleNet_v2.0` (around 91.5K parameters).
- Heavier than Silero, intended for server pipelines.

### 3.5 Pyannote VAD

Part of pyannote.audio (segmentation model). Strong on overlap and conversational audio. Heavier than Silero (around 17M parameters segmentation backbone). Not really intended as a standalone frontend VAD; it's the building block of pyannote's diarization pipeline.

### 3.6 Picking one for browser

For Off The Record-style apps:

- **Default**: `@ricky0123/vad-web` (Silero v5). Mature, browser-tested, npm install.
- **If endpointing latency matters and you can compile WASM**: TEN VAD.
- **If you need to ship without ONNX Runtime Web**: WebRTC VAD compiled to WASM, or the browser's native `AudioWorkletProcessor` with an energy plus zero-crossing heuristic for a 10-line fallback.

---

## 4. Speaker diarization

### 4.1 pyannote.audio (3.1, 3.x)

The de-facto open-source diarization toolkit. Pipeline `pyannote/speaker-diarization-3.1`:

1. **Segmentation**: a 5-second window powerset segmentation model produces per-frame speaker labels (up to 3 concurrent speakers via the powerset trick).
2. **Embedding**: each segment is embedded (currently uses an ECAPA-TDNN or WeSpeaker variant, pure PyTorch in 3.1; v2 used SincNet via onnxruntime).
3. **Clustering**: agglomerative hierarchical clustering on cosine distances between embeddings.
4. **Stitching**: speaker labels propagated across windows.

Inputs mono 16 kHz; multi-channel is downmixed. Online streaming variant exists (`diart`, `pyannote-audio` streaming inference) but is research-grade.

- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)

### 4.2 WhisperX diarization integration

[m-bain/whisperX](https://github.com/m-bain/whisperX) chains: faster-whisper ASR, then wav2vec2 forced alignment for word timestamps, then pyannote diarization, then token-to-speaker assignment by timestamp overlap. Word-level timestamps reach roughly plus or minus 50 ms vs Whisper's native plus or minus 500 ms, which is what makes per-word speaker labels actually accurate.

### 4.3 NeMo diarization

NVIDIA's diarization stack:

- **MSDD** (Multi-scale Diarization Decoder): handles overlap by predicting per-frame multi-speaker activity across multiple temporal scales.
- **TitaNet-Large** speaker embeddings (next section).
- **VAD** via MarbleNet.

Production-grade and fast on NVIDIA GPUs; not browser-portable.

### 4.4 EEND (End-to-End Neural Diarization)

Reformulates diarization as frame-level multi-label classification: each frame outputs a fixed-size vector of speaker-present flags. Trained with **Permutation Invariant Training (PIT)** to avoid the speaker-ordering problem.

- Handles overlapping speech natively (unlike clustering pipelines).
- Variants: EEND-EDA (encoder-decoder attractors, scales to unknown number of speakers), AED-EEND, transformer-attractor EEND.
- **LS-EEND** (arXiv:2410.06670, 2024): long-form streaming EEND with online attractor extraction. Real practical streaming option.
- **O-EENC-SD** (MERL 2025): online end-to-end neural clustering for streaming diarization with bounded state.

### 4.5 Reverb Diarization (Rev.ai)

Open-weights diarization from Rev (October 2024). v1 uses pyannote 3.0 architecture, v2 uses WavLM replacing SincNet features. Fine-tuned on around 26 K hours of Rev's labeled data. Non-commercial license; SOTA on conversational benchmarks. [revdotcom/reverb](https://github.com/revdotcom/reverb), arXiv:2410.03930.

### 4.6 Speaker embeddings

The clustering step in any pipeline needs fixed-length speaker vectors.

- **TitaNet-Large** (NVIDIA, NeMo): around 23M parameters. ContextNet backbone, 1D depth-wise separable convolutions, Squeeze-and-Excitation, channel-attention statistics pooling. Outputs 192-dim or 512-dim t-vector. Trained on VoxCeleb plus Fisher plus Switchboard. ONNX export available.
- **CAM++** (modelscope/3D-Speaker, arXiv:2303.00332): D-TDNN backbone with Context-Aware Masking modules at every layer (a lighter form of CAM that masks feature maps with an auxiliary utterance-level embedding from global statistic pooling). Multi-granularity pooling combines global and segment-level context. Faster and more accurate than ECAPA-TDNN per the paper.
- **ECAPA-TDNN** (Desplanques et al., Interspeech 2020, arXiv:2005.07143): Res2Net frame layers, multi-layer feature aggregation, channel-dependent frame attention in the statistics pooling, Squeeze-and-Excitation channel attention. Dominant before CAM++ and WeSpeaker. SpeechBrain ships [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb).
- **WavLM** (Microsoft): used in Reverb v2 as the embedding feature extractor.
- **3D-Speaker** (modelscope): trains CAM++, ERes2Net, ECAPA-TDNN end to end with a unified training recipe.

### 4.7 Browser diarization

No clean browser story today. Options:

- Run diarization server-side; the browser only handles audio capture.
- Wait for the end of the recording, then run a pyannote-style pipeline via ONNX Runtime Web (no maintained ONNX exports of the full pyannote 3.1 pipeline as of 2025).
- For 2-speaker assumed cases, a frame-level Silero-VAD plus simple speaker-change detector on a small embedding model can be hacked into the browser, with poor quality on overlap.

For Off The Record, diarization is a future feature. The pragmatic approach is post-hoc diarization on the batch audio at end-of-recording, not live.

---

## 5. Forced alignment and word timestamps

### 5.1 Whisper DTW word timestamps

Whisper's official `word_timestamps=True` (and the `_timestamped` ONNX exports used by transformers.js):

1. Decode segments normally.
2. Extract **cross-attention** weights from a small subset of decoder heads (chosen during training to be most aligned).
3. Run **Dynamic Time Warping** between the per-token attention curves and audio frames to assign each token a `(start, end)` time.
4. Group tokens into words.

Accuracy: roughly plus or minus 200 ms typical, can be worse around chunk boundaries. The DTW path constraint enforces monotonic alignment but does not guarantee precision.

**LocalAgreement-2 depends on these word timestamps** in the Off The Record implementation: each committed word carries `(start, end, text)` and the audio anchor advances by `end` of the last committed word at a sentence boundary.

### 5.2 whisper-timestamped

[linto-ai/whisper-timestamped](https://github.com/linto-ai/whisper-timestamped): same DTW approach, exposed as a Python library, plus per-word and per-segment confidence scores derived from log probabilities. Useful reference for the algorithm.

### 5.3 Wav2Vec2 CTC forced alignment

Separate model, separate pass. Used by WhisperX. The flow:

1. Transcribe with Whisper (any size).
2. Run a phoneme-CTC Wav2Vec2 model on the same audio.
3. Use CTC alignment (Viterbi over the CTC posteriors with the transcript as constraint) to get per-character or per-phoneme timestamps.
4. Aggregate to words.

Accuracy: plus or minus 50 ms typical. Better than DTW. Cost: an extra acoustic forward pass over the whole audio, plus a language-specific Wav2Vec2 model (around 315 MB for `wav2vec2-large-xlsr`).

- [WhisperX alignment code](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py).
- [torchaudio.pipelines.Wav2Vec2FABundle](https://docs.pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html): batched CTC `forced_align()` op.

### 5.4 Montreal Forced Aligner (MFA)

Kaldi GMM-HMM based, still the standard in speech research. Requires a pronunciation lexicon and an acoustic model per language. Highly accurate but offline, command-line, not browser. Useful for ground truth, not production.

### 5.5 MMS-aligned and CTC forced aligner

[MahmoudAshraf97/ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner): wraps Meta's MMS multilingual CTC model (`mms-300m-1130-forced-aligner`) for one-shot alignment over 1000+ languages. ONNX export available; SRT and WebVTT output. Closest to a drop-in alignment library today.

---

## 6. Hallucination mitigation in Whisper

Whisper hallucinates on:

- Long silence ("Thank you.", "Bye.", subscribe-and-like phrases learned from YouTube subtitles).
- Constant low-energy noise (ventilation, music).
- Repeated speech that triggers loop emission ("the the the the...").

### 6.1 Parameter levers (transformers.js, openai-whisper)

| Param | Typical default | Notes |
|---|---|---|
| `no_speech_threshold` | 0.6 (default), 0.2 for stricter rejection | If the `<\|nospeech\|>` token probability is greater than threshold AND `avg_logprob` is below `logprob_threshold`, treat segment as silence and drop. |
| `logprob_threshold` | -1.0 to -0.5 | Average log probability of tokens. Lower is worse. Below threshold triggers temperature fallback. |
| `compression_ratio_threshold` | 2.4 (default), 1.35 stricter | Ratio of UTF-8 byte length of text to zlib-compressed byte length. Above threshold means text is repetitive ("the the the the..."). Triggers fallback. |
| `temperature` fallback | `(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)` | If output fails compression or logprob check, re-decode with the next temperature. Temperature 0 is greedy; higher temperatures introduce randomness which often breaks the repetition loop. |
| `condition_on_previous_text` | True | False stops bad context from poisoning later segments at the cost of consistency. |

### 6.2 VAD frontend gate

Most effective single mitigation in streaming. Drop any chunk classified as silence before it reaches Whisper. Silero VAD or WebRTC VAD aggressiveness 2 to 3.

### 6.3 Per-segment dropping

Drop committed segments whose:

- compression ratio is greater than 2.4 (looped output),
- average logprob is less than -1.0 (low confidence over the whole segment),
- contain known hallucination phrases (regex match).

The Off The Record `heuristics.ts::isHallucinationWord` does this on individual committed words; LocalAgreement-2 still has it as a final filter after agreement.

### 6.4 Calm-Whisper

arXiv:2505.12969: a finetuning approach that adds head-level dropout to "calm down" the cross-attention heads most responsible for non-speech hallucinations. Useful research direction; needs model retraining so not a drop-in for browser apps.

### 6.5 Whisper hallucination on non-speech audio

Two surveys to know:

- arXiv:2501.11378 "Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio".
- [openai/whisper Discussion #679](https://github.com/openai/whisper/discussions/679): community thread with empirical thresholds.

---

## 7. Endpointing

When is the user done talking? Three families:

### 7.1 Silence-based (classical)

Trigger endpoint when trailing silence is greater than `eos_silence_ms` (typically 300 to 1000 ms). Uses a VAD or energy detector. Simple, predictable. Bad with thinking pauses ("Uh... I think... that's right.").

### 7.2 Prosodic and acoustic

Use prosodic features (pitch, energy, duration) in addition to silence. Pipecat's turn detector takes this approach. Better than pure silence on conversational audio.

### 7.3 Semantic and neural

A small text classifier reads the (partial) transcript and decides if it's a complete-sounding utterance. AssemblyAI's "Universal Streaming" semantic endpointing and Deepgram's `endpointing` parameter combine this with silence. Adds robustness for thinking pauses.

### 7.4 Two-pass endpointing (arXiv:2401.08916)

Train an ASR model to emit an `<EOW>` (end-of-word) token. Use it together with VAD trailing silence to decide endpoint. Lower latency than VAD alone.

### 7.5 Improving endpoint detection in streaming ASR (arXiv:2505.17070)

Joint VAD and ASR with a small VAD head sharing the audio encoder. Same encoder, two outputs. Lower latency than serial pipelines.

---

## 8. Partial-result smoothing in commercial APIs

Patterns you can copy for the live UI.

### 8.1 Deepgram

- `interim_results=true`: stream interim transcripts continuously. `is_final` flags the last interim before the next final.
- `endpointing=N` (ms): silence threshold for "speech finished".
- `utterance_end_ms`: emits `UtteranceEnd` event after N ms of silence.
- `smart_format`: automatic punctuation, capitalization, ITN.
- Clients debounce updates to avoid redrawing the DOM on every frame.

### 8.2 AssemblyAI Universal-Streaming

- Partials are always on. Events: `Begin`, `Turn` (`end_of_turn=true` means final), `Termination`.
- Semantic end-of-turn detection (combines silence and transcript analysis).
- Word-level confidence scores in finals.

### 8.3 Google Speech-to-Text

- `interim_results=true` in `StreamingRecognitionConfig`.
- `enable_automatic_punctuation`.
- `single_utterance=true` ends the stream when the user pauses.
- Continuous interim stream of growing hypotheses; finals are emitted on punctuation or silence boundary.

### 8.4 UI smoothing patterns

- Render committed text in normal style, tentative text in a softer color or italic (Off The Record uses grey plus black).
- Throttle DOM updates to around 10 Hz; aggregate within frame.
- Use a stable key per word or token so React can reuse DOM nodes; replacing text instead of recreating nodes is critical for performance.
- Always render *something* within 200 ms of audio start (even a flickering wrong word) to give the user feedback. Empty waits look broken.

---

## 9. Punctuation, capitalization, normalization

### 9.1 Whisper's built-in punctuation

Whisper emits punctuation and casing natively (trained on web data with both). For most applications no post-processing is needed. Note: this only works with `task=transcribe` and the right language, and only if `condition_on_previous_text=True` so the model has context to know where sentence boundaries should fall.

### 9.2 BERT-based punctuation restoration

Used when ASR strips punctuation:

- [felflare/bert-restore-punctuation](https://huggingface.co/felflare/bert-restore-punctuation): finetuned BERT, predicts {., ,, ?, !} per token plus capitalization label.
- PunctFormer, BertPunc: similar pattern.
- arXiv:2101.07343 "Automatic punctuation restoration with BERT models".
- arXiv:1908.02404 "Fast and Accurate Capitalization and Punctuation... with Transformer and Chunk Merging": chunked decoding for streaming.

### 9.3 Whisper Normalizer

Distance and WER computation reference text normalizer that ships with Whisper. Two classes:

- `BasicTextNormalizer`: language-agnostic. Lowercase, strip punctuation, normalize whitespace, remove bracketed expressions.
- `EnglishTextNormalizer`: English-specific. Expands contractions, removes filler words (hmm, uh, um, mhm), handles numbers and currency.

[whisper-normalizer on PyPI](https://pypi.org/project/whisper-normalizer/). Use for WER computation only; never display normalized text to a user.

### 9.4 NeMo Text Normalization and Inverse Text Normalization

- **TN**: written to spoken ("$5" becomes "five dollars"). For TTS.
- **ITN**: spoken to written ("two thousand twenty four" becomes "2024"). For ASR post-processing.
- WFST grammars (no model), context-aware WFSTs with neural LMs, audio-based TN, fully neural TN and ITN. arXiv:2104.05055.
- [NVIDIA/NeMo-text-processing](https://github.com/NVIDIA/NeMo-text-processing).

Whisper's training included a lot of ITN-style examples, so for English it's usually unnecessary. For other languages it's worth applying NeMo ITN as a post-step.

---

## 10. Latency metrics

Mostly from the SimulMT literature, applicable to streaming ASR.

### 10.1 AL (Average Lagging)

Mean lag in audio time per emitted word: how many source frames behind real-time you are when emitting each target word. Cherry and Foster, 2019. Standard metric.

Problem: underestimates lag for systems that over-generate (emit too many words). Off-by-length-ratio.

### 10.2 LAAL (Length-Adaptive Average Lagging)

Papi et al., AutoSimTrans 2022 (arXiv:2206.05807): divides AL by the max of source and target length so over-generation doesn't get a free pass. Now the recommended replacement for AL.

### 10.3 Computation-aware (CA) latency

Add real wall-clock decoding time to the audio-time lag. Often labeled `AL_CA` or `LAAL_CA`. Critical for slow models. Whisper-large at 1x real-time on CPU has AL of 1 s but AL_CA of 3 s.

Recent work: arXiv:2410.16011 "CA*: Addressing Evaluation Pitfalls in Computation-Aware Latency" shows AL_CA grows linearly with input length for inefficient systems, so reporting AL_CA on a single utterance length is misleading.

### 10.4 T-AL, AL-CA, ATD

- **T-AL** (Token Average Lag): per-token latency.
- **ATD** (Average Token Delay, arXiv:2311.14353): duration-aware, accounts for token-emission spacing.
- **EOS lag**: time between end of speech and emission of last token. Specific to streaming ASR.

### 10.5 What to report for streaming Whisper

For Off The Record-style apps:

- Time-to-first-token after speech onset.
- Per-word commit latency (audio time of word end minus wall clock when committed).
- Final-emission latency (end of speech minus last commit).
- Inference duty cycle: wall-clock inference time over audio duration.

---

## 11. Speculative decoding for ASR

### 11.1 Distil-Whisper as draft model

Standard speculative decoding pattern from LLMs, applied to Whisper:

1. Run **Distil-Whisper** (fewer decoder layers, same encoder) as a fast draft model.
2. Verify k draft tokens with a single forward pass of **Whisper-large-v3** (target model).
3. Accept the longest matching prefix of accepted draft tokens. Resample the first mismatch from the target's distribution.
4. Repeat.

Math: outputs are exactly identical to greedy Whisper-large-v3. Up to roughly 2x faster on a GPU with enough memory to hold both models.

- [distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3): explicit drop-in assistant.
- [distil-whisper/distil-large-v3.5](https://huggingface.co/distil-whisper/distil-large-v3.5): 1.5x faster than Whisper-Large-v3-Turbo on long-form. ONNX export available.
- [huggingface/distil-whisper](https://github.com/huggingface/distil-whisper).

### 11.2 Browser portability

Transformers.js supports `assistant_model` in `pipeline()` but loading two Whisper checkpoints in the browser is memory-hostile (Whisper-large-v3 is around 3 GB even quantized, distil-large-v3 is around 750 MB). For mobile, not viable. For desktop WebGPU with 4 GB plus VRAM, doable. Off The Record currently picks one model per panel rather than chaining; speculative decoding is a future direction once browsers commonly have more than 8 GB VRAM allocations.

### 11.3 Encoder reuse

Distil-Whisper keeps the encoder identical to the teacher. That means encoder embeddings can be computed once and shared between draft and target. Cuts compute roughly in half for the encoder-bound part of inference.

---

## 12. Practical recommendations for Off The Record

Based on the survey, the current architecture is well-positioned. Notes for future work:

- **Stay on LocalAgreement-2**. Adding AlignAtt would require wiring cross-attention into the consumer worker. Worth doing for shorter-chunk regimes (sub-1-s) but not currently a pain point.
- **Add Silero VAD frontend**. `@ricky0123/vad-web` drops in cleanly in the producer worklet. Eliminates a class of hallucinations and reduces inference load.
- **Don't switch to confidence-based commit**. The LocalAgreement signal is more robust than logprob thresholding for browser Whisper.
- **Word timestamp accuracy is the floor**. DTW-based word timestamps on transformers.js `_timestamped` models are good enough; CTC alignment via Wav2Vec2 in the browser is achievable but adds around 300 MB of model weights and an extra pass.
- **For diarization, defer to end-of-recording**. Run a post-hoc pipeline on the audio archive, not live.
- **For latency metrics, log per-commit and per-emission latencies** so the dual-panel comparison can quantify the live vs batch tradeoff.
- **Hallucination heuristics already cover the main vectors**. The non-silent-but-garbage case is hard; deferring (the current behavior) is the right call.

---

## Source URLs

### LocalAgreement and Whisper streaming

- [arXiv:2307.14743 "Turning Whisper into Real-Time Transcription System"](https://arxiv.org/abs/2307.14743)
- [arXiv:2307.14743 HTML](https://arxiv.org/html/2307.14743)
- [ufal/whisper_streaming](https://github.com/ufal/whisper_streaming)
- [ufal/SimulStreaming](https://github.com/ufal/SimulStreaming)
- [collabora/WhisperLive](https://github.com/collabora/WhisperLive)
- [collabora/WhisperLive vad.py](https://github.com/collabora/WhisperLive/blob/main/whisper_live/vad.py)
- [Collabora "Transforming speech technology with WhisperLive" blog](https://www.collabora.com/news-and-blog/blog/2024/05/28/transforming-speech-technology-with-whisperlive/)
- [codesdancing/whisper_streaming_web](https://github.com/codesdancing/whisper_streaming_web)

### AlignAtt, Simul-Whisper, CarelessWhisper

- [arXiv:2305.11408 AlignAtt](https://arxiv.org/abs/2305.11408)
- [Interspeech 2023 AlignAtt](https://www.isca-archive.org/interspeech_2023/papi23_interspeech.html)
- [arXiv:2406.10052 Simul-Whisper](https://arxiv.org/abs/2406.10052)
- [arXiv:2508.12301 CarelessWhisper, WhisperRT](https://arxiv.org/html/2508.12301v2)
- [arXiv:2506.12154 "Adapting Whisper for Streaming Speech Recognition via Two-Pass Decoding"](https://arxiv.org/html/2506.12154v1)

### Streaming architectures

- [arXiv:2102.01547 WeNet](https://arxiv.org/pdf/2102.01547v2)
- [arXiv:2203.15455 WeNet 2.0](https://ar5iv.labs.arxiv.org/html/2203.15455)
- [arXiv:2012.05481 U2: Unified Streaming and Non-streaming Two-pass](https://arxiv.org/pdf/2012.05481)
- [wenet-e2e/wenet](https://github.com/wenet-e2e/wenet)
- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [Sherpa Zipformer streaming docs](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html)
- [arXiv:2602.24245 CHAT Chunk-wise Attention Transducer](https://arxiv.org/abs/2602.24245)
- [arXiv:2010.11395 Streaming Transformer Transducer](https://arxiv.org/pdf/2010.11395)
- [arXiv:1712.05382 Monotonic Chunkwise Attention](https://arxiv.org/abs/1712.05382)
- [arXiv:2005.00205 Multi-head Monotonic Chunkwise Attention](https://arxiv.org/abs/2005.00205)

### VAD

- [snakers4/silero-vad](https://github.com/snakers4/silero-vad)
- [@ricky0123/vad npm](https://www.npmjs.com/package/@ricky0123/vad)
- [ricky0123/vad GitHub](https://github.com/ricky0123/vad)
- [Silero VAD browser discussion](https://github.com/snakers4/silero-vad/discussions/534)
- [wiseman/py-webrtcvad](https://github.com/wiseman/py-webrtcvad)
- [TEN-framework/ten-vad](https://github.com/ten-framework/ten-vad)
- [TEN VAD docs](https://theten.ai/docs/ten_vad)
- [NVIDIA Frame VAD Multilingual MarbleNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_multilingual_frame_marblenet)
- [NeMo VAD tutorial notebook](https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/Voice_Activity_Detection.ipynb)

### Diarization

- [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [revdotcom/reverb](https://github.com/revdotcom/reverb)
- [arXiv:2410.03930 Reverb](https://arxiv.org/abs/2410.03930)
- [m-bain/whisperX](https://github.com/m-bain/whisperX)
- [WhisperX DeepWiki forced alignment](https://deepwiki.com/m-bain/whisperX/3.3-forced-alignment-system)
- [arXiv:2410.06670 LS-EEND](https://arxiv.org/html/2410.06670v1)
- [NVIDIA TitaNet large model card](https://huggingface.co/nvidia/speakerverification_en_titanet_large)
- [arXiv:2303.00332 CAM++](https://arxiv.org/abs/2303.00332)
- [modelscope/3D-Speaker](https://github.com/modelscope/3D-Speaker)
- [arXiv:2005.07143 ECAPA-TDNN](https://arxiv.org/abs/2005.07143)
- [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)

### Forced alignment and word timestamps

- [linto-ai/whisper-timestamped](https://github.com/linto-ai/whisper-timestamped)
- [Whisper word timestamps DeepWiki](https://deepwiki.com/openai/whisper/3.5-word-timestamps)
- [MahmoudAshraf97/ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner)
- [Torchaudio forced alignment tutorial](https://docs.pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html)
- [Montreal Forced Aligner overview](https://www.emergentmind.com/topics/montreal-forced-aligner)
- [whisperX alignment.py](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py)

### Hallucinations and confidence

- [openai/whisper Discussion #679 hallucinations](https://github.com/openai/whisper/discussions/679)
- [openai/whisper Discussion #284 confidence scores](https://github.com/openai/whisper/discussions/284)
- [openai/whisper Discussion #2420 compression-ratio threshold](https://github.com/openai/whisper/discussions/2420)
- [arXiv:2502.13446 "Adopting Whisper for Confidence Estimation"](https://arxiv.org/html/2502.13446v1)
- [arXiv:2505.12969 Calm-Whisper](https://arxiv.org/html/2505.12969v1)
- [arXiv:2501.11378 Whisper ASR hallucinations from non-speech audio](https://arxiv.org/html/2501.11378v1)
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)

### Endpointing

- [arXiv:2401.08916 Two-pass endpoint detection](https://arxiv.org/html/2401.08916v1)
- [arXiv:2505.17070 Improving endpoint detection in streaming ASR](https://arxiv.org/html/2505.17070v1)
- [AssemblyAI turn detection blog](https://www.assemblyai.com/blog/turn-detection-endpointing-voice-agent)
- [Deepgram endpointing docs](https://developers.deepgram.com/docs/understand-endpointing-interim-results)
- [Deepgram interim results docs](https://developers.deepgram.com/docs/interim-results)
- [Deepgram end-of-speech detection](https://developers.deepgram.com/docs/understanding-end-of-speech-detection)

### Punctuation and normalization

- [arXiv:2101.07343 Automatic punctuation restoration with BERT](https://arxiv.org/abs/2101.07343)
- [arXiv:1908.02404 Fast capitalization and punctuation with Transformer](https://arxiv.org/abs/1908.02404)
- [felflare/bert-restore-punctuation](https://huggingface.co/felflare/bert-restore-punctuation)
- [whisper-normalizer PyPI](https://pypi.org/project/whisper-normalizer/)
- [WhisperNormalizer English module docs](https://kurianbenoy.github.io/whisper_normalizer/english.html)
- [NVIDIA NeMo Text Normalization docs](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/text_normalization/wfst/wfst_text_normalization.html)
- [arXiv:2104.05055 NeMo Inverse Text Normalization](https://arxiv.org/abs/2104.05055)
- [NVIDIA/NeMo-text-processing](https://github.com/NVIDIA/NeMo-text-processing)

### Latency metrics

- [arXiv:2206.05807 LAAL Length-Adaptive Average Lagging](https://arxiv.org/abs/2206.05807)
- [arXiv:2410.16011 CA* computation-aware latency](https://arxiv.org/html/2410.16011)
- [arXiv:2311.14353 Average Token Delay](https://arxiv.org/html/2311.14353)
- [arXiv:2509.17349 Better Late Than Never: latency metric evaluation](https://arxiv.org/html/2509.17349v1)

### Speculative decoding and Distil-Whisper

- [huggingface/distil-whisper](https://github.com/huggingface/distil-whisper)
- [distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3)
- [distil-whisper/distil-large-v3.5](https://huggingface.co/distil-whisper/distil-large-v3.5)
- [distil-whisper/distil-large-v3.5-ONNX](https://huggingface.co/distil-whisper/distil-large-v3.5-ONNX)

### Browser inference stack

- [ONNX Runtime Web docs](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
- [ONNX Runtime Web WebGPU announcement](https://opensource.microsoft.com/blog/2024/02/29/onnx-runtime-web-unleashes-generative-ai-in-the-browser-using-webgpu/)
- [Transformers.js](https://github.com/xenova/transformers.js)
