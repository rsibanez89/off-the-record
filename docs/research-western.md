# Western non-Whisper offline speech-to-text: research notes

Scope: state-of-the-art ASR systems from Western labs and vendors, excluding OpenAI Whisper and Chinese model families (covered by sibling research notes). Compiled May 2026. Aimed at informing a browser-based offline transcription app, so deployment friction (license, ONNX/WebGPU availability, model size) is weighted heavily.

## Executive summary

- The **Open ASR Leaderboard** is the canonical benchmark. As of late 2025 / early 2026, the English-track top performers (lowest WER) are LLM-decoder hybrids: **NVIDIA Canary-Qwen-2.5B (5.63%)**, **IBM Granite-Speech-3.3-8B (~5.74%)**, **Microsoft Phi-4-Multimodal-Instruct (~6.02%)**, then **NVIDIA Parakeet-TDT-0.6B-v2/v3** (~6.05% to 6.34%). For **throughput**, NVIDIA Parakeet-CTC-1.1B leads at RTFx around 2,793, orders of magnitude above Whisper-large-v3 (~68).
- **Best for browser/edge today**: **UsefulSensors Moonshine v2** (MIT license, 27M / 61M params, runs via ONNX Runtime Web + Transformers.js with WebGPU). It is explicitly the only model in this set designed for in-browser real-time use.
- **Best truly streaming open model**: **Kyutai STT** (CC-BY-4.0, 1B/2.6B params, 0.5 s / 2.5 s delay). Closest competitor to Whisper for live caption use cases.
- **Best multilingual open accuracy + speed**: **NVIDIA Canary-1B-v2 / Parakeet-TDT-0.6B-v3**, both CC-BY-4.0, 25 European languages, trained on NVIDIA's new **Granary** dataset.
- **Best LLM-style audio understanding (not just transcription)**: **Mistral Voxtral Mini-3B / Small-24B** (Apache-2.0). Beats Whisper-large-v3 on transcription and adds QA/summarization.
- **Commercial reference points** (closed, but useful for ceiling comparison): AssemblyAI Universal-2, Deepgram Nova-3, Speechmatics Ursa 2.
- **Legacy / still-shipped offline**: Vosk/Kaldi (small, fast, mediocre accuracy, runs as WASM in browsers today), PicoVoice Leopard/Cheetah (commercial WASM).
- **Deprecated / archived**: Mozilla DeepSpeech, Coqui STT.

---

## NVIDIA NeMo ecosystem (Parakeet + Canary)

NVIDIA is dominant in open ASR right now. All models live under the NeMo framework and ship from `huggingface.co/nvidia`. Architecture is **FastConformer encoder** (aggressive subsampling + depthwise-separable convs, around 2x faster than vanilla Conformer) with one of: **CTC**, **RNN-T**, **TDT** (Token-and-Duration Transducer), or a Transformer decoder (Canary).

### TDT, what it is

Token-and-Duration Transducer generalizes RNN-T by predicting a **token** *and* a **duration** (how many frames to skip) at each step. Up to 4-frame skips give roughly 2.8x inference speedup vs. RNN-T with equal or better WER. Reference: NVIDIA dev blog "Turbocharge ASR Accuracy and Speed with Parakeet-TDT".

### Parakeet family

| Model | Params | Decoder | Languages | License | Notes |
|---|---|---|---|---|---|
| `parakeet-ctc-0.6b` | 0.6B | CTC | English | CC-BY-4.0 | Fastest, lower WER ceiling |
| `parakeet-ctc-1.1b` | 1.1B | CTC | English | CC-BY-4.0 | RTFx around 2,793, top throughput on leaderboard |
| `parakeet-rnnt-0.6b` | 0.6B | RNN-T | English | CC-BY-4.0 | |
| `parakeet-rnnt-1.1b` | 1.1B | RNN-T | English | CC-BY-4.0 | Replaced by TDT for inference speed |
| `parakeet-tdt-1.1b` | 1.1B | TDT | English | CC-BY-4.0 | Topped leaderboard in early 2024 |
| `parakeet-tdt-0.6b-v2` | 0.6B | TDT | English | CC-BY-4.0 | Mean WER around 6.05%, RTFx around 3,380 (highest English) |
| `parakeet-tdt-0.6b-v3` | 0.6B | TDT | 25 EU languages | CC-BY-4.0 | Released Aug 2025, Granary-trained, mean WER 6.34%, RTFx 749 |
| `parakeet-tdt_ctc-1.1b` | 1.1B | TDT+CTC | English | CC-BY-4.0 | Hybrid head |
| `parakeet-tdt_ctc-0.6b-ja` | 0.6B | TDT+CTC | Japanese | CC-BY-4.0 | |

Streaming: NeMo ships a chunked-streaming inference script (`speech_to_text_streaming_infer_rnnt.py`). Chunk size + left/right context are tunable. Word-, segment-, and char-level timestamps are output. ONNX export is supported via NeMo's `export()` API; community ports exist (`FluidInference/parakeet-tdt-0.6b-v3-ov` for OpenVINO, `NexaAI/parakeet-tdt-0.6b-v3-npu` / `-ane` for NPU / Apple Neural Engine).

### Canary family

Encoder-decoder (FastConformer encoder + Transformer decoder). Task tokens drive ASR vs AST and language selection. Supports punctuation, capitalization, and word/segment timestamps via the integrated **NeMo Forced Aligner (NFA)**.

| Model | Params | Languages | License | Mean WER (Open ASR) | RTFx |
|---|---|---|---|---|---|
| `canary-1b` | ~1B | en/de/fr/es ASR + AST | CC-BY-4.0 | ~6.5% | ~200 |
| `canary-1b-flash` | 883M (32-enc / 4-dec) | en/de/fr/es ASR + AST | CC-BY-4.0 | 6.35% | 1,045 (A100), 1,669 (H100) |
| `canary-180m-flash` | 182M (17-enc / 4-dec) | en/de/fr/es | CC-BY-4.0 | not on Open ASR | 1,200+ |
| `canary-1b-v2` | 978M (32-enc / 8-dec) | 25 EU languages + AST | CC-BY-4.0 | Fleurs 8.40% / MLS 7.27% / LS-clean 2.18% | 749 |
| `canary-qwen-2.5b` | 2.5B (FastConformer + Qwen3-1.7B decoder) | English | CC-BY-4.0 | **5.63%, leaderboard #1 (Jul 2025)** | 418 |

Canary-1B-Flash hits **1.48% WER on LibriSpeech test-clean** and BLEU 32 to 41 on FLEURS En to {De, Es, Fr}. Audio length capped at 40 s per inference call; long-form requires the chunked-inference helper script. **No streaming** (fixed-length encoder-decoder).

### NeMo Forced Aligner (NFA)

Token-, word-, and segment-level timestamps via Viterbi decoding over a CTC head's log-probabilities. Bundled with Canary and used by Parakeet for word-level timestamps. Optional ground-truth text alignment mode. Useful as a separate utility if you already have a transcript and want timing.

### Granary dataset

Open dataset (~1M hours: 650k ASR + 350k AST) covering 25 European languages including low-resource (Maltese, Estonian, Croatian). Released August 2025. Underpins Canary-1b-v2 and Parakeet-TDT-0.6b-v3. Critical for any future EU-language work; first time NVIDIA released the training data, not just the weights.

### FastConformer architecture

Drop-in successor to Conformer. Aggressive 8x subsampling (vs Conformer's 4x) + depthwise-separable conv + simplified block layout. Roughly 2x faster encoder. Standard across all current NeMo ASR releases. Paper: NeMo docs + arXiv 2305.05084 (FastConformer).

---

## Meta open-speech releases

### MMS, Massively Multilingual Speech

- **Repo**: `facebook/mms-1b-all`, `mms-1b-l1107`, `mms-1b-fl102`, plus base `mms-300m`
- **License**: **CC-BY-NC-4.0 (non-commercial)**. Disqualifier for many commercial products.
- **Architecture**: Wav2Vec2 backbone + language-specific CTC adapters
- **Languages**: **1,162** for ASR (1,107 fine-tuned + 55 dev); 1,406 pretrained
- **Sample rate**: 16 kHz
- **Mean Open ASR WER**: 22.54% (much weaker on standard English benchmarks than dedicated models, e.g. LibriSpeech clean around 12.6%); designed for breadth, not accuracy on resource-rich languages
- **RTFx**: around 230
- **Paper**: Pratap et al., "Scaling Speech Technology to 1,000+ Languages", arXiv 2305.13516

**Take**: If you need *any* support for African / Indigenous / low-resource languages, MMS is the only open game in town. For English/EU accuracy it is dominated by Parakeet/Canary.

### Seamless suite

- **SeamlessM4T-v2** (`facebook/seamless-m4t-v2-large`): speech-to-speech, speech-to-text, text-to-speech, text-to-text across ~100 input / ~35 output languages. **License: CC-BY-NC-4.0** (non-commercial).
- **SeamlessStreaming**: streaming variant of M4T, same non-commercial license.
- **SeamlessExpressive**: emotion- and style-preserving translation, custom Seamless license (more restrictive than CC-BY-NC).
- **GitHub**: `facebookresearch/seamless_communication`

Strong for translation; for pure ASR, Canary/Parakeet eat its lunch on benchmark accuracy and inference speed, and the licence rules it out for commercial transcription apps anyway.

### Voicebox / Audiobox

TTS / generative audio, not ASR. **Voicebox is closed** (research only). **Audiobox is research-only** (CC-BY-NC, restricted from IL/TX users). Out of scope for this project but mentioned because the brief asked.

---

## Mistral Voxtral

July 2025 release (not 2024 as the brief states; the dated checkpoints are `2507`). Mistral's first audio model, an audio-aware LLM built on top of their Ministral / Mistral Small text models.

| Model | Params | License | Languages | Audio length | RTFx | Notes |
|---|---|---|---|---|---|---|
| `Voxtral-Mini-3B-2507` | 3B | Apache-2.0 | 8 (en/es/fr/pt/hi/de/nl/it) | 30 min (ASR), 40 min (QA) | 109 | Built on Ministral-3B |
| `Voxtral-Small-24B-2507` | 24B | Apache-2.0 | 8 | 30/40 min | not benched | Built on Mistral-Small |
| `Voxtral-Mini-4B-Realtime-2602` | 4B | Apache-2.0 | 8 | streaming, 240 ms to 2.4 s configurable | not benched | Causal audio encoder, streaming-native |

WER benchmarks (Voxtral-Mini-3B):

- LibriSpeech clean: **1.88%**
- LibriSpeech other: 4.1%
- SPGISpeech: 2.37%
- GigaSpeech: 10.24%
- Earnings22: 10.69%
- AMI: 16.3%
- Open ASR mean: around 7.05%

Voxtral-Small beats Whisper-large-v3, GPT-4o-mini-Transcribe, and Gemini 2.5 Flash on English short-form + Common Voice. Beyond transcription it handles QA, summarization, multi-turn audio dialogue, and function calling from voice. Useful if you want to skip a downstream LLM step.

**Browser viability**: poor at 3B to 24B params + 9.5 GB VRAM (bf16). Designed for vLLM / GPU serving, not edge. Could matter if your "offline" target includes a beefy laptop with WebGPU + 8 GB VRAM and you're willing to quantize aggressively, but Moonshine is a more obvious fit for browsers.

- **HF**: https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
- **Paper**: arXiv 2507.13264
- **Mistral blog**: https://mistral.ai/news/voxtral

---

## Kyutai (Moshi + STT)

Kyutai is a Paris-based non-profit lab; all releases are **CC-BY-4.0**. Both lines share the **Mimi** neural audio codec (12.5 Hz frame rate, 32 audio tokens/frame, 24 kHz mono) and the **Delayed Streams Modeling (DSM)** framework.

### Delayed Streams Modeling (DSM)

Decoder-only LM operating on time-aligned audio + text streams with fixed inter-stream delays. Delaying text relative to audio gives ASR; delaying audio relative to text gives TTS. Pre-processing handles alignment so inference is straight autoregressive next-token prediction. Paper: arXiv 2509.08753 ("Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling").

### Kyutai STT

| Model | Params | Languages | Delay | License | Open ASR mean WER |
|---|---|---|---|---|---|
| `kyutai/stt-1b-en_fr` | ~1B | en + fr | **0.5 s** | CC-BY-4.0 | not reported |
| `kyutai/stt-2.6b-en` | ~2.6B | en | 2.5 s | CC-BY-4.0 | **6.4%** (LS-clean 1.7%, LS-other 4.32%) |

- One H100 can run **400 concurrent real-time streams** of the 1B model.
- Word-level timestamps, capitalization, punctuation.
- Trained on 2.5M hours pretraining + 24k hours fine-tune (English) / Fisher + proprietary (en/fr).
- Native HF Transformers support (`>=4.53.0`).
- 1B model has semantic VAD; 2.6B does not.
- **No ONNX export** documented yet. Main runtimes are `moshi` Python lib, MLX (Apple Silicon), and Rust/Candle.

### Moshi

First real-time full-duplex spoken LLM. Mimi-codec encoder + 7B-scale decoder. Around 200 ms practical latency. Models speaker turns implicitly. CC-BY-4.0. Quantized MLX (q4/q8) and Rust/Candle ports exist. **Moshiko** = male voice, **Moshika** = female. More a voice agent than a transcription tool, but the same codec/stack powers Kyutai STT, so it is the technical foundation.

- **Repo**: https://github.com/kyutai-labs/moshi, https://github.com/kyutai-labs/delayed-streams-modeling
- **Paper**: arXiv 2410.00037 (Moshi), arXiv 2509.08753 (DSM)

---

## UsefulSensors Moonshine: most relevant model for this project

The only family in this list designed from the ground up for **edge / browser** real-time use. Moonshine v1 (October 2024) and v2 (early 2026) ship via ONNX + Transformers.js + WebGPU.

### Moonshine v1

- **Architecture**: Encoder-decoder Transformer, RoPE position embeddings, no zero-padding on variable-length inputs.
- **Tiny**: 27M params. **Base**: 61M params.
- **License**: **MIT**.
- **Languages**: English only (v1 base). LibriSpeech clean WER 3.38%, other 8.15%, Open ASR mean 9.99% (base). Matches or beats Whisper tiny.en / base.en.
- **RTFx**: 565.
- **Compute**: 5x less than Whisper-tiny on a 10 s clip.
- **Paper**: arXiv 2410.15608 ("Moonshine: Speech Recognition for Live Transcription and Voice Commands").

### Moonshine v2, "Ergodic Streaming Encoder"

- Standard Transformer stack but **sliding-window self-attention** in the encoder (no full attention, no positional encodings; the encoder is translation-invariant in time, i.e. "ergodic").
- Attention configuration: (16 left, 4 right) tokens for first/last two layers, (16, 0) for middle layers; purely causal in the middle, mildly bidirectional at boundaries.
- Standardized to 50 Hz features (matches Whisper) for easier comparison.
- 13.1x faster than Whisper-small on Apple M3 (148 ms latency).
- Paper: arXiv 2602.12241.

### Browser deployment

- **Transformers.js** has native Moonshine support (PR #1099). Use the standard `pipeline('automatic-speech-recognition', 'onnx-community/moonshine-...')` API.
- WebGPU + WASM fallback via **ONNX Runtime Web**.
- "Moonshine Web" demo (Xenova): around 150 MB download with WebGPU, around 120 MB with WASM. Fully offline after initial load.
- Source: https://github.com/huggingface/transformers.js-examples/tree/main/moonshine-web

### Flavors of Moonshine

Late 2025 release of **task-specialized tiny variants** for specific edge use cases (intent detection, command recognition). arXiv 2509.02523.

- **Repo**: https://github.com/moonshine-ai/moonshine (formerly `usefulsensors/moonshine`)
- **HF org**: https://huggingface.co/UsefulSensors, mirror in `onnx-community/`

**Bottom line for off-the-record app**: Moonshine v2 base via Transformers.js + WebGPU is the closest thing to a drop-in offline Whisper replacement that actually runs full speed in the browser. English-only is the constraint.

---

## Rev Reverb (Reverb-ASR + Reverb-Diarization)

Rev open-sourced its commercial in-house models in October 2024.

- **Repo**: https://github.com/revdotcom/reverb, https://huggingface.co/Revai/reverb-asr
- **Paper**: arXiv 2410.03930
- **License**: **Non-commercial** (custom Rev license). Commercial use requires contacting `licensing@rev.com`.
- **Architecture**: WeNet-based joint CTC/attention. 18-layer Conformer encoder + 6-layer bidirectional Transformer decoder. Around 600M params.
- **Languages**: English only.
- **Training data**: 200,000 hours of human-transcribed English. The largest such corpus ever open-sourced for ASR.
- **Verbatimicity parameter** (0 to 1) controls whether disfluencies and false starts are preserved.
- **Decoding modes**: attention, ctc_greedy, ctc_prefix_beam, attention_rescoring, joint_decoding.
- **Diarization**: pyannote-based companion models fine-tuned on 26,000 hours of labeled data.

### Benchmark highlights vs Whisper-large-v3 and Canary-1B

| Dataset | Reverb ASR | Whisper L-v3 | Canary-1B |
|---|---|---|---|
| Earnings21 | **9.68%** | 14.26% | 14.40% |
| Earnings22 | **13.68%** | 19.05% | 19.01% |

Reverb's win on **long-form earnings call audio** is significant: exactly the domain Whisper struggles with. But the non-commercial license rules it out of most product builds.

---

## IBM Granite Speech

IBM's open-weight speech-language models. Apache-2.0.

| Model | Params | Languages | Notes |
|---|---|---|---|
| `granite-speech-3.2-8b` | 8.4B | English | Initial release |
| `granite-speech-3.3-2b` | 2B | en/fr/de/es/pt | Smaller, faster |
| `granite-speech-3.3-8b` | 8.4B | en/fr/de/es/pt | Mean Open ASR WER **5.74%**, LS-clean 1.43% |
| `granite-speech-4.1-2b` / `4.1-2b-plus` / `4.1-2b-nar` | 2B | en/fr/de/es/pt/ja | 4.1 family, NAR = non-autoregressive |
| `granite-4.0-1b-speech` | 1B | not stated | Smallest |

**Architecture**: Two-pass design. 16-block Conformer CTC speech encoder, then a 2-layer q-former projector, then Granite-3.3-8b-instruct LLM with **rank-64 LoRA adapters** on Q/V projections. 10 Hz acoustic embeddings, 5x temporal downsampling.

Training: 32x H100 GPUs, 13 days, IBM Blue Vela cluster. 18k hours of En to {de, es, fr, it, ja, pt, zh} translation data on top of ASR data.

Supports speech translation (X to En and En to X for the 5 ASR languages plus En to Ja and En to Zh). **No streaming.** Open ASR Leaderboard RTFx: 145.

**Browser viability**: Poor (LLM-class). Strong server-side option if you want a permissive commercial license and high accuracy.

---

## Microsoft Phi-4-Multimodal

- **HF**: `microsoft/Phi-4-multimodal-instruct`
- **License**: **MIT**
- **Params**: 5.6B (Phi-4-Mini-Instruct backbone + vision encoder + speech encoder + adapters)
- **Speech languages**: en, zh, de, fr, it, ja, es, pt
- **Open ASR Leaderboard**: 6.02% mean WER (LS-clean 1.69%, LS-other 3.82%, GigaSpeech 9.33%, Earnings22 10.16%, AMI 11.09%). Ranked #1 as of March 2025; since dethroned by Canary-Qwen-2.5B.
- **Audio length**: 40 s for ASR/AST, up to 30 min for summarization.
- **Tasks**: ASR, AST, summarization, speech QA, vision-speech combined, function calling.
- **Streaming**: not supported.
- Beats Whisper-v3 (ASR) and SeamlessM4T-v2-Large (translation) in published comparisons.

Same caveats as Granite: too big for browser, ideal server-side.

---

## Microsoft self-supervised foundations (Wav2Vec2 / HuBERT / WavLM / SpeechT5)

These are **representation models**, not end-to-end ASR systems, but they are the substrate most of the open ASR community fine-tunes on, including Meta MMS (wav2vec2) and many SUPERB-benchmark systems.

- **Wav2Vec2** (Meta, 2020): self-supervised contrastive learning over masked latent speech reps. arXiv 2006.11477. Apache-2.0.
- **HuBERT** (Meta, 2021): masked prediction of clustered targets. Single-speaker focus. arXiv 2106.07447.
- **WavLM** (Microsoft, 2022): HuBERT extension with denoising + utterance mixing for multi-speaker. arXiv 2110.13900. State-of-the-art on SUPERB at release. MIT license on the released checkpoints.
- **SpeechT5** (Microsoft, ACL 2022): unified encoder-decoder for ASR / TTS / S2S translation / voice conversion / speech enhancement. 12-layer encoder + 6-layer decoder, d=768. arXiv 2110.07205. MIT.
- **Data2vec / WavLM-Plus / X-LSR**: same family, less production-relevant for an offline app.

For an offline transcription product these matter mostly as **fine-tuning starting points** if you train a custom model. Direct use of pretrained wav2vec2-large gets you LibriSpeech-clean WER around 2.5% but requires you to ship 300M+ params and a CTC head, so it's almost always worse than a purpose-built model.

---

## Commercial reference (closed source, for benchmarking only)

You can't ship these offline, but they define the accuracy ceiling.

### AssemblyAI Universal-1 / Universal-2

- **Universal-2** (Oct 2024): sub-7% WER on diverse audio, 24% better proper-noun accuracy than U-1, 14.8% better formatting/casing, around 3% lower WER.
- LibriSpeech clean around 2.1%.
- **Streaming**: Universal-Streaming hits around 300 ms P50 emission latency, 41% faster median than Deepgram Nova-3. Multilingual streaming (en, es, fr, de, it, pt) handles intra-utterance code-switching with shared embedding space.
- 99+ languages batch.
- Adds diarization, sentiment, PII redaction, topic detection in pipeline.

### Deepgram Nova-2 / Nova-3

- **Nova-3** (Jan 2025): claimed 54.3% WER reduction streaming, 47.4% batch vs. competitors. Per-language WER not publicly broken down.
- **Nova-3 Multilingual** (2025): around 34% relative reduction batch mean WER, around 21% streaming vs. Nova-2 Multilingual. Strong code-switching gains.
- Continually adding language families through 2025 (10 new langs in Southern Europe / Baltics / SE Asia + Italian, Turkish, Norwegian, Indonesian, plus es/fr/pt, de/nl/sv/da).
- Streaming WebSocket API is the canonical use case.

### Speechmatics Ursa / Ursa 2

- **Ursa 2** (Oct 2024): 18% WER reduction over Ursa across 50+ languages on FLEURS. Sub-1 s real-time latency.
- Most accurate on 62% of supported languages; top-3 on 92%.
- Multiple deployment modes including **on-device container**. Relevant if you ever need a private-cloud or air-gapped option.

These three (plus ElevenLabs Scribe, which is newer and worth tracking) are the WER ceiling. Open models are now within 1 to 2% absolute on standard benchmarks but commercial wins remain in **proper nouns, alphanumerics, accented speech, formatting**.

---

## Legacy / archived but still deployed

### Vosk (alphacephei)

- **License**: Apache-2.0.
- Built on **Kaldi** (TDNN-F / chain models, not Transformer).
- 20+ languages, **50 MB per-language portable models** (also larger 1+ GB server models).
- **Browser**: `vosk-browser` npm package. Kaldi compiled to WebAssembly. Real microphone input + audio file support in 13 languages directly from the page. https://www.npmjs.com/package/vosk-browser
- Streaming-first API. Zero latency in practice.
- Accuracy is significantly behind transformer models: 2024 comparisons show LibriSpeech-clean WER around 8 to 12% depending on model size. Acceptable for command-and-control, voice search, dictation; weak for free-form transcription.

For an offline browser app this is the **other** realistic option besides Moonshine: much smaller download (50 MB) and proven WASM track record, at the cost of accuracy.

### Kaldi

The toolkit underneath Vosk. Still in active use in academic and enterprise contexts but not a modern foundation model. No reason to use it directly in a 2026 product.

### PicoVoice Leopard / Cheetah

- **Commercial** on-device (free tier exists). Leopard = batch, Cheetah = streaming.
- **WebAssembly** packages: `@picovoice/leopard-web`, `@picovoice/cheetah-web`.
- Around 20 MB model. Stored in IndexedDB.
- WER: Cheetah 14.34%, Leopard around 11% (vendor-reported on a generic benchmark).
- 300k-word base vocab, "type-and-train" customisation via Console.
- License: per-user pricing, not Apache.

Good drop-in if you want a polished WASM SDK and are willing to pay; weaker accuracy than Moonshine for a similar size.

### Mozilla DeepSpeech

**Archived November 2021.** Do not use in new code.

### Coqui STT

Coqui shut down active development late 2023; repo still readable but no model zoo, no updates. **Avoid for new builds.**

---

## OWSM, Open Whisper-Style Speech Models (CMU WAVLab)

Academic effort to reproduce Whisper with public data and open toolkit (ESPnet).

| Model | Encoder | Params | Notes |
|---|---|---|---|
| `owsm_v3.1_ebf_base` | E-Branchformer | 101M | |
| `owsm_v3.1_ebf_small` | E-Branchformer | 367M | |
| `owsm_v3.1_ebf` | E-Branchformer | 1.02B | Beats Whisper-large in 8 of 9 English tests |
| `owsm_ctc_v3.1_1B` | E-Branchformer | 1B | CTC head, faster |

- **License**: CC-BY-4.0 (per ESPnet model card practice).
- **Architecture**: E-Branchformer encoder + Transformer decoder. Whisper-style multitask (ASR + AST + LID + alignment) trained on around 180k hours of public data.
- **Training**: 16 to 64 A100s, piecewise-linear LR schedule.
- **Paper**: Peng et al., Interspeech 2024, arXiv 2401.16658.

Mostly relevant as a **reproducible research baseline**. Production users tend to pick Parakeet/Canary instead for speed and Apache/CC-BY-4.0 weights. But OWSM is fully transparent (data + recipes), which Whisper is not, so it's good for academic comparison.

---

## Google open releases (closed-source caveat)

### USM (Universal Speech Model)

- 2B-parameter Conformer encoder, pretrained on **12M hours of unlabeled audio** across 300+ languages.
- ASR for 100+ languages, including under-resourced (Amharic, Cebuano, Assamese, Azerbaijani).
- Around 32.7% relative WER reduction vs. Whisper across 18 evaluated languages.
- **Not open source.** Closed, accessible only via Google's APIs / YouTube captioning.
- Paper: arXiv 2303.01037.

### Gemini Live / Gemini Audio

Closed. Audio input via the Gemini API. Out of scope for offline.

Mentioned because the brief asked; both are non-options for a local-first app.

---

## Distil-Whisper (community)

The brief asked specifically about non-Whisper, but Distil-Whisper is **community Whisper distillation** and worth a one-paragraph mention because it competes with Moonshine in the small-model space.

- **HF org**: `distil-whisper`, plus CTranslate2 ports (`Systran/faster-distil-whisper-large-v3`).
- License: MIT (inherits from Whisper).
- `distil-large-v3.5` is around 1.5x faster than Whisper-large-v3-Turbo on long-form.
- `distil-small.en` is 5.6x faster than Whisper-large-v2 within 3% WER.
- English-focused; community is working on multilingual distillation.

If your reason for "no Whisper" is *license*, it's fine (MIT); if it's *architectural diversity*, Distil-Whisper is still a Whisper. Counts as out-of-scope by the strict reading.

---

## Open ASR Leaderboard, Feb 2026 snapshot

The canonical benchmark. 60+ models, 18 organizations, 11 datasets, three tracks: English short-form, multilingual, long-form. Eval data covers AMI, Earnings-22, GigaSpeech, LibriSpeech (clean + other), SPGISpeech, TED-LIUM, VoxPopuli, CommonVoice, FLEURS.

### English short-form, top by WER

| Rank (approx.) | Model | Params | Mean WER | RTFx | License |
|---|---|---|---|---|---|
| 1 | NVIDIA Canary-Qwen-2.5B | 2.5B | **5.63%** | 418 | CC-BY-4.0 |
| 2 | IBM Granite-Speech-3.3-8B | 8.4B | 5.74% | 145 | Apache-2.0 |
| 3 | Microsoft Phi-4-Multimodal | 5.6B | 6.02% | not reported | MIT |
| 4 | NVIDIA Parakeet-TDT-0.6B-v2 | 0.6B | ~6.05% | 3,380 | CC-BY-4.0 |
| 5 | NVIDIA Parakeet-TDT-0.6B-v3 | 0.6B | 6.34% | 749 | CC-BY-4.0 |
| 6 | NVIDIA Canary-1B-Flash | 0.9B | 6.35% | 1,045 (A100) | CC-BY-4.0 |
| also | NVIDIA Parakeet-CTC-1.1B | 1.1B | 6.68% | **2,793 (top throughput)** | CC-BY-4.0 |
| also | Mistral Voxtral-Mini-3B | 3B | 7.05% | 109 | Apache-2.0 |
| also | OpenAI Whisper-large-v3 | 1.55B | 7.4% | 68 | MIT |
| also | Kyutai STT-2.6B-en | 2.6B | 6.4% (streaming) | 88 | CC-BY-4.0 |
| also | UsefulSensors Moonshine base | 61M | 9.99% | 566 | MIT |

(Exact ranks shuffle constantly; treat as directional.)

### Multilingual track

Whisper-large-v3 was the strongest open baseline through mid-2025; **Canary-1b-v2 and Parakeet-TDT-0.6b-v3** are now better on the 25 EU languages they target, at far higher RTFx. Meta MMS leads on **language breadth** (1100+) but with mediocre WER.

### Long-form track

Closed systems still lead in absolute WER. Among open: Whisper-large-v3 has the strongest accuracy; **Parakeet-CTC-1.1B** has the strongest throughput (around 40x faster than Whisper). Reverb-ASR wins on earnings-call audio specifically.

### Paper

Hugging Face Audio team + collaborators, "Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual and Long-Form Speech Recognition Evaluation", arXiv 2510.06961. The blog at https://huggingface.co/blog/open-asr-leaderboard is the human-readable summary.

---

## Consolidated benchmark table (English Open ASR)

| Model | Params | Arch | Open ASR mean | LS-clean | LS-other | RTFx | License | Streaming | Browser-ready |
|---|---|---|---|---|---|---|---|---|---|
| Canary-Qwen-2.5B | 2.5B | FastConformer + Qwen3 LLM | 5.63 | n/a | n/a | 418 | CC-BY-4.0 | No | No |
| Granite-Speech-3.3-8B | 8.4B | Conformer + Granite LLM | 5.74 | 1.43 | 2.86 | 145 | Apache-2.0 | No | No |
| Phi-4-Multimodal | 5.6B | Conformer + Phi-4 LLM | 6.02 | 1.69 | 3.82 | n/a | MIT | No | No |
| Parakeet-TDT-0.6B-v2 | 0.6B | FastConformer + TDT | 6.05 | 1.69 | 3.19 | 3,380 | CC-BY-4.0 | Yes (chunked) | Via ONNX (community) |
| Parakeet-TDT-0.6B-v3 | 0.6B | FastConformer + TDT | 6.34 | 1.93 | 3.59 | 749 | CC-BY-4.0 | Yes (chunked) | Via ONNX (community) |
| Canary-1B-Flash | 0.9B | FastConformer + Transformer | 6.35 | 1.48 | 2.87 | 1,045 | CC-BY-4.0 | No | No |
| Kyutai STT-2.6B-en | 2.6B | DSM decoder + Mimi | 6.4 (streaming) | 1.7 | 4.32 | 88 | CC-BY-4.0 | **Yes (2.5 s)** | Not yet |
| Parakeet-CTC-1.1B | 1.1B | FastConformer + CTC | 6.68 | n/a | n/a | **2,793** | CC-BY-4.0 | Yes | Via ONNX |
| Voxtral-Mini-3B | 3B | Audio enc + Mistral LLM | 7.05 | 1.88 | 4.1 | 109 | Apache-2.0 | Realtime variant only | No |
| Reverb-ASR | 0.6B | Conformer + bi-Transformer (WeNet) | n/a | n/a | n/a | n/a | Non-commercial | No | No |
| Moonshine v2 base | 61M | Sliding-window Transformer enc-dec | 9.99 | 3.38 | 8.15 | 566 | **MIT** | **Yes (ergodic)** | **Yes (Transformers.js + WebGPU)** |
| Moonshine v2 tiny | 27M | Sliding-window Transformer enc-dec | ~12 | ~5 | ~11 | ~1,200 | MIT | Yes | Yes |
| Meta MMS-1B-all | 1B | Wav2Vec2 + CTC adapters | 22.54 | 12.63 | 15.99 | 231 | CC-BY-NC-4.0 | No | Possible via ONNX |
| Vosk small-en-us | ~50M | Kaldi TDNN-F | ~15 | ~10 | ~22 | streaming-native | Apache-2.0 | **Yes** | **Yes (WASM)** |
| OWSM v3.1 medium | 1B | E-Branchformer + Transformer | ~8 | ~3 | ~7 | n/a | CC-BY-4.0 | No | No |

Empty cells = not reported on that exact split.

---

## Recommendation for off-the-record (offline browser transcription)

Picking the model is a tradeoff of `WER x size x license x browser-ready`:

1. **Default (English-only, browser, WebGPU available)**: **Moonshine v2 base** via `@huggingface/transformers` (Transformers.js). MIT, 61M params, around 120 to 150 MB download, streaming, runs fully in browser. WER is significantly worse than Whisper-large-v3 but matches/beats Whisper-tiny/base, which are the realistic browser comparison points.
2. **Fallback (smaller download, broader language support, less accuracy)**: **Vosk** WASM in 13 languages, 50 MB/language.
3. **If user accepts a one-time large download for higher accuracy and is willing to wait for ONNX/WebGPU port**: Parakeet-TDT-0.6B-v2 (English) or v3 (25 EU languages). 0.6B params is on the edge of WebGPU-runnable on a 6+ GB GPU. Community ONNX exports exist (`FluidInference/parakeet-tdt-0.6b-v3-ov`) but no plug-and-play Transformers.js demo as of writing.
4. **If you want streaming and you're willing to ship a 1B-param model**: Kyutai STT-1B-en_fr. CC-BY-4.0, 0.5 s delay, English+French. No browser runtime yet, would need custom ONNX export.

For partner / IMG context: **none** of the open models on this list are tuned to Australian English or medical vocab, so domain-specific term recognition (drugs, anatomy, procedure names) will be patchy regardless of model choice. Realistic fix is either (a) commercial API with biasing, (b) fine-tune Parakeet/Canary on a private medical corpus, or (c) ship a post-correction LLM step.

---

## Source URLs

### NVIDIA NeMo

- https://huggingface.co/nvidia/canary-1b-flash
- https://huggingface.co/nvidia/canary-180m-flash
- https://huggingface.co/nvidia/canary-1b-v2
- https://huggingface.co/nvidia/canary-qwen-2.5b
- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2
- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- https://huggingface.co/nvidia/parakeet-tdt-1.1b
- https://huggingface.co/nvidia/parakeet-ctc-1.1b
- https://huggingface.co/nvidia/parakeet-tdt_ctc-1.1b
- https://developer.nvidia.com/blog/turbocharge-asr-accuracy-and-speed-with-nvidia-nemo-parakeet-tdt/
- https://developer.nvidia.com/blog/pushing-the-boundaries-of-speech-recognition-with-nemo-parakeet-asr-models/
- https://developer.nvidia.com/blog/nvidia-speech-and-translation-ai-models-set-records-for-speed-and-accuracy/
- https://blogs.nvidia.com/blog/speech-ai-dataset-models/ (Granary)
- https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/nemo_forced_aligner.html
- arXiv 2509.14128 (Canary-1B-v2 / Parakeet-TDT-0.6B-v3 paper)
- arXiv 2408.13106 (NEST: Self-supervised FastConformer)

### Meta

- https://huggingface.co/facebook/mms-1b-all
- https://huggingface.co/facebook/mms-1b-l1107
- https://huggingface.co/facebook/seamless-m4t-v2-large
- https://github.com/facebookresearch/seamless_communication
- https://ai.meta.com/resources/models-and-libraries/seamless-communication-models/
- arXiv 2305.13516 (MMS)
- arXiv 2006.11477 (wav2vec 2.0)
- arXiv 2106.07447 (HuBERT)

### Mistral Voxtral

- https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
- https://huggingface.co/mistralai/Voxtral-Small-24B-2507
- https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
- https://mistral.ai/news/voxtral
- https://mistral.ai/news/voxtral-transcribe-2
- arXiv 2507.13264

### Kyutai

- https://huggingface.co/kyutai/stt-1b-en_fr
- https://huggingface.co/kyutai/stt-2.6b-en
- https://huggingface.co/kyutai/moshiko-pytorch-bf16
- https://github.com/kyutai-labs/delayed-streams-modeling
- https://github.com/kyutai-labs/moshi
- https://kyutai.org/stt
- arXiv 2509.08753 (DSM)
- arXiv 2410.00037 (Moshi)

### UsefulSensors Moonshine

- https://github.com/moonshine-ai/moonshine
- https://huggingface.co/UsefulSensors/moonshine-base
- https://huggingface.co/UsefulSensors/moonshine-tiny
- https://huggingface.co/UsefulSensors/moonshine-streaming-tiny
- https://github.com/huggingface/transformers.js-examples/tree/main/moonshine-web
- arXiv 2410.15608 (Moonshine v1)
- arXiv 2602.12241 (Moonshine v2)
- arXiv 2509.02523 (Flavors of Moonshine)

### Rev Reverb

- https://github.com/revdotcom/reverb
- https://huggingface.co/Revai/reverb-asr
- https://www.rev.com/blog/introducing-reverb-open-source-asr-diarization
- arXiv 2410.03930

### IBM Granite Speech

- https://huggingface.co/ibm-granite/granite-speech-3.3-8b
- https://huggingface.co/ibm-granite/granite-speech-3.3-2b
- https://huggingface.co/ibm-granite/granite-speech-4.1-2b
- https://huggingface.co/ibm-granite/granite-speech-3.2-8b
- https://www.ibm.com/granite/docs/models/speech
- https://www.ibm.com/new/announcements/ibm-granite-3-3-speech-recognition-refined-reasoning-rag-loras

### Microsoft Phi-4 + foundations

- https://huggingface.co/microsoft/Phi-4-multimodal-instruct
- https://github.com/microsoft/SpeechT5
- arXiv 2110.07205 (SpeechT5)
- arXiv 2110.13900 (WavLM)

### Commercial reference

- https://www.assemblyai.com/blog/universal-2-delivers-accuracy-where-it-matters
- https://www.assemblyai.com/universal-2
- https://www.assemblyai.com/blog/introducing-multilingual-universal-streaming
- https://deepgram.com/learn/introducing-nova-3-speech-to-text-api
- https://deepgram.com/learn/nova-3-multilingual-major-wer-improvements-across-languages
- https://www.speechmatics.com/company/articles-and-news/ursa-2-elevating-speech-recognition-across-52-languages

### Legacy / edge

- https://alphacephei.com/vosk/
- https://alphacephei.com/vosk/models
- https://www.npmjs.com/package/vosk-browser
- https://github.com/alphacep/vosk-api
- https://picovoice.ai/platform/leopard/
- https://picovoice.ai/platform/cheetah/
- https://www.npmjs.com/package/@picovoice/leopard-web
- https://www.npmjs.com/package/@picovoice/cheetah-web

### OWSM (CMU)

- https://www.wavlab.org/activities/2024/owsm/
- https://huggingface.co/espnet/owsm_v3.1_ebf
- https://huggingface.co/espnet/owsm_ctc_v3.1_1B
- arXiv 2401.16658

### Google (closed)

- https://research.google/blog/universal-speech-model-usm-state-of-the-art-speech-ai-for-100-languages/
- arXiv 2303.01037

### Open ASR Leaderboard

- https://huggingface.co/spaces/hf-audio/open_asr_leaderboard
- https://huggingface.co/blog/open-asr-leaderboard
- https://github.com/huggingface/open_asr_leaderboard
- arXiv 2510.06961 (the leaderboard paper)

### Misc context

- https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks
- https://the-decoder.com/open-asr-leaderboard-tests-more-than-60-speech-recognition-models-for-accuracy-and-speed/
- https://slator.com/nvidia-microsoft-elevenlabs-top-automatic-speech-recognition-leaderboard/
