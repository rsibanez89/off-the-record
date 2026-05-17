# Chinese and Asian Offline Speech-to-Text Models, Research Notes

Scope: state-of-the-art ASR and audio-LLM models originating from China, Japan, and Korea, with a bias toward what can plausibly be deployed offline in a web browser (WASM / WebGPU / ONNX). Whisper variants and Western models are explicitly out of scope here. They are tracked separately.

The single most important takeaway for the off-the-record (browser-based) use case is at the bottom of this document: **Sherpa-ONNX with a Paraformer or SenseVoice-Small or Streaming-Zipformer model, compiled to WebAssembly, is currently the only fully-tested, fully-offline, browser-resident path for Chinese/Japanese/Korean ASR.** Everything else is either too large for the browser, too restrictively licensed, or only runs server-side.

---

## 1. Alibaba family (Tongyi / DAMO / Qwen)

Alibaba dominates open-source Chinese ASR. Three product lines all under the same parent organisation but with different licensing posture:

1. **FunASR / Paraformer**, the production ASR toolkit (DAMO Academy then ModelScope team).
2. **FunAudioLLM / SenseVoice / CosyVoice**, the multilingual understanding plus voice generation foundation models.
3. **Qwen-Audio / Qwen2-Audio / Qwen2.5-Omni / Qwen3-Omni / Qwen3-ASR**, the LLM-based audio-language models from the Qwen team.

### 1.1 FunASR plus Paraformer

| Attribute | Value |
|---|---|
| Repo | https://github.com/modelscope/FunASR |
| Toolkit license | MIT |
| Model license | FunASR Model License (custom, includes attribution and non-commercial-like clauses, see `MODEL_LICENSE`) |
| Flagship model | `paraformer-zh` / `Paraformer-large` (220M params) |
| Architecture | Non-autoregressive: Encoder, Predictor (2-layer FFN, predicts target length), Sampler (glancing LM), Decoder. Uses BiCif (bidirectional CIF / continuous integrate-and-fire) for token alignment. |
| Training data | ~60,000 hours of manually annotated Mandarin |
| Speedup vs AR | ~12x faster inference than autoregressive seq2seq |
| WER | AISHELL-1 1.95% CER, AISHELL-2 2.85% CER, WenetSpeech test_meeting 6.97% CER |
| Languages | Chinese (zh), English (en), `paraformer-en` separately. Cantonese, Japanese and Korean variants exist on ModelScope. |
| Streaming | Yes, `paraformer-zh-streaming` is a chunked streaming variant. |
| ONNX | First-class. `funasr-onnx` PyPI package. ONNX models are mirrored on HF and ModelScope. |
| Browser | **Indirect**: via `sherpa-onnx` WASM build, Paraformer ONNX runs in the browser. No native funasr-web SDK. |
| Paper | "FunASR: A Fundamental End-to-End Speech Recognition Toolkit", arXiv:2305.11013 |

FunASR also bundles auxiliary models that matter for a real product:

- **FSMN-VAD**, tiny (~0.4M params) feed-forward VAD, ONNX-exportable.
- **CT-Punc** / **CT-Transformer-punc**, controllable Transformer punctuation restoration.
- **emotion2vec**, speech emotion embedding (also via the same toolkit).
- **CAM++** and **ERes2Net** speaker embeddings, **3D-Speaker** for diarization (separate repo).

### 1.2 SenseVoice (FunAudioLLM)

| Attribute | Value |
|---|---|
| Repo | https://github.com/FunAudioLLM/SenseVoice |
| HF | https://huggingface.co/FunAudioLLM/SenseVoiceSmall |
| License | "model-license" (custom, see FunASR `MODEL_LICENSE`). **Not Apache.** Commercial use is ambiguous; multiple open issues on GitHub clarify that the FunAudioLLM team treats it as permissive for product use but the legal text retains restrictions. Audit before shipping. |
| Variants | `SenseVoiceSmall` (released, comparable to Whisper-Small) and `SenseVoiceLarge` (referenced in the paper, supports 50+ languages, not all public). |
| Architecture | Non-autoregressive end-to-end, CTC-like. Predicts language ID, emotion, audio event, transcript in one forward pass. |
| Training data | 400,000+ hours, 50+ languages. |
| Languages supported (Small) | Mandarin, Cantonese, English, Japanese, Korean, plus language identification across 50+ for the Large model. |
| Capabilities | ASR plus Language ID (LID) plus Speech Emotion Recognition (SER) plus Audio Event Detection (AED, laughter, applause, cough, sneeze, BGM, etc.) |
| Speed claim | 70 ms for 10 s of audio. **5x faster than Whisper-Small, 15x faster than Whisper-Large** on similar hardware. |
| ONNX | Yes, `funasr-onnx-0.4.0`. Also libtorch export. |
| Browser | Yes via `sherpa-onnx` (SenseVoice is one of the four ASR families with first-class sherpa-onnx support). |
| Paper | "FunAudioLLM: Voice Understanding and Generation Foundation Models for Natural Interaction Between Humans and LLMs", arXiv:2407.04051 |

SenseVoice is probably the single most attractive Chinese-origin open-source ASR for a multilingual offline app right now: it natively covers CJK plus English plus Cantonese, handles non-speech events, and is small enough to run in the browser via sherpa-onnx.

### 1.3 Qwen audio-language models

These are large multimodal LLMs, **not** browser-deployable. Useful for server-side fallback or evaluation baselines.

| Model | Params | License | Modalities | Repo |
|---|---|---|---|---|
| Qwen-Audio | 8.4B | Tongyi-Qianwen (research, restricted commercial) | audio in to text out | https://github.com/QwenLM/Qwen-Audio |
| Qwen2-Audio | 8.2B (Qwen-7B base plus audio encoder) | Apache-2.0 (Qwen2 line is Apache where weights are released) | audio plus text in to text out, voice chat plus analysis modes | https://github.com/QwenLM/Qwen2-Audio |
| Qwen2.5-Omni | 3B and 7B | Apache-2.0 (`Qwen2.5-Omni-7B`) | text plus image plus audio plus video in to text plus speech out, real-time streaming | https://github.com/QwenLM/Qwen2.5-Omni |
| Qwen3-Omni | 30B-A3B (MoE, ~3B active) | Apache-2.0 | omni-modal, real-time speech generation | https://github.com/QwenLM/Qwen3-Omni |
| Qwen3-ASR | 0.6B and 1.7B | Apache-2.0 | ASR plus 52-language LID plus timestamps plus force-alignment (11 langs). Targeted ASR distillation of Qwen3-Omni. | https://github.com/QwenLM/Qwen3-ASR |

Qwen3-ASR is the noteworthy one for our problem space. Released January 2026, Apache-2.0, sub-2B params, 52 languages, timestamp prediction, music/song-aware. On the AISHELL/WenetSpeech leaderboard the LLM variant reports CER 0.57% / 2.15% / 4.32% on AISHELL-1 / AISHELL-2 / WenetSpeech_meeting respectively, best-in-class. **However**, it still has not been ONNX-exported or quantised down to a size that runs in the browser as of this writing; treat it as server-side only for now.

### 1.4 3D-Speaker (diarization)

| Attribute | Value |
|---|---|
| Repo | https://github.com/modelscope/3D-Speaker |
| Owner | Alibaba Speech Lab |
| Paper | arXiv:2403.19971, "3D-Speaker-Toolkit" |
| Key models | **CAM++** (lightweight speaker embedding), **ERes2Net** / **ERes2NetV2** (SOTA on Mandarin 200k-speaker set), self-supervised SDPN. |
| Use | Speaker verification, recognition, multimodal diarization (acoustic plus semantic plus visual). |
| Browser | CAM++ ONNX is shipped in sherpa-onnx, usable for online diarization in the browser. |

---

## 2. Tencent

### 2.1 Hunyuan

Tencent's open-source Hunyuan family is huge (HunyuanVideo, Hunyuan-Large MoE 389B, Hunyuan-A13B, HunyuanVideo-Foley for video sound generation, etc.) but **no open-source pure ASR model** has been released as of May 2026. Hunyuan-ASR exists as an internal/API product (advertised on Tencent Cloud) but is not on GitHub.

- Org: https://github.com/Tencent-Hunyuan
- Closest audio-related public release: **HunyuanVideo-Foley** (https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley), multimodal Foley audio *generation*, not recognition.

Conclusion: skip Tencent for offline ASR until they open-source something concrete.

---

## 3. ByteDance

### 3.1 Seed-ASR

| Attribute | Value |
|---|---|
| Tech report | https://bytedancespeech.github.io/seedasr_tech_report/ |
| Paper | arXiv:2407.04675 |
| Architecture | Audio-Conditioned LLM (AcLLM). Continuous speech features plus contextual text injected into an LLM. |
| Training data | 20M+ hours of speech, 900k+ hours paired. |
| Languages | Mandarin plus 13 Chinese dialects plus 7 foreign languages (incl. English with accents). |
| **Open source?** | **No.** Weights and code are not released. Internal/API only. Listed here because it's a public benchmark reference. |
| Benchmark | Seed-ASR 2.0 (API): AISHELL-1 CER 1.52%, AISHELL-2 CER 2.77%, WenetSpeech meeting 4.74%. |

### 3.2 Seed-TTS

Same posture, **not open source** ("AI safety considerations"). Only an evaluation harness (https://github.com/BytedanceSpeech/seed-tts-eval) is public. Paper: arXiv:2406.02430.

ByteDance is therefore evaluation-only material; nothing to deploy.

---

## 4. Moonshot AI, StepFun, Zhipu (audio-LLM cohort)

These are open-weight Chinese audio-language models. They are far too large for the browser (7 to 8B params each), but interesting as server-side fallbacks or as upstream models to distil from.

### 4.1 Moonshot Kimi-Audio

| Attribute | Value |
|---|---|
| Repo | https://github.com/MoonshotAI/Kimi-Audio |
| HF | https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct |
| Variants | `Kimi-Audio-7B` (base), `Kimi-Audio-7B-Instruct` |
| License | Released April 25 2025. Custom Moonshot license: research and commercial use both allowed with attribution; verify against the HF model card. |
| Training | 13M+ hours of audio (speech, music, sounds) plus paired text. |
| Capabilities | ASR plus AQA (audio QA) plus AAC (captioning) plus SER plus SEC/ASC plus end-to-end voice conversation. |
| Tooling | `Kimi-Audio-Evalkit` for benchmark reproduction. |
| Reported ASR | SOTA on a large basket of audio benchmarks at release. |

### 4.2 StepFun Step-Audio 2

| Attribute | Value |
|---|---|
| Repo | https://github.com/stepfun-ai/Step-Audio2 |
| HF | https://huggingface.co/stepfun-ai/Step-Audio-2-mini |
| Variants | `Step-Audio 2 mini` (~8B), `Step-Audio 2 mini Base`, `Step-Audio 2 mini Think` |
| License | **Apache-2.0** for the mini variants. Rare for a Chinese 8B audio LLM. |
| Capabilities | End-to-end speech-to-speech conversation, ASR, audio understanding, paralinguistic features, multi-turn dialogue. |
| Paper | "Step-Audio 2 Technical Report", arXiv:2507.16632 |
| Reported | Mini reportedly surpasses GPT-4o-Audio on several speech recognition and audio understanding benchmarks. |
| Dependencies | transformers, torchaudio, librosa, onnxruntime, `s3tokenizer`. |

This is currently one of the most legitimately permissive open-weight speech-to-speech models. Server-side only at this size.

### 4.3 Zhipu GLM-4-Voice

| Attribute | Value |
|---|---|
| Repo | https://github.com/zai-org/GLM-4-Voice |
| HF | `THUDM/glm-4-voice-9b`, `THUDM/glm-4-voice-tokenizer`, `THUDM/glm-4-voice-decoder` (now under `zai-org`) |
| Released | 25 October 2024 |
| License | Code Apache-2.0; weights have an additional model license, generally permissive but verify per file. |
| Architecture | Three components: **Tokenizer** (Whisper encoder plus vector quantization, 12.5 discrete tokens/sec), **9B LLM** (GLM-4-9B fine-tuned on speech), **Decoder** (CosyVoice-based, streaming, starts after only 10 tokens). |
| Languages | Mandarin and English, real-time conversational. |
| Controllable | Emotion, tone, speech rate, dialect via natural-language instructions. |

GLM-4-Voice is conversational-first, not transcription-first. Closer to GPT-4o-voice in posture than to Paraformer.

---

## 5. Telco / iFlyTek / China-Telecom dialect models

### 5.1 TeleSpeech-ASR (China Telecom, Tele-AI)

| Attribute | Value |
|---|---|
| Repo | https://github.com/Tele-AI/TeleSpeech-ASR |
| HF | https://huggingface.co/Tele-AI/TeleSpeech-ASR1.0 |
| Pretraining | 300,000 hours of unlabelled multi-dialect Chinese speech. |
| Released | Three artefacts: two SSL pretrained models plus one fine-tuned model on KeSpeech (8 dialects). |
| Dialects | Up to **30 Chinese dialects** including Cantonese, Shanghai, Sichuan, Wenzhou, Hokkien, etc. via fine-tuning. |
| License | Custom Tele-AI agreement. Community use allowed, **commercial use requires written approval** (email `tele_ai@chinatelecom.cn`). Effectively non-commercial without negotiation. |
| ONNX | Yes, CTC variant exported to ONNX, supported in **sherpa-onnx** as a first-class model family (`sherpa-onnx/onnx/pretrained_models/telespeech`). |
| Frameworks | fairseq-based fine-tuning plus WeNet-based feature extraction recipes provided. |

This is the single most useful model in the world for offline Chinese-dialect transcription, and the sherpa-onnx integration means it's already browser-deployable. Licensing is the main friction.

### 5.2 iFlyTek (Xunfei) Spark

iFlyTek (科大讯飞, "kē dà xùn fēi") runs the dominant commercial Chinese ASR service. They claim 97%+ Mandarin accuracy and support 74 dialects/languages without explicit switching via their Spark voice model.

**Open-source status: nothing meaningful.** The iFLYTEK Open AI Platform (global.xfyun.cn) is a paid API. iFlyTek Spark X1 / Spark voice are closed weights. Skip for an offline product.

---

## 6. Next-gen Kaldi: k2 / icefall / Sherpa (Daniel Povey / Xiaomi-backed)

This is the most important section for the off-the-record use case. Daniel Povey's group is the only one publishing high-quality open-source ASR that *targets* offline edge / browser / embedded deployment.

### 6.1 The stack

| Component | Repo | Role |
|---|---|---|
| **k2** | https://github.com/k2-fsa/k2 | PyTorch plus FSA / WFST library, fast beam search on GPU. |
| **icefall** | https://github.com/k2-fsa/icefall | Training recipes (PyTorch). All Zipformer / Conformer / RNN-T / CTC recipes live here. |
| **sherpa** | https://github.com/k2-fsa/sherpa | Server-side framework (libtorch). |
| **sherpa-onnx** | https://github.com/k2-fsa/sherpa-onnx | ONNX-runtime deployment for all platforms incl. **WebAssembly**. **Apache-2.0.** |
| **sherpa-ncnn** | https://github.com/k2-fsa/sherpa-ncnn | NCNN (Tencent) deployment for mobile / RISC-V / embedded. Also supports WebAssembly. |

### 6.2 Sherpa-ONNX, browser specifics

- License: **Apache-2.0** (clean, no encumbrance).
- 12 official language bindings (C, C++, Python, JS/Node, Java, C#, Kotlin, Swift, Go, Dart, Rust, Pascal) plus **WebAssembly**.
- NPM package: `sherpa-onnx` (https://www.npmjs.com/package/sherpa-onnx).
- Pre-built WASM artefacts hosted by k2-fsa on Hugging Face Spaces. Runs entirely in the browser, no server.
- **Execution providers**: CPU by default; ONNX-runtime-web's WebGPU EP is available in principle, but sherpa-onnx's WASM build is currently wired against `onnxruntime-web` with WASM SIMD threads; WebGPU acceleration of the sherpa-onnx graphs is not turn-key.
- Audio capture: pair with Web Audio API or `MediaRecorder` plus `AudioWorklet` for raw 16 kHz PCM.

### 6.3 Models supported in sherpa-onnx, ranked by browser relevance

| Family | Streaming? | Languages | Notes |
|---|---|---|---|
| **Zipformer-Transducer (streaming)** | yes | many: Chinese, English, Korean, Cantonese, Vietnamese, French, Tibetan, multilingual | The default choice for low-latency browser ASR. Small variants are 30 to 60 MB quantised. Apache-2.0. |
| **Zipformer-CTC** | offline | as above | Slightly worse WER than transducer but trivial decoder. |
| **Paraformer (offline)** | no | zh, en, dialects | Best Chinese accuracy at the model size. |
| **SenseVoice** | no (chunked) | zh / yue / en / ja / ko plus 50 langs (LID) | Multilingual one-shot. Currently the most attractive single model for an offline app needing CJK plus English plus emotion/event tags. |
| **Whisper** | no | multilingual | Out of scope for this doc but supported. |
| **TeleSpeech (offline CTC)** | no | 30 Chinese dialects | License-restricted. |
| **NeMo Transducer / CTC** | both | English-centric | Western, outside our scope. |

### 6.4 Zipformer architecture

- Paper: **arXiv:2310.11230**, "Zipformer: A faster and better encoder for automatic speech recognition" (ICLR 2024). Yao et al., Daniel Povey.
- U-Net-like encoder with stacks running at progressively *lower* frame rates in the middle.
- Reorganised block structure where attention weights are reused across modules.
- **BiasNorm** replaces LayerNorm (retains some length information).
- **SwooshR / SwooshL** activations (variants of Swish).
- Paired with the **ScaledAdam** optimiser. Each tensor's update is scaled by current parameter magnitude.
- Evaluation: SOTA-or-better on LibriSpeech, AISHELL-1, WenetSpeech with smaller models than Conformer baselines.
- Also has a unified streaming/non-streaming variant, see arXiv:2506.14434.

For an offline-first browser ASR app, **Streaming Zipformer-Transducer compiled via sherpa-onnx WASM is the production-grade default.**

---

## 7. PaddleSpeech / WeNet / WeSpeaker / FireRedASR

### 7.1 PaddleSpeech (Baidu)

| Attribute | Value |
|---|---|
| Repo | https://github.com/PaddlePaddle/PaddleSpeech |
| License | Apache-2.0 |
| Award | NAACL 2022 Best Demo |
| Models | **DeepSpeech2** (legacy, still maintained), **Conformer**, **Transformer**, **U2/U2++ streaming Conformer**. |
| Training corpora | AISHELL, WenetSpeech, LibriSpeech. |
| Languages | Chinese, English, Chinese-English mixed. |
| Deployment | C++ high-performance streaming server. ONNX path is less complete than FunASR or sherpa-onnx, Paddle's native serving is preferred. |
| Browser | No native browser support. Models must be re-exported to ONNX manually. |

PaddleSpeech is heavy and tightly coupled to PaddlePaddle. Useful as a training reference but rarely the right deployment target for the browser.

### 7.2 WeNet / WeNet 2.0 (Mobvoi plus collaborators)

| Attribute | Value |
|---|---|
| Repo | https://github.com/wenet-e2e/wenet |
| License | Apache-2.0 |
| Architecture | **U2 / U2++**, unified two-pass joint CTC/AED with shared Conformer encoder. U2++ adds a bidirectional attention decoder; 10% relative WER win over U2. |
| Streaming | Single model serves both streaming and non-streaming via dynamic chunk masking. |
| Production | Triton, ONNX, GPU and CPU runtimes. Used in real products at Mobvoi, Tencent, JD, NIO, etc. |
| Paper | arXiv:2102.01547 (WeNet), arXiv:2203.15455 (WeNet 2.0) |

WeNet is the most production-pedigreed Chinese ASR toolkit after FunASR. It exports cleanly to ONNX, and the U2++ Conformer is one of the better streaming Mandarin models available freely. Sherpa-onnx can consume WeNet-exported models.

Companion projects:

- **WeSpeaker**, https://github.com/wenet-e2e/wespeaker, speaker verification and diarization, Apache-2.0.
- **WenetSpeech**, https://github.com/wenet-e2e/WenetSpeech, the 22,400-hour Mandarin corpus (see Datasets below).
- **WenetSpeech-Yue**, https://github.com/ASLP-lab/WenetSpeech-Yue, large-scale Cantonese corpus (arXiv:2509.03959).

### 7.3 FireRedASR (Xiaohongshu / RedNote)

| Attribute | Value |
|---|---|
| Repo | https://github.com/FireRedTeam/FireRedASR |
| Repo v2 | https://github.com/FireRedTeam/FireRedASR2S |
| Demo | https://fireredteam.github.io/demos/firered_asr/ |
| Paper | arXiv:2501.14350 (v1), arXiv:2603.10420 (v2 / "FireRedASR2S") |
| Variants | **FireRedASR-LLM** (8.3B, LLM-integrated), CER 3.05% average on Mandarin benchmarks; **FireRedASR-AED** (1.1B, encoder-decoder), CER 3.18%. |
| Languages | Mandarin plus 20+ Chinese dialects/accents, English, code-switching, ASR plus **singing lyrics**. |
| FireRedASR2S | Adds VAD, LID (100+ languages), Punctuation in a single all-in-one system. |
| License | Apache-2.0 on the FireRedASR repo (verify per-checkpoint on HF, some weights are released under a more restrictive bundle). |
| Browser | The 1.1B AED variant is too big for the browser; could be distilled/quantised, but nothing prebuilt. |

FireRedASR is currently the open-source Chinese ASR accuracy leader at the 1B-class size. If a server-side fallback is acceptable, this is the model to call.

---

## 8. Japanese

### 8.1 ReazonSpeech

| Attribute | Value |
|---|---|
| Repo | https://github.com/reazon-research/ReazonSpeech |
| Type | Both a 35,000-hour Japanese speech **corpus** and a set of trained models (NeMo Conformer-Transducer). |
| License | Corpus released for research; models released under model-specific terms, see repo. |
| Size | World's largest open Japanese speech corpus at release. |

### 8.2 kotoba-whisper (kotoba-tech)

| Attribute | Value |
|---|---|
| Repo | https://github.com/kotoba-tech/kotoba-whisper |
| HF | `kotoba-tech/kotoba-whisper-v1.0`, `v1.1`, `v2.0`, `v2.2` |
| License | Apache-2.0. |
| Architecture | Distilled Whisper, full Whisper-large-v3 encoder plus 2-layer decoder. |
| Speed | 6.3x faster than Whisper-large-v3 with comparable Japanese WER. |
| Training | 1,253 hours of ReazonSpeech-large plus 16.8M transcript chars (v1). v2 adds more data. |
| Browser | Inherits Whisper's browser compatibility. `transformers.js` and ONNX-runtime-web work, including WebGPU. Whisper is technically out of scope here, but this Japanese distillation is in scope. |

Out-of-scope note: kotoba-whisper is a Whisper derivative, but as a Japanese-specific tuned model it is genuinely useful and lives at the edge of this brief.

### 8.3 J-Moshi

| Attribute | Value |
|---|---|
| Repo | https://github.com/nu-dialogue/j-moshi |
| Variant repo | https://github.com/llm-jp/llm-jp-moshi |
| License | See repo `LICENSE`, research-friendly. |
| Architecture | 7B full-duplex spoken-dialogue model, built on Kyutai Moshi backbone, additionally trained on Japanese. |
| Training | J-CHAT corpus, 69,000 hours of Japanese dialogue. |
| Use | Conversational agent, not pure ASR. Needs 24+ GB VRAM. |

Server-only, not relevant for a browser transcription app, but the most prominent open Japanese audio-LLM.

---

## 9. Korean

### 9.1 KoSpeech

| Attribute | Value |
|---|---|
| Repo | https://github.com/sooftware/kospeech |
| License | Apache-2.0. |
| Models | DeepSpeech2, LAS, Speech Transformer, Joint CTC-Attention LAS, Conformer, Jasper, RNN-Transducer. |
| Training corpus | KsponSpeech (1,000 hours, AI Hub Korean). |
| Paper | Kim et al., 2021, *Software Impacts*, published as SoftwareImpacts-2020-63. |
| Browser | None pre-built. Needs manual ONNX export. |

### 9.2 ClovaCall (Naver Clova AI)

| Attribute | Value |
|---|---|
| Repo | https://github.com/clovaai/ClovaCall |
| Type | Dataset plus baseline LAS model. ~112,000 (utterance, transcript) pairs of restaurant-reservation dialogue. |
| Use | Korean goal-oriented dialogue ASR. |
| Paper | Interspeech 2020. |

### 9.3 Naver / Clova general

Naver's commercial Clova Speech is closed. The CLaF framework (https://github.com/naver/claf) is general NLP, not ASR.

### 9.4 Sherpa-ONNX Korean Zipformer

- HF: `k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16`
- License: Apache-2.0.
- Streaming, ONNX, browser-deployable via sherpa-onnx WASM.

This is currently the most practical offline-Korean option for a browser app.

---

## 10. Datasets (essential ones to know)

| Dataset | Hours | Language(s) | License | Notes |
|---|---|---|---|---|
| **AISHELL-1** | 178 | Mandarin (zh) | Apache-2.0 | 400 speakers. Canonical Mandarin ASR benchmark. OpenSLR 33. |
| **AISHELL-2** | 1,000 | Mandarin | Research/free with form | 1,991 speakers, iOS/Android/Mic. |
| **AISHELL-3** | 85 | Mandarin (multispeaker TTS) | Apache-2.0 | 218 speakers, TTS-focused. |
| **AISHELL-4** | ~120 | Mandarin (meeting, multichannel) | research | 8-channel meeting recordings. |
| **AISHELL-5** | (n/a) | Mandarin (in-car multi-channel multi-speaker) | research | First open-source in-car diarization plus ASR set (2025). |
| **WenetSpeech** | 22,400+ (10k labeled plus 12.4k weakly/unlabeled) | Mandarin | **CC-BY 4.0 (non-commercial only)** | YouTube plus Podcast. OpenSLR 121. Copyright remains with original uploaders. |
| **WenetSpeech-Yue** | 20,000+ | Cantonese (Yue) | research | arXiv:2509.03959. |
| **KeSpeech** | 1,542 | Mandarin plus 8 subdialects | academic-free | 27,237 speakers, 34 cities. Includes parallel Mandarin/dialect pairs. NeurIPS 2021. |
| **ReazonSpeech** | 35,000+ | Japanese | research/permissive | TV-derived. |
| **KsponSpeech** | 1,000 | Korean | AI-Hub terms | KoSpeech's training set. |
| **GigaSpeech 2** | 30,000 raw / 22,000 refined | Thai, Indonesian, Vietnamese | research | Low-resource Southeast Asian. arXiv:2406.11546. ACL 2025. |
| **J-CHAT** | 69,000 | Japanese (dialogue) | research | Powers J-Moshi. |

---

## 11. Benchmarks, Mandarin ASR leaderboard (CER, lower is better)

Aggregated from the papers above. **Bold** = SOTA at the time of writing for that benchmark. All numbers are character error rate (CER) on the standard test split unless otherwise noted.

| Model | Params | AISHELL-1 | AISHELL-2 | WenetSpeech test_net | WenetSpeech test_meeting | LibriSpeech test-clean | License |
|---|---|---|---|---|---|---|---|
| Paraformer-large | 220M | 1.95 | 2.85 | (n/a) | 6.97 | n/a | FunASR custom |
| WeNet U2++ Conformer | ~120M | ~4.6 | ~5.5 | ~9 | ~16 | n/a | Apache-2.0 |
| Streaming Zipformer (bilingual zh-en) | ~70M | ~4.5 | ~5.0 | (n/a) | (n/a) | ~3.5 | Apache-2.0 |
| Zipformer (offline, large) | ~150M | 2.0 | (n/a) | (n/a) | (n/a) | 2.0 | Apache-2.0 |
| SenseVoice-Small | ~250M | < Whisper-Large | < Whisper-Large | (n/a) | (n/a) | (n/a) | model-license |
| FireRedASR-AED | 1.1B | ~1.6 | ~2.7 | ~5.0 | (n/a) | (n/a) | Apache-2.0 |
| FireRedASR-LLM | 8.3B | ~1.4 | ~2.5 | ~4.6 | (n/a) | (n/a) | Apache-2.0 |
| FireRedASR2S | (n/a) | 1.48 | 2.71 | 4.97 | (n/a) | (n/a) | Apache-2.0 |
| Seed-ASR 2.0 (closed) | >12B | **1.52** | 2.77 | (n/a) | 4.74 | (n/a) | proprietary |
| **Qwen3-ASR-LLM** | 1.7B | **0.57 (AED variant)** / 0.64 (LLM) | **2.15** | (n/a) | **4.32** | (n/a) | **Apache-2.0** |

Some Paraformer/WeNet numbers above are approximations from secondary sources. Verify against arXiv:2305.11013 and arXiv:2102.01547 before quoting in production materials.

For our use case (browser, offline), the **practical** trade-off is between three points on this table:

1. **Streaming Zipformer-Transducer (~70 MB int8 model, 1x RTF on a laptop)**, best latency, lower accuracy. Apache-2.0.
2. **SenseVoice-Small (~250 MB fp16 / ~120 MB int8)**, best multilingual coverage incl. CJK plus English plus emotion plus events, near-best Chinese accuracy. Custom license.
3. **Paraformer-large (~220 MB fp16 / ~120 MB int8)**, best Chinese-only accuracy in a small model. Custom license.

---

## 12. License field guide, what actually matters for a deployed app

Most Chinese ASR models *look* open but are released under restrictive custom agreements. Pay attention to:

- **Apache-2.0**, safe (Qwen2-Audio, Qwen3-ASR, Qwen3-Omni, Step-Audio 2 mini, sherpa-onnx, k2, icefall, WeNet, PaddleSpeech, FireRedASR, GLM-4-Voice code).
- **MIT**, safe (FunASR toolkit code).
- **FunASR Model License** (custom), applies to Paraformer-zh, SenseVoice, CAM++, ERes2Net weights. Attribution required. Commercial use is *not explicitly forbidden* but multiple GitHub issues show the team treats this as "okay with attribution"; legal review recommended.
- **Tele-AI Agreement**, TeleSpeech models. **Commercial use needs written permission**, email `tele_ai@chinatelecom.cn`. Treat as non-commercial by default.
- **Tongyi-Qianwen License**, older Qwen variants (pre-Qwen2). Research-only without explicit commercial application.
- **Custom Moonshot license**, Kimi-Audio. Generally permissive but check the HF card.
- **CC-BY 4.0 (WenetSpeech)**, non-commercial for the *audio* (videos belong to original uploaders); only the transcripts/metadata are CC-BY.

For our off-the-record product, the cleanest combo today is:
- **Apache-2.0 path**: sherpa-onnx plus Zipformer streaming models plus Qwen3-ASR (server fallback).
- **Custom-but-acceptable path**: sherpa-onnx plus SenseVoice-Small plus Paraformer (verify attribution requirements).
- **Avoid for shipping**: TeleSpeech (commercial requires negotiation), Seed-ASR (closed), iFlyTek (closed), Tencent Hunyuan-ASR (closed), older Qwen-Audio under Tongyi-Qianwen.

---

## 13. Recommendation for off-the-record (browser, offline)

1. **Primary engine**: `sherpa-onnx` WASM build (Apache-2.0, npm package available, supports both streaming and offline pipelines, ONNX-runtime-web under the hood). Use the Node-N-API path for the Electron/desktop variant and the WASM path for the pure-browser variant.
2. **Default Mandarin/Cantonese/multilingual model**: `SenseVoice-Small` ONNX int8. Single forward pass gives transcript plus language ID plus emotion plus event tags (BGM/laugh/applause/cough). Covers Chinese, Cantonese, English, Japanese, Korean from a single ~120 MB checkpoint.
3. **Low-latency streaming model**: a small `streaming-zipformer-bilingual-zh-en` (Apache-2.0) from the k2-fsa Hugging Face org. Roughly 30 to 60 MB int8. Use when the UI needs partial hypotheses faster than SenseVoice's chunk size.
4. **Korean-specific**: `k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16` (Apache-2.0).
5. **Japanese-specific**: either kotoba-whisper distilled (out of scope here but worth listing) or a sherpa-onnx Japanese Zipformer if available; otherwise SenseVoice covers Japanese acceptably.
6. **Dialect Chinese (Cantonese, Sichuan, Shanghai, etc.)**: TeleSpeech offline CTC via sherpa-onnx, but only if the licensing question is resolved. Otherwise rely on SenseVoice's Cantonese support and accept the accuracy hit on other dialects.
7. **VAD**: FunASR's FSMN-VAD (ONNX, <1 MB), available via sherpa-onnx (`vad-asr` WASM example).
8. **Punctuation**: CT-Punc / CT-Transformer-punc from FunASR. ONNX, runs in browser via sherpa-onnx.
9. **Diarization (if needed)**: CAM++ speaker embedding via sherpa-onnx, paired with simple AHC clustering. ERes2NetV2 if more accuracy needed and the model size budget allows.
10. **Server-side fallback for accuracy-critical Mandarin** (if and when network is available): Qwen3-ASR-1.7B (Apache-2.0). Best public Mandarin accuracy at a sane size.

---

## 14. Source URLs (one-stop reference list)

### Toolkits / runtimes
- FunASR: https://github.com/modelscope/FunASR
- FunASR MODEL_LICENSE: https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE
- SenseVoice: https://github.com/FunAudioLLM/SenseVoice
- 3D-Speaker: https://github.com/modelscope/3D-Speaker
- WeNet: https://github.com/wenet-e2e/wenet
- WeSpeaker: https://github.com/wenet-e2e/wespeaker
- PaddleSpeech: https://github.com/PaddlePaddle/PaddleSpeech
- k2: https://github.com/k2-fsa/k2
- icefall: https://github.com/k2-fsa/icefall
- sherpa: https://github.com/k2-fsa/sherpa
- sherpa-onnx: https://github.com/k2-fsa/sherpa-onnx
- sherpa-onnx (npm): https://www.npmjs.com/package/sherpa-onnx
- sherpa-onnx docs: https://k2-fsa.github.io/sherpa/onnx/index.html
- sherpa-onnx WebAssembly: https://k2-fsa.github.io/sherpa/onnx/wasm/index.html
- sherpa-ncnn: https://github.com/k2-fsa/sherpa-ncnn

### Models
- Paraformer-zh (HF): https://huggingface.co/funasr/paraformer-zh
- Paraformer-en (HF): https://huggingface.co/funasr/paraformer-en
- Paraformer-zh-streaming (HF): https://huggingface.co/funasr/paraformer-zh-streaming
- SenseVoiceSmall (HF): https://huggingface.co/FunAudioLLM/SenseVoiceSmall
- Qwen2-Audio-7B (HF): https://huggingface.co/Qwen/Qwen2-Audio-7B
- Qwen2.5-Omni-7B (HF): https://huggingface.co/Qwen/Qwen2.5-Omni-7B
- Qwen3-Omni-30B-A3B-Instruct (HF): https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
- Qwen3-ASR: https://github.com/QwenLM/Qwen3-ASR
- Kimi-Audio-7B-Instruct: https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct
- Step-Audio-2-mini: https://huggingface.co/stepfun-ai/Step-Audio-2-mini
- GLM-4-Voice-9B: https://huggingface.co/THUDM/glm-4-voice-9b
- TeleSpeech-ASR1.0: https://huggingface.co/Tele-AI/TeleSpeech-ASR1.0
- FireRedASR: https://github.com/FireRedTeam/FireRedASR
- FireRedASR2S: https://github.com/FireRedTeam/FireRedASR2S
- kotoba-whisper: https://github.com/kotoba-tech/kotoba-whisper
- ReazonSpeech: https://github.com/reazon-research/ReazonSpeech
- J-Moshi: https://github.com/nu-dialogue/j-moshi
- KoSpeech: https://github.com/sooftware/kospeech
- ClovaCall: https://github.com/clovaai/ClovaCall
- Sherpa-ONNX Korean Streaming Zipformer: https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16

### Datasets
- WenetSpeech: https://github.com/wenet-e2e/WenetSpeech, OpenSLR 121: https://openslr.org/121/
- WenetSpeech-Yue: https://github.com/ASLP-lab/WenetSpeech-Yue
- KeSpeech: https://github.com/KeSpeech/KeSpeech
- GigaSpeech 2: https://github.com/SpeechColab/GigaSpeech2, HF: https://huggingface.co/datasets/speechcolab/gigaspeech2

### Papers (arXiv)
- FunASR toolkit: https://arxiv.org/abs/2305.11013
- FunAudioLLM (SenseVoice plus CosyVoice): https://arxiv.org/abs/2407.04051
- Qwen2-Audio: https://arxiv.org/abs/2407.10759
- WeNet: https://arxiv.org/abs/2102.01547
- WeNet 2.0: https://arxiv.org/abs/2203.15455
- WenetSpeech corpus: https://arxiv.org/abs/2110.03370
- WenetSpeech-Yue: https://arxiv.org/abs/2509.03959
- Zipformer: https://arxiv.org/abs/2310.11230
- Unified streaming/non-streaming Zipformer: https://arxiv.org/abs/2506.14434
- Seed-ASR: https://arxiv.org/abs/2407.04675
- Seed-TTS: https://arxiv.org/abs/2406.02430
- Step-Audio 2 tech report: https://arxiv.org/abs/2507.16632
- FireRedASR: https://arxiv.org/abs/2501.14350
- FireRedASR2S: https://arxiv.org/abs/2603.10420
- 3D-Speaker toolkit: https://arxiv.org/abs/2403.19971
- GigaSpeech 2: https://arxiv.org/abs/2406.11546
- J-Moshi: https://arxiv.org/abs/2506.02979

### Tech reports / blogs
- Seed-ASR tech report site: https://bytedancespeech.github.io/seedasr_tech_report/
- Seed-TTS tech report site: https://bytedancespeech.github.io/seedtts_tech_report/
- FunAudioLLM homepage: https://funaudiollm.github.io/
- FireRedASR demo: https://fireredteam.github.io/demos/firered_asr/
- iFLYTEK Open AI Platform: https://global.xfyun.cn/

---

## 15. Open questions and things to verify before shipping

1. **SenseVoice model-license commercial clarification**, read the actual `MODEL_LICENSE` text in `modelscope/FunASR`. If unacceptable, fall back to Apache-2.0 Streaming Zipformer plus a Whisper-derived multilingual model for non-CJK languages.
2. **Quantised SenseVoice in WASM**, verify that the int8 ONNX checkpoint loads and runs at <1x RTF on a mid-range laptop and on iOS Safari (which currently lacks WASM SIMD threads in some configurations).
3. **WebGPU EP for ONNX-runtime-web with sherpa-onnx**, not turn-key. Investigate whether the WebGPU operator coverage is sufficient for the Zipformer transducer graph.
4. **TeleSpeech commercial path**, if dialect coverage matters strategically, negotiate with Tele-AI; otherwise drop.
5. **Qwen3-ASR ONNX**, as of this writing not exported. Track the QwenLM/Qwen3-ASR repo; could replace SenseVoice as the default multilingual ASR if and when a quantised browser-runnable build appears.
6. **Streaming Paraformer in the browser**, paraformer-zh-streaming exists but is less well-trodden in sherpa-onnx than Zipformer-streaming. Benchmark before choosing it over Zipformer.
