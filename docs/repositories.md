# Repositories: Public Code for Offline Speech Recognition

> Flat catalogue of every public repository surfaced during the state of the art research, grouped by family. Each entry: URL, license, one line description. Full analysis lives in the `research-*.md` files.
> Last updated 2026-05-16.

## How to read this

- License is captured per release. Apache-2.0, MIT, BSD: shippable. CC-BY-4.0: shippable with attribution. CC-BY-NC and custom non commercial: research only without a paid license.
- "Browser" column means there is a public WebAssembly, WebGPU, or transformers.js path that runs the model in a tab. "Server only" means runs in Python or native, no public browser port.
- All URLs are direct GitHub or Hugging Face links unless noted.

---

## 1. OpenAI Whisper family

### Reference

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| openai/whisper | https://github.com/openai/whisper | MIT | via ports | Reference PyTorch implementation. 99.5k stars. |
| openai HF org (weights) | https://huggingface.co/openai | MIT | via ports | tiny, base, small, medium, large (v1/v2/v3), large-v3-turbo. |

### Inference engines (Whisper)

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| ggml-org/whisper.cpp | https://github.com/ggml-org/whisper.cpp | MIT | yes (WASM) | Pure C/C++ via ggml. Metal, CUDA, Vulkan, CoreML, WASM SIMD backends. 49.7k stars. |
| SYSTRAN/faster-whisper | https://github.com/SYSTRAN/faster-whisper | MIT | no | CTranslate2 backend, INT8/FP16/INT8-FP16, batched inference. 22.9k stars. |
| huggingface/transformers.js | https://github.com/huggingface/transformers.js | Apache-2.0 | yes (WebGPU + WASM) | Reference JS port. ONNX Runtime Web. WebGPU since v3. |
| xenova/whisper-web | https://github.com/xenova/whisper-web | Apache-2.0 | yes | Template for in browser Whisper via transformers.js. |
| sanchit-gandhi/whisper-jax | https://github.com/sanchit-gandhi/whisper-jax | Apache-2.0 | no | JAX impl for TPU. ~70x speedup on TPU v4-8. |
| Vaibhavs10/insanely-fast-whisper | https://github.com/Vaibhavs10/insanely-fast-whisper | Apache-2.0 | no | HF Transformers plus Flash Attention 2 plus BetterTransformer. |
| argmaxinc/argmax-oss-swift (WhisperKit) | https://github.com/argmaxinc/argmax-oss-swift | MIT | no (native iOS/macOS) | Swift plus CoreML, encoder on Apple Neural Engine. |
| ml-explore/mlx-examples | https://github.com/ml-explore/mlx-examples | MIT | no | mlx-whisper, Apple Silicon native via MLX. |
| huggingface/candle | https://github.com/huggingface/candle | Apache-2.0 / MIT | yes (WASM) | Rust ML framework, candle-examples/examples/whisper. |
| onnx-community on HF | https://huggingface.co/onnx-community | model deps | yes | Curated ONNX exports of Whisper Turbo, base, large-v3 for transformers.js. |

### Streaming Whisper

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| ufal/whisper_streaming | https://github.com/ufal/whisper_streaming | MIT | port-able | Reference LocalAgreement-2 implementation. 3.6k stars. |
| ufal/SimulStreaming | https://github.com/ufal/SimulStreaming | MIT | no | Successor with AlignAtt. ~5x faster than whisper_streaming. IWSLT 2025 winner. |
| altalt-org/Lightning-SimulWhisper | https://github.com/altalt-org/Lightning-SimulWhisper | MIT | no | Apple Silicon MLX/CoreML port of SimulStreaming. |
| collabora/WhisperLive | https://github.com/collabora/WhisperLive | MIT | client only | WebSocket server plus client. Backends: faster-whisper, TensorRT-LLM, OpenVINO. |
| QuentinFuxa/WhisperLiveKit | https://github.com/QuentinFuxa/WhisperLiveKit | MIT | no | whisper_streaming plus SimulStreaming plus diarization in one Python package. |
| codesdancing/whisper_streaming_web | https://github.com/codesdancing/whisper_streaming_web | MIT | partial | Web wrapper around whisper_streaming. |
| alesaccoia/VoiceStreamAI | https://github.com/alesaccoia/VoiceStreamAI | MIT | client only | WebSocket based real time wrapper. |

### Specialised Whisper forks

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| huggingface/distil-whisper | https://github.com/huggingface/distil-whisper | MIT | yes (ONNX) | Distilled student keeping full encoder, 2 layer decoder. 4.1k stars. |
| m-bain/whisperX | https://github.com/m-bain/whisperX | BSD-2 | no | faster-whisper plus wav2vec2 alignment plus pyannote diarization. ~70x RT on 8 GB GPU. |
| linto-ai/whisper-timestamped | https://github.com/linto-ai/whisper-timestamped | MIT | port-able | DTW word timestamps from cross attention. |
| nyrahealth/CrisperWhisper | https://github.com/nyrahealth/CrisperWhisper | **CC-BY-NC-4.0** | no | Verbatim transcription, tight word timestamps. Non commercial. |
| jianfch/stable-ts | https://github.com/jianfch/stable-ts | MIT | no | Stable timestamps wrapper. Silence suppression, refine, SRT/VTT/ASS export. |
| Systran/faster-distil-whisper-large-v3 | https://huggingface.co/Systran/faster-distil-whisper-large-v3 | MIT | no | CTranslate2 port of distil-large-v3. |
| distil-whisper/distil-large-v3.5 | https://huggingface.co/distil-whisper/distil-large-v3.5 | MIT | yes (ONNX) | Multi format release: PT, ONNX, CT2, GGML. 7.08% OOD WER. |
| distil-whisper/distil-large-v3.5-ONNX | https://huggingface.co/distil-whisper/distil-large-v3.5-ONNX | MIT | yes | ONNX export of the v3.5 student. |
| yinruiqing/pyannote-whisper | https://github.com/yinruiqing/pyannote-whisper | MIT | no | Diarization wrapper. |
| NbAiLab/nb.whisperX | https://github.com/NbAiLab/nb.whisperX | varies | no | Norwegian fine tune. |

---

## 2. Chinese and Asian models

### Alibaba (FunASR, FunAudioLLM, Qwen, 3D-Speaker)

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| modelscope/FunASR | https://github.com/modelscope/FunASR | MIT (toolkit) | yes (via sherpa-onnx) | Paraformer, FSMN-VAD, CT-Punc, timestamp prediction. Model weights have custom MODEL_LICENSE. |
| FunAudioLLM/SenseVoice | https://github.com/FunAudioLLM/SenseVoice | custom model-license | yes (via sherpa-onnx) | 50+ langs, ASR plus LID plus emotion plus event detection. 15x faster than Whisper-Large claim. |
| QwenLM/Qwen-Audio | https://github.com/QwenLM/Qwen-Audio | Tongyi-Qianwen (restricted) | no | 8.4B audio LLM (legacy). |
| QwenLM/Qwen2-Audio | https://github.com/QwenLM/Qwen2-Audio | Apache-2.0 | no | 8.2B audio LLM, voice chat plus analysis modes. |
| QwenLM/Qwen2.5-Omni | https://github.com/QwenLM/Qwen2.5-Omni | Apache-2.0 | no | Text plus image plus audio plus video in, text plus speech out, real time. |
| QwenLM/Qwen3-Omni | https://github.com/QwenLM/Qwen3-Omni | Apache-2.0 | no | 30B-A3B MoE, real time speech. |
| QwenLM/Qwen3-ASR | https://github.com/QwenLM/Qwen3-ASR | Apache-2.0 | not yet | 0.6B and 1.7B ASR distillation, 52 languages, timestamps. SOTA Mandarin. |
| modelscope/3D-Speaker | https://github.com/modelscope/3D-Speaker | Apache-2.0 (toolkit) | yes (CAM++ via sherpa-onnx) | CAM++, ERes2Net, ERes2NetV2 speaker embeddings; multimodal diarization. |

### Tencent

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| Tencent-Hunyuan | https://github.com/Tencent-Hunyuan | varies | no | No open ASR; HunyuanVideo-Foley is multimodal sound generation. |

### ByteDance

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| BytedanceSpeech/seed-tts-eval | https://github.com/BytedanceSpeech/seed-tts-eval | research | no | Eval harness only. Seed-ASR weights are closed. |

### StepFun, Moonshot, Zhipu

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| stepfun-ai/Step-Audio2 | https://github.com/stepfun-ai/Step-Audio2 | Apache-2.0 (mini variants) | no | 8B end to end speech to speech, mini Base, mini Think. |
| MoonshotAI/Kimi-Audio | https://github.com/MoonshotAI/Kimi-Audio | custom Moonshot | no | 7B audio LLM, voice chat plus QA plus captioning plus SER plus SEC. |
| zai-org/GLM-4-Voice | https://github.com/zai-org/GLM-4-Voice | Apache-2.0 (code) | no | 9B bilingual zh/en spoken chatbot, Whisper plus VQ tokenizer plus CosyVoice decoder. |

### Telco / iFlyTek

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| Tele-AI/TeleSpeech-ASR | https://github.com/Tele-AI/TeleSpeech-ASR | Tele-AI (non commercial without approval) | yes (via sherpa-onnx) | 30 Chinese dialects. Commercial use requires written approval. |

### Next gen Kaldi (k2-fsa, Daniel Povey)

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| k2-fsa/k2 | https://github.com/k2-fsa/k2 | Apache-2.0 | no | PyTorch plus FSA / WFST library, GPU beam search. |
| k2-fsa/icefall | https://github.com/k2-fsa/icefall | Apache-2.0 | no | Training recipes for Zipformer, Conformer, RNN-T, CTC. |
| k2-fsa/sherpa | https://github.com/k2-fsa/sherpa | Apache-2.0 | no (libtorch) | Server side framework. |
| k2-fsa/sherpa-onnx | https://github.com/k2-fsa/sherpa-onnx | Apache-2.0 | **yes (WASM)** | The right runtime for Zipformer, Paraformer, SenseVoice, TeleSpeech in a browser. 12 language bindings. NPM `sherpa-onnx`. |
| k2-fsa/sherpa-ncnn | https://github.com/k2-fsa/sherpa-ncnn | Apache-2.0 | yes (WASM) | NCNN (Tencent) deployment, mobile and embedded. |

### PaddleSpeech, WeNet, FireRedASR

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| PaddlePaddle/PaddleSpeech | https://github.com/PaddlePaddle/PaddleSpeech | Apache-2.0 | no | DeepSpeech2, Conformer, Transformer, U2/U2++. PaddlePaddle native. |
| wenet-e2e/wenet | https://github.com/wenet-e2e/wenet | Apache-2.0 | port-able | U2/U2++ unified streaming Conformer. Production ASR. |
| wenet-e2e/wespeaker | https://github.com/wenet-e2e/wespeaker | Apache-2.0 | no | Speaker verification and diarization. |
| wenet-e2e/WenetSpeech | https://github.com/wenet-e2e/WenetSpeech | CC-BY 4.0 (non commercial) | n/a | 22,400 hour Mandarin corpus. |
| ASLP-lab/WenetSpeech-Yue | https://github.com/ASLP-lab/WenetSpeech-Yue | research | n/a | Large scale Cantonese corpus. |
| FireRedTeam/FireRedASR | https://github.com/FireRedTeam/FireRedASR | Apache-2.0 | no | FireRedASR-AED 1.1B, FireRedASR-LLM 8.3B. Mandarin SOTA. |
| FireRedTeam/FireRedASR2S | https://github.com/FireRedTeam/FireRedASR2S | Apache-2.0 | no | VAD plus LID plus punctuation plus ASR all in one. |

### Japanese

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| reazon-research/ReazonSpeech | https://github.com/reazon-research/ReazonSpeech | varies | no | 35k hour Japanese corpus plus NeMo Conformer-Transducer models. |
| kotoba-tech/kotoba-whisper | https://github.com/kotoba-tech/kotoba-whisper | Apache-2.0 | yes (via transformers.js) | Distilled Whisper-large-v3, Japanese, 6.3x faster than large-v3. |
| nu-dialogue/j-moshi | https://github.com/nu-dialogue/j-moshi | research | no | 7B full duplex Japanese spoken dialogue. |
| llm-jp/llm-jp-moshi | https://github.com/llm-jp/llm-jp-moshi | research | no | Variant of J-Moshi. |

### Korean

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| sooftware/kospeech | https://github.com/sooftware/kospeech | Apache-2.0 | no | DeepSpeech2, LAS, Conformer, Jasper. KsponSpeech trained. |
| clovaai/ClovaCall | https://github.com/clovaai/ClovaCall | research | no | Korean dialogue ASR dataset plus baseline LAS. |
| k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16 | https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16 | Apache-2.0 | yes (via sherpa-onnx) | Streaming Korean Zipformer ONNX. |

---

## 3. Western non Whisper

### NVIDIA NeMo

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| NVIDIA-NeMo/NeMo | https://github.com/NVIDIA-NeMo/NeMo | Apache-2.0 | no | Full speech AI toolkit, FastConformer, Conformer, training recipes. |
| nvidia/parakeet-tdt-0.6b-v2 | https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2 | CC-BY-4.0 | via ONNX | English, 0.6B, mean WER 6.05%, RTFx 3,380 on Open ASR LB. |
| nvidia/parakeet-tdt-0.6b-v3 | https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 | CC-BY-4.0 | via ONNX | 25 EU languages, Granary trained, mean WER 6.34%, RTFx 749. |
| nvidia/parakeet-tdt-1.1b | https://huggingface.co/nvidia/parakeet-tdt-1.1b | CC-BY-4.0 | via ONNX | 1.1B TDT, topped leaderboard early 2024. |
| nvidia/parakeet-ctc-1.1b | https://huggingface.co/nvidia/parakeet-ctc-1.1b | CC-BY-4.0 | via ONNX | 1.1B CTC, RTFx 2,793 (throughput king). |
| nvidia/parakeet-rnnt-1.1b | https://huggingface.co/nvidia/parakeet-rnnt-1.1b | CC-BY-4.0 | via ONNX | Predecessor of TDT. |
| nvidia/parakeet-tdt_ctc-1.1b | https://huggingface.co/nvidia/parakeet-tdt_ctc-1.1b | CC-BY-4.0 | via ONNX | Hybrid TDT+CTC head. |
| nvidia/canary-1b-flash | https://huggingface.co/nvidia/canary-1b-flash | CC-BY-4.0 | no | 883M, en/de/fr/es ASR + AST. 1,045 RTFx A100. WER 6.35%. |
| nvidia/canary-180m-flash | https://huggingface.co/nvidia/canary-180m-flash | CC-BY-4.0 | no | 182M, en/de/fr/es. 1,200+ RTFx. |
| nvidia/canary-1b-v2 | https://huggingface.co/nvidia/canary-1b-v2 | CC-BY-4.0 | no | 978M, 25 EU langs + AST. LS-clean 2.18%. |
| nvidia/canary-qwen-2.5b | https://huggingface.co/nvidia/canary-qwen-2.5b | CC-BY-4.0 | no | 2.5B FastConformer + Qwen3-1.7B decoder. 5.63% mean WER (leaderboard #1). |
| FluidInference/parakeet-tdt-0.6b-v3-ov | https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-ov | CC-BY-4.0 | via OpenVINO | Community OpenVINO export. |
| NexaAI/parakeet-tdt-0.6b-v3-npu | https://huggingface.co/NexaAI/parakeet-tdt-0.6b-v3-npu | CC-BY-4.0 | NPU | Community NPU export. |

### Meta

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| facebook/mms-1b-all | https://huggingface.co/facebook/mms-1b-all | **CC-BY-NC-4.0** | possible via ONNX | 1,162 languages ASR. Non commercial. |
| facebook/mms-1b-l1107 | https://huggingface.co/facebook/mms-1b-l1107 | CC-BY-NC-4.0 | possible | Pretrained backbone. |
| facebook/seamless-m4t-v2-large | https://huggingface.co/facebook/seamless-m4t-v2-large | CC-BY-NC-4.0 | no | Speech to speech, text, translation across 100 langs. Non commercial. |
| facebookresearch/seamless_communication | https://github.com/facebookresearch/seamless_communication | CC-BY-NC | no | SeamlessM4T, SeamlessExpressive, SeamlessStreaming. |
| facebook/wav2vec2-large-xlsr-53 | https://huggingface.co/facebook/wav2vec2-large-xlsr-53 | Apache-2.0 | yes (ONNX) | Self supervised backbone used by WhisperX alignment. |

### Mistral

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| mistralai/Voxtral-Mini-3B-2507 | https://huggingface.co/mistralai/Voxtral-Mini-3B-2507 | Apache-2.0 | no | 3B audio LLM, 8 langs, 30 min audio. LS-clean 1.88%. |
| mistralai/Voxtral-Small-24B-2507 | https://huggingface.co/mistralai/Voxtral-Small-24B-2507 | Apache-2.0 | no | 24B audio LLM. Beats Whisper-large-v3. |
| mistralai/Voxtral-Mini-4B-Realtime-2602 | https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602 | Apache-2.0 | no | 4B streaming variant, 240 ms to 2.4 s latency. |

### Kyutai

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| kyutai-labs/moshi | https://github.com/kyutai-labs/moshi | CC-BY-4.0 | no (yet) | Full duplex spoken LLM, 7B scale. |
| kyutai-labs/delayed-streams-modeling | https://github.com/kyutai-labs/delayed-streams-modeling | CC-BY-4.0 | no | DSM framework. |
| kyutai/stt-1b-en_fr | https://huggingface.co/kyutai/stt-1b-en_fr | CC-BY-4.0 | no | 1B streaming ASR, en+fr, 0.5 s delay. |
| kyutai/stt-2.6b-en | https://huggingface.co/kyutai/stt-2.6b-en | CC-BY-4.0 | no | 2.6B en streaming, 2.5 s delay, 6.4% mean WER. |

### UsefulSensors (Moonshine)

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| moonshine-ai/moonshine | https://github.com/moonshine-ai/moonshine | MIT | yes (transformers.js + WebGPU) | Tiny 27M, Base 61M. 5x to 13x faster than Whisper tiny/small. |
| UsefulSensors/moonshine-base | https://huggingface.co/UsefulSensors/moonshine-base | MIT | yes | 61M base. LS-clean 3.38%. |
| UsefulSensors/moonshine-tiny | https://huggingface.co/UsefulSensors/moonshine-tiny | MIT | yes | 27M tiny. |
| UsefulSensors/moonshine-streaming-tiny | https://huggingface.co/UsefulSensors/moonshine-streaming-tiny | MIT | yes | v2 streaming, ergodic encoder, ~75 ms latency. |

### Rev

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| revdotcom/reverb | https://github.com/revdotcom/reverb | **non commercial** | no | Reverb-ASR (600M Conformer) plus Reverb-Diarization. Trained on 200k hours. |
| Revai/reverb-asr | https://huggingface.co/Revai/reverb-asr | non commercial | no | Best in class on Earnings22 long form. |

### IBM

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| ibm-granite/granite-speech-3.3-8b | https://huggingface.co/ibm-granite/granite-speech-3.3-8b | Apache-2.0 | no | 8.4B, en/fr/de/es/pt, LS-clean 1.43%. Open ASR LB #2 at 5.74%. |
| ibm-granite/granite-speech-3.3-2b | https://huggingface.co/ibm-granite/granite-speech-3.3-2b | Apache-2.0 | no | 2B, same languages, smaller. |
| ibm-granite/granite-speech-4.1-2b | https://huggingface.co/ibm-granite/granite-speech-4.1-2b | Apache-2.0 | no | 2B, adds Japanese. Non autoregressive variant available. |

### Microsoft

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| microsoft/Phi-4-multimodal-instruct | https://huggingface.co/microsoft/Phi-4-multimodal-instruct | MIT | no | 5.6B multimodal, en/zh/de/fr/it/ja/es/pt speech, 6.02% mean WER. |
| microsoft/SpeechT5 | https://github.com/microsoft/SpeechT5 | MIT | port-able | Unified encoder decoder for ASR/TTS/S2S/voice conversion. |
| microsoft/wavlm-large | https://huggingface.co/microsoft/wavlm-large | MIT | port-able | Self supervised backbone used by Reverb v2 and others. |

### CMU OWSM

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| espnet/espnet | https://github.com/espnet/espnet | Apache-2.0 | no | Toolkit hosting OWSM. |
| espnet/owsm_v3.1_ebf | https://huggingface.co/espnet/owsm_v3.1_ebf | CC-BY-4.0 | no | 1.02B E-Branchformer, open Whisper style. Beats Whisper-large in 8/9 English tests. |
| espnet/owsm_ctc_v3.1_1B | https://huggingface.co/espnet/owsm_ctc_v3.1_1B | CC-BY-4.0 | no | CTC head version, faster decode. |

### Legacy / edge

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| alphacep/vosk-api | https://github.com/alphacep/vosk-api | Apache-2.0 | yes (vosk-browser WASM) | Kaldi based. 20+ languages, 50 MB per language. Mediocre accuracy but tiny. |
| vosk-browser (npm) | https://www.npmjs.com/package/vosk-browser | Apache-2.0 | yes | Kaldi compiled to WebAssembly, 13 language WASM models. |
| Picovoice Leopard | https://picovoice.ai/platform/leopard/ | commercial | yes (WASM) | On device batch ASR. ~20 MB. |
| Picovoice Cheetah | https://picovoice.ai/platform/cheetah/ | commercial | yes (WASM) | On device streaming ASR. |
| @picovoice/leopard-web | https://www.npmjs.com/package/@picovoice/leopard-web | commercial | yes | Web SDK. |
| @picovoice/cheetah-web | https://www.npmjs.com/package/@picovoice/cheetah-web | commercial | yes | Web SDK. |
| mozilla/DeepSpeech | https://github.com/mozilla/DeepSpeech | MPL-2.0 | yes (legacy) | **Archived November 2021.** Do not use. |
| coqui-ai/STT | https://github.com/coqui-ai/STT | MPL-2.0 | yes (legacy) | **Inactive since late 2023.** Avoid for new builds. |

---

## 4. Browser runtimes and infrastructure

### Inference runtimes

| Name | URL | License | Description |
|---|---|---|---|
| huggingface/transformers.js | https://github.com/huggingface/transformers.js | Apache-2.0 | JS port of Transformers, WebGPU and WASM. v3 added WebGPU. |
| microsoft/onnxruntime | https://github.com/microsoft/onnxruntime | MIT | ONNX Runtime including Web bindings (`onnxruntime-web`). |
| mlc-ai/web-llm | https://github.com/mlc-ai/web-llm | Apache-2.0 | WebLLM (LLM only, not ASR). |
| mybigday/whisper.rn | https://github.com/mybigday/whisper.rn | MIT | React Native binding of whisper.cpp via JSI. Mobile only. |

### Demos and examples

| Name | URL | License | Description |
|---|---|---|---|
| huggingface/transformers.js-examples | https://github.com/huggingface/transformers.js-examples | Apache-2.0 | Includes `realtime-whisper-webgpu` and `moonshine-web` examples. |
| Xenova/whisper-web (HF Space) | https://huggingface.co/spaces/Xenova/whisper-web | Apache-2.0 | Vanilla file upload demo. |
| Xenova/realtime-whisper-webgpu (HF Space) | https://huggingface.co/spaces/Xenova/realtime-whisper-webgpu | Apache-2.0 | Real time microphone, fully client side, multilingual. |
| Xenova/whisper-webgpu (HF Space) | https://huggingface.co/spaces/Xenova/whisper-webgpu | Apache-2.0 | WebGPU file demo. |
| ggml.ai whisper.cpp WASM | https://ggml.ai/whisper.cpp/stream.wasm/ | MIT | Live microphone whisper.cpp in browser. |
| ggml.ai whisper.cpp bench | https://ggml.ai/whisper.cpp/bench.wasm/ | MIT | WASM benchmarking demo. |
| sanchit-gandhi/whisper-jax (HF Space) | https://huggingface.co/spaces/sanchit-gandhi/whisper-jax | Apache-2.0 | JAX/TPU demo. |
| lmz/candle-whisper (HF Space) | https://huggingface.co/spaces/lmz/candle-whisper | Apache-2.0 | Candle/Rust Whisper running entirely in browser. |
| Moonshine Web | https://github.com/huggingface/transformers.js-examples/tree/main/moonshine-web | Apache-2.0 | Moonshine via transformers.js. |

### Audio capture and resampling

| Name | URL | License | Description |
|---|---|---|---|
| aolsenjazz/libsamplerate-js | https://github.com/aolsenjazz/libsamplerate-js | MIT | libsamplerate WASM port, browser resampling. |
| rochars/wave-resampler | https://github.com/rochars/wave-resampler | MIT | Pure JS resampler. |

### VAD and diarization

| Name | URL | License | Browser | Description |
|---|---|---|---|---|
| snakers4/silero-vad | https://github.com/snakers4/silero-vad | Apache-2.0 | yes (via @ricky0123/vad-web) | Tiny (~1 MB) neural VAD, 100+ langs. |
| ricky0123/vad | https://github.com/ricky0123/vad | MIT | yes | Browser wrapper around Silero VAD plus AudioWorklet. npm `@ricky0123/vad-web`. |
| wiseman/py-webrtcvad | https://github.com/wiseman/py-webrtcvad | MIT | port-able | Python binding of Google WebRTC VAD. Legacy but tiny. |
| ten-framework/ten-vad | https://github.com/ten-framework/ten-vad | varies | yes (WASM) | Lower offset latency than Silero. Frame level. |
| pyannote/pyannote-audio | https://github.com/pyannote/pyannote-audio | MIT (code) / model varies | no | The de facto diarization toolkit. |
| pyannote/speaker-diarization-3.1 | https://huggingface.co/pyannote/speaker-diarization-3.1 | model-specific | no | v3.1 pipeline, powerset segmentation. |
| MahmoudAshraf97/ctc-forced-aligner | https://github.com/MahmoudAshraf97/ctc-forced-aligner | MIT | yes (ONNX) | Meta MMS forced alignment for 1000+ langs. SRT and WebVTT output. |
| speechbrain/spkrec-ecapa-voxceleb | https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb | Apache-2.0 | no | ECAPA-TDNN speaker embedding. |
| NVIDIA TitaNet large | https://huggingface.co/nvidia/speakerverification_en_titanet_large | CC-BY-4.0 | no | Speaker verification, ContextNet backbone. |

---

## 5. Speech datasets

| Name | URL | License | Description |
|---|---|---|---|
| LibriSpeech | https://www.openslr.org/12/ | CC-BY-4.0 | 1,000 hours read English audiobooks. |
| Common Voice | https://commonvoice.mozilla.org/ | CC0 | Crowd sourced, 100+ languages. |
| FLEURS | https://huggingface.co/datasets/google/fleurs | CC-BY-4.0 | 102 langs, FLoRes-101 parallel speech. |
| GigaSpeech | https://github.com/SpeechColab/GigaSpeech | varies | 10k hour English, audiobooks/podcasts/YouTube. |
| GigaSpeech 2 | https://github.com/SpeechColab/GigaSpeech2 | research | 30k hours Thai, Indonesian, Vietnamese. |
| People's Speech | https://huggingface.co/datasets/MLCommons/peoples_speech | CC-BY-SA | 30,000 hour commercial use English. |
| WenetSpeech | https://github.com/wenet-e2e/WenetSpeech | CC-BY-NC-4.0 (audio) | 22,400 hour Mandarin. |
| WenetSpeech-Yue | https://github.com/ASLP-lab/WenetSpeech-Yue | research | 20,000+ hour Cantonese. |
| AISHELL-1 | https://www.openslr.org/33/ | Apache-2.0 | 178 hour Mandarin, 400 speakers. |
| AISHELL-2 | http://www.aishelltech.com/aishell_2 | research/free | 1,000 hour Mandarin. |
| AISHELL-3 | https://www.openslr.org/93/ | Apache-2.0 | 85 hour multispeaker TTS Mandarin. |
| KeSpeech | https://github.com/KeSpeech/KeSpeech | academic-free | 1,542 hour Mandarin + 8 subdialects. |
| ReazonSpeech | https://github.com/reazon-research/ReazonSpeech | varies | 35,000+ hour Japanese. |
| KsponSpeech | AI-Hub Korea | AI-Hub terms | 1,000 hour Korean. |
| VoxLingua107 | https://bark.phon.ioc.ee/voxlingua107/ | CC-BY-4.0 | 6,628 hour 107 language LID. |
| AMI Meeting Corpus | https://groups.inf.ed.ac.uk/ami/corpus/ | CC-BY-4.0 | 100 hour multimodal meeting audio. |
| Granary (NVIDIA) | release Aug 2025 | CC-BY-4.0 | 1M hour, 25 EU languages, ASR + AST. |
| J-CHAT | research release | research | 69,000 hour Japanese dialogue. |

---

## 6. Benchmarks and leaderboards

| Name | URL | Description |
|---|---|---|
| Hugging Face Open ASR Leaderboard | https://huggingface.co/spaces/hf-audio/open_asr_leaderboard | The canonical benchmark. 60+ models, 11 datasets, 3 tracks. |
| Open ASR Leaderboard GitHub | https://github.com/huggingface/open_asr_leaderboard | Reproduction code. |
| Open ASR Leaderboard paper | https://arxiv.org/abs/2510.06961 | "Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual and Long-Form Speech Recognition Evaluation". |
| HF blog post | https://huggingface.co/blog/open-asr-leaderboard | Human readable summary. |
| Northflank "Best Open Source STT in 2026" | https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks | Third party benchmark digest. |
| Slator leaderboard summary | https://slator.com/nvidia-microsoft-elevenlabs-top-automatic-speech-recognition-leaderboard/ | Industry view. |
| VoicePing offline benchmark (16 models) | https://voiceping.net/en/blog/research-offline-speech-transcription-benchmark/ | Independent comparison. |

---

## 7. Foundational / supporting code

| Name | URL | License | Description |
|---|---|---|---|
| OpenNMT/CTranslate2 | https://github.com/OpenNMT/CTranslate2 | MIT | C++ Transformer inference engine. Powers faster-whisper. |
| huggingface/optimum | https://github.com/huggingface/optimum | Apache-2.0 | ONNX export plus optimization for HF models. |
| onnx/onnx | https://github.com/onnx/onnx | Apache-2.0 | Open Neural Network Exchange spec. |
| NVIDIA/NeMo-text-processing | https://github.com/NVIDIA/NeMo-text-processing | Apache-2.0 | Text normalization and ITN, WFST grammars. |
| whisper-normalizer (PyPI) | https://pypi.org/project/whisper-normalizer/ | MIT | Whisper text normalizer for WER computation. |
| felflare/bert-restore-punctuation | https://huggingface.co/felflare/bert-restore-punctuation | MIT | BERT based punctuation restoration. |

---

## 8. Commercial reference (closed, cannot ship offline)

| Name | URL | Description |
|---|---|---|
| AssemblyAI Universal-2 | https://www.assemblyai.com/universal-2 | Closed. Sub 7% WER. 99+ langs. 300 ms P50 streaming latency. |
| Deepgram Nova-3 | https://deepgram.com/learn/introducing-nova-3-speech-to-text-api | Closed. WebSocket streaming. Multilingual code switching. |
| Speechmatics Ursa 2 | https://www.speechmatics.com/company/articles-and-news/ursa-2-elevating-speech-recognition-across-52-languages | Closed. On device container deployment available. |
| Google USM | https://research.google/blog/universal-speech-model-usm-state-of-the-art-speech-ai-for-100-languages/ | Closed. 2B Conformer, 12M hours unlabeled, 300+ langs. |

---

## 9. Total inventory

Counted at write time, 2026-05-16:

- Whisper ecosystem repos and weights: 40+
- Chinese and Asian repos and weights: 35+
- Western non Whisper repos and weights: 50+
- Browser runtimes, demos, and tooling: 25+
- VAD and diarization: 15+
- Datasets: 17
- Total catalogued: 180+ public artefacts.
