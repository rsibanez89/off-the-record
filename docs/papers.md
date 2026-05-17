# Papers: Annotated Bibliography of Modern ASR

> Curated bibliography of papers and whitepapers underpinning the state of the art in offline speech recognition. Each entry: title, lead authors, year, primary URL, short description, and relevance to Off The Record (browser based Whisper + LocalAgreement-2).
> Last updated 2026-05-16. Working notes in `research-papers.md`.

## Reading order for the impatient

If you read only five papers:

1. **Whisper** (Radford et al. 2022): https://arxiv.org/abs/2212.04356. The model Off The Record runs.
2. **LocalAgreement-2** (Machacek, Dabre, Bojar 2023): https://arxiv.org/abs/2307.14743. The streaming algorithm in the live panel.
3. **Distil-Whisper** (Gandhi, von Platen, Rush 2023): https://arxiv.org/abs/2311.00430. Smaller faster student, formalises speculative decoding for Whisper.
4. **Moonshine** (Jeffries et al. 2024): https://arxiv.org/abs/2410.15608. Edge first ASR designed for browsers.
5. **Conformer** (Gulati et al. 2020): https://arxiv.org/abs/2005.08100. The encoder family underneath every non Whisper SOTA model.

---

## 1. Foundations (2006 to 2020)

### Connectionist Temporal Classification (CTC)
- Title: Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks
- Authors: Alex Graves, Santiago Fernandez, Faustino Gomez, Jurgen Schmidhuber
- Year: 2006 (ICML)
- URL: https://www.cs.toronto.edu/~graves/icml_2006.pdf
- Summary: Introduces a loss that lets an RNN map a frame sequence to a shorter label sequence without prior alignment, by marginalising over all valid frame to label alignments via a blank symbol. Made end to end ASR possible.
- Relevance: CTC is half of modern ASR's output families (the other being autoregressive seq2seq, which Whisper uses). Underlies wav2vec 2.0, Paraformer, Zipformer-CTC, WhisperX alignment.

### RNN Transducer (RNN-T)
- Title: Sequence Transduction with Recurrent Neural Networks
- Author: Alex Graves
- Year: 2012
- URL: https://arxiv.org/abs/1211.3711
- Summary: Encoder over audio plus prediction network over previously emitted labels plus joiner. Native streaming with monotonic alignment.
- Relevance: Canonical streaming architecture. Substrate for TDT (NVIDIA Parakeet), U2 (WeNet), every on device mobile ASR.

### Listen, Attend and Spell (LAS)
- Title: Listen, Attend and Spell
- Authors: William Chan, Navdeep Jaitly, Quoc V. Le, Oriol Vinyals
- Year: 2015
- URL: https://arxiv.org/abs/1508.01211
- Summary: First fully neural seq2seq ASR. Pyramidal BLSTM encoder plus attention based RNN decoder. Removed the need for CTC, HMMs, lexicons.
- Relevance: Direct ancestor of Whisper's encoder decoder Transformer. Hallucination on long form audio is inherited from LAS; LocalAgreement-2 was designed to mitigate it.

### Deep Speech 2
- Title: Deep Speech 2: End to End Speech Recognition in English and Mandarin
- Authors: Dario Amodei et al. (Baidu)
- Year: 2015
- URL: https://arxiv.org/abs/1512.02595
- Summary: Scaled end to end CTC ASR with HPC tooling, matched human transcribers, replaced hand engineered pipelines.
- Relevance: Historical baseline. Antecedent of Whisper's "data plus scale" approach.

### Transformer
- Title: Attention Is All You Need
- Authors: Ashish Vaswani et al.
- Year: 2017
- URL: https://arxiv.org/abs/1706.03762
- Summary: Pure self attention plus feed forward, no recurrence, massive parallelism. Basis of every modern foundation model.
- Relevance: Whisper encoder and decoder are direct Transformers. Cross attention weights are what `return_timestamps: "word"` uses for DTW timestamps.

### Streaming On-Device E2E ASR
- Title: Streaming End to End Speech Recognition For Mobile Devices
- Authors: Yanzhang He et al. (Google)
- Year: 2018
- URL: https://arxiv.org/abs/1811.06621
- Summary: First production RNN-T shipped on device (Pixel keyboard). Engineering co-design for live mobile ASR.
- Relevance: Canonical reference for "ASR in your pocket". Engineering trade offs (quantisation, partial result emission, decoder pruning) apply to browser too.

### Conformer
- Title: Conformer: Convolution augmented Transformer for Speech Recognition
- Authors: Anmol Gulati et al. (Google)
- Year: 2020 (Interspeech)
- URL: https://arxiv.org/abs/2005.08100
- Summary: Combines self attention (global) with depthwise convolutions (local) in a macaron feed forward block. New SOTA on LibriSpeech.
- Relevance: Direct ancestor of FastConformer, Branchformer, Zipformer, Squeezeformer. Underpins every non Whisper SOTA.

---

## 2. Self supervised audio (2020 to 2022)

### wav2vec 2.0
- Title: wav2vec 2.0: A Framework for Self Supervised Learning of Speech Representations
- Authors: Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli (Meta)
- Year: 2020 (NeurIPS)
- URL: https://arxiv.org/abs/2006.11477
- Summary: Pre trains a CNN plus Transformer encoder by masking latent speech reps and solving a contrastive task. 10 minutes of labels can match prior 100 hour systems.
- Relevance: Phoneme aligner used by WhisperX. The substrate for Meta MMS.

### HuBERT
- Title: HuBERT: Self Supervised Speech Representation Learning by Masked Prediction of Hidden Units
- Authors: Wei-Ning Hsu et al. (Meta)
- Year: 2021 (IEEE/ACM TASLP)
- URL: https://arxiv.org/abs/2106.07447
- Summary: BERT style masked prediction of offline clustered targets. Matches or beats wav2vec 2.0.
- Relevance: Underpins pyannote 3.x segmentation and many self supervised speech encoders.

### WavLM
- Title: WavLM: Large Scale Self Supervised Pre Training for Full Stack Speech Processing
- Authors: Sanyuan Chen et al. (Microsoft)
- Year: 2022 (IEEE J STSP)
- URL: https://arxiv.org/abs/2110.13900
- Summary: HuBERT scaled to 94k hours with gated relative positional bias plus denoising plus utterance mixing. Strong on non ASR tasks too.
- Relevance: Backbone of pyannote 3.x speaker segmentation and Reverb-Diarization v2 features.

### SpeechT5
- Title: SpeechT5: Unified Modal Encoder Decoder Pre Training for Spoken Language Processing
- Authors: Junyi Ao et al. (Microsoft)
- Year: 2022 (ACL)
- URL: https://arxiv.org/abs/2110.07205
- Summary: T5 style shared encoder decoder with modality specific pre/post nets for ASR, TTS, voice conversion, enhancement.
- Relevance: Context for cross modal speech text pre training. Less directly used than wav2vec / HuBERT / WavLM in production ASR.

---

## 3. Whisper era (2022 to 2025)

### Whisper
- Title: Robust Speech Recognition via Large Scale Weak Supervision
- Authors: Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever (OpenAI)
- Year: 2022 (ICML 2023)
- URL: https://arxiv.org/abs/2212.04356, https://cdn.openai.com/papers/whisper.pdf
- Summary: Encoder decoder Transformer trained on 680,000 hours of weakly supervised multilingual, multitask data (ASR + ST + LID + VAD). Generalises zero shot to most benchmarks.
- Relevance: The core model of Off The Record. Every architectural decision in the app (chunked encoder, 30 s windows, word level timestamps via cross attention, hallucination on silence) traces back here.

### WhisperX
- Title: WhisperX: Time Accurate Speech Transcription of Long Form Audio
- Authors: Max Bain, Jaesung Huh, Tengda Han, Andrew Zisserman
- Year: 2023 (Interspeech)
- URL: https://arxiv.org/abs/2303.00747
- Summary: Wraps Whisper with VAD cut and merge plus wav2vec 2.0 phoneme forced aligner plus pyannote diarization. ~12x batched speedup, ~50 ms word timestamps.
- Relevance: Production grade alternative to a one shot Whisper call. Off The Record's batch panel covers similar ground without server side alignment.

### Whisper Streaming (LocalAgreement-2)
- Title: Turning Whisper into Real Time Transcription System
- Authors: Dominik Machacek, Raj Dabre, Ondrej Bojar
- Year: 2023 (IJCNLP AACL system demonstrations)
- URL: https://arxiv.org/abs/2307.14743
- Summary: LocalAgreement-n streaming policy: re run Whisper on growing audio, commit the longest prefix that two (or n) consecutive hypotheses agree on. ~3.3 s end to end latency on unsegmented long form.
- Relevance: The algorithm that powers Off The Record's live panel. `hypothesisBuffer.ts` is the TypeScript port.

### Distil-Whisper
- Title: Distil-Whisper: Robust Knowledge Distillation via Large Scale Pseudo Labelling
- Authors: Sanchit Gandhi, Patrick von Platen, Alexander M. Rush (Hugging Face)
- Year: 2023
- URL: https://arxiv.org/abs/2311.00430
- Summary: Sequence level KD on 22k hours of pseudo labels. Student keeps full encoder, 2 layer decoder. 5.8x faster, within 1% WER of teacher OOD. Formalises Whisper speculative decoding.
- Relevance: Directly applicable to a faster live panel. Distil-large-v3.5 (March 2025) has ONNX checkpoints and beats Turbo on OOD short form WER (7.08% vs 7.83%).

### CrisperWhisper
- Title: CrisperWhisper: Accurate Timestamps on Verbatim Speech Transcriptions
- Authors: Laurin Wagner, Bernhard Thallinger, Mario Zusag et al.
- Year: 2024 (Interspeech)
- URL: https://arxiv.org/abs/2408.16589
- Summary: Tokenizer adjusted plus fine tuned on verbatim transcripts retaining fillers, stutters, false starts. DTW on cross attention plus custom attention loss for tight word timestamps. Plus hallucination mitigation via 1% noise only training.
- Relevance: Strong for medical and clinical contexts. **CC-BY-NC-4.0**, commercial use blocked.

### Faster-Whisper / CTranslate2
- Authors: SYSTRAN, Guillaume Klein et al.
- Year: 2022 onward
- URL: https://github.com/SYSTRAN/faster-whisper, https://github.com/OpenNMT/CTranslate2
- Summary: C++ Transformer inference with quantisation, batched decoding, KV cache management. ~4x faster than reference PyTorch at the same accuracy.
- Relevance: Reference point for "fast Whisper" outside the browser. Browser analogue is ONNX Runtime Web + transformers.js + WebGPU.

### Whisper-Medusa
- Title: Whisper in Medusa's Ear: Multi head Efficient Decoding for Transformer based ASR
- Authors: Yael Segal-Feldman, Aviv Shamsian, Aviv Navon, Gill Hetz, Joseph Keshet (aiOla)
- Year: 2024
- URL: https://arxiv.org/abs/2409.15869
- Summary: Adds Medusa style auxiliary decoding heads. Multiple future tokens drafted in parallel each step. 1.5x to 5x decode speedup at minimal WER cost.
- Relevance: Decoder side speedup compatible with any Whisper checkpoint. Future direction for sub LocalAgreement-2 latency.

### WhisperKit
- Title: WhisperKit: On Device Real Time ASR with Billion Scale Transformers
- Authors: Atila Orhon et al. (Argmax)
- Year: 2025 (ICML 2025 On Device Workshop)
- URL: https://arxiv.org/abs/2507.10860
- Summary: Production on device Whisper Turbo for Apple Silicon. Causal encoder reshape plus partial input decoder plus native ANE kernels. 0.46 s end to end latency, 2.2% WER LibriSpeech.
- Relevance: Closest real implementation of what Off The Record aims for (fully on device, billion parameter, streaming Whisper). WebGPU and WASM are the browser analogues of ANE.

### CarelessWhisper
- Title: CarelessWhisper: Turning Whisper into a Causal Streaming Model
- Year: 2025
- URL: https://arxiv.org/abs/2508.12301
- Summary: LoRA fine tunes Whisper's bidirectional encoder into a causal/streaming encoder. Eliminates the need for full 30 s windows for low latency.
- Relevance: Future direction. Avoids the re decoding overhead of LocalAgreement entirely. Needs retraining.

### SimulStreaming (IWSLT 2025)
- Title: Simultaneous Translation with Offline Speech and LLM Models
- Authors: CUNI / Machacek et al.
- Year: 2025
- URL: https://arxiv.org/abs/2506.17077
- Summary: Successor to whisper_streaming. AlignAtt policy instead of LocalAgreement. ~5x faster. Winning entry IWSLT 2025 Simultaneous Speech Translation Shared Task.
- Relevance: The natural evolution of LocalAgreement-2 for Off The Record's live panel. Requires routing cross attention.

### Two-Pass Decoding (2025)
- Title: Adapting Whisper for Streaming Speech Recognition via Two Pass Decoding
- Year: 2025
- URL: https://arxiv.org/abs/2506.12154
- Summary: Alternative streaming approach using CTC/Attention two pass.
- Relevance: Conceptually similar to U2++ but applied to Whisper. Worth tracking.

### Whisper Quantization Analysis (2025)
- Title: Quantization for OpenAI's Whisper Models: A Comparative Analysis
- Year: 2025
- URL: https://arxiv.org/abs/2503.09905
- Summary: Comparative analysis of INT8/INT4 across CTranslate2, llama.cpp, ONNX backends.
- Relevance: Empirical guidance for quantisation choices in browser deployment.

### uDistil-Whisper (2024)
- Title: uDistil-Whisper: Label Free Data Filtering for Knowledge Distillation
- Year: 2024
- URL: https://arxiv.org/abs/2407.01257
- Summary: Label free filtering for distillation.
- Relevance: Background on distillation quality.

### Calm-Whisper (2025)
- Title: Calm-Whisper (hallucination mitigation)
- Year: 2025
- URL: https://arxiv.org/abs/2505.12969
- Summary: Adds head level dropout to "calm down" cross attention heads most responsible for non speech hallucinations.
- Relevance: Future research direction for hallucination handling. Needs model retraining.

### Whisper Hallucinations from Non-Speech Audio
- Year: 2025
- URL: https://arxiv.org/abs/2501.11378
- Summary: Investigation of when and why Whisper hallucinates on non speech inputs.
- Relevance: Empirical background for Off The Record's silence gate plus hallucination heuristics.

### Adopting Whisper for Confidence Estimation (2025)
- URL: https://arxiv.org/abs/2502.13446
- Summary: Trains a calibrator on top of Whisper for per-token confidence.
- Relevance: Alternative to LocalAgreement based commit gating. Less robust at chunk boundaries.

---

## 4. Non Whisper SOTA encoders and decoders

### Paraformer
- Title: Paraformer: Fast and Accurate Parallel Transformer for Non Autoregressive End to End Speech Recognition
- Authors: Zhifu Gao et al. (Alibaba DAMO)
- Year: 2022 (Interspeech)
- URL: https://arxiv.org/abs/2206.08317
- Summary: CIF predictor estimates target length, hidden tokens decoded in parallel, glancing LM refines them. Matches AR Conformer-Transducer at >10x speed.
- Relevance: Flagship non Whisper ASR for Mandarin. Backbone of FunASR.

### FunASR
- Title: FunASR: A Fundamental End to End Speech Recognition Toolkit
- Authors: Zhifu Gao et al. (Alibaba DAMO)
- Year: 2023 (Interspeech)
- URL: https://arxiv.org/abs/2305.11013
- Summary: Open source toolkit packaging Paraformer plus FSMN-VAD plus CT-Transformer punctuation plus timestamps. 60k hours Mandarin.
- Relevance: Reference complete pipeline. Mandarin first counterpoint to Whisper's English centric tokenizer.

### SenseVoice (FunAudioLLM)
- Title: FunAudioLLM: Voice Understanding and Generation Foundation Models
- Authors: SenseVoice / CosyVoice team (Alibaba)
- Year: 2024
- URL: https://arxiv.org/abs/2407.04051
- Summary: SenseVoice-Small: low latency 5 language ASR. SenseVoice-Large: 50+ langs with ~20% WER reduction over Whisper-large-v3 plus emotion plus event in one pass.
- Relevance: SOTA challenger to Whisper for CJK plus emotion plus event detection. Browser deployable via sherpa-onnx.

### Zipformer
- Title: Zipformer: A faster and better encoder for automatic speech recognition
- Authors: Zengwei Yao et al. (k2-fsa / Daniel Povey)
- Year: 2023 (ICLR 2024)
- URL: https://arxiv.org/abs/2310.11230
- Summary: U-Net like multi rate Conformer replacement. BiasNorm replaces LayerNorm. SwooshR/L activations. ScaledAdam optimiser. Faster and more accurate at same params.
- Relevance: Powers sherpa-onnx streaming. Architectural ideas likely to show up in future Whisper distillations.

### FastConformer
- Title: Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition
- Authors: Dima Rekesh et al. (NVIDIA)
- Year: 2023 (ASRU)
- URL: https://arxiv.org/abs/2305.05084
- Summary: 8x stride subsampling plus depthwise separable conv plus simplified blocks. ~2x faster encoder than Conformer. Sliding window attention for 11 hour audio.
- Relevance: Encoder used by NVIDIA Canary and Parakeet.

### NVIDIA Canary
- Title: Less is More: Accurate Speech Recognition and Translation without Web Scale Data
- Authors: Krishna C. Puvvada et al. (NVIDIA)
- Year: 2024 (Interspeech)
- URL: https://arxiv.org/abs/2406.19674
- Summary: FastConformer encoder plus Transformer decoder, only 85k hours training (one tenth of Whisper). Outperforms Whisper-large-v3 on en/es/de/fr.
- Relevance: Evidence that small well curated data beats Whisper scale. Important when evaluating "should we move off Whisper?".

### Canary-1B-v2 and Parakeet-TDT-0.6B-v3
- Title: Canary 1B v2 and Parakeet TDT 0.6B v3: Efficient and High Performance Models for Multilingual ASR and AST
- Authors: NVIDIA NeMo speech team
- Year: 2025
- URL: https://arxiv.org/abs/2509.14128
- Summary: Updated Canary spans 25 EU langs. Parakeet TDT 0.6B v3 first model to break 7% mean WER on Open ASR LB at 600M params.
- Relevance: Current SOTA open weights ASR. Smaller than Whisper-large and may suit browser deployment when ONNX matures.

### TDT (Token and Duration Transducer)
- Title: Efficient Sequence Transduction by Jointly Predicting Tokens and Durations
- Authors: Hainan Xu, Fei Jia, Somshubra Majumdar, He Huang, Shinji Watanabe, Boris Ginsburg (NVIDIA)
- Year: 2023 (ICML)
- URL: https://arxiv.org/abs/2304.06795
- Summary: Generalises RNN-T with a duration head. Up to 2.82x faster decoding than RNN-T at equal or better WER.
- Relevance: Architecture of Parakeet-TDT. The frame skip trick parallels LocalAgreement's "skip what we know".

### U2 and U2++
- Title: U2++: Unified Two pass Bidirectional End to End Model for Speech Recognition
- Authors: Di Wu, Binbin Zhang, Chao Yang et al. (WeNet)
- Year: 2021
- URL: https://arxiv.org/abs/2106.05642
- Summary: Single CTC plus Attention model with both forward and backward decoders. Same checkpoint serves streaming (CTC) and non streaming (attention rescoring). SpecSub augmentation. 5 to 8% WERR over U2.
- Relevance: Architectural blueprint for "one model, two latency modes", mirroring Off The Record's dual panel UX.

### WeNet
- Title: WeNet: Production Oriented Streaming and Non Streaming End to End Speech Recognition Toolkit
- Authors: Binbin Zhang et al.
- Year: 2021
- URL: https://arxiv.org/abs/2102.01547
- Summary: Production focused toolkit, U2 unified arch, runtime quantisation, ONNX/TensorRT export.
- Relevance: Closest spiritual ancestor to "one model, two latency modes".

### WeNet 2.0
- Year: 2022
- URL: https://arxiv.org/abs/2203.15455
- Summary: Refines U2 to U2++ with bidirectional attention decoder.

### Branchformer
- Title: Branchformer: Parallel MLP Attention Architectures
- Authors: Yifan Peng, Siddharth Dalmia, Ian Lane, Shinji Watanabe
- Year: 2022 (ICML)
- URL: https://arxiv.org/abs/2207.02971
- Summary: Splits Conformer block into self attention (global) and convolutional gating MLP (local) parallel branches, merges them.
- Relevance: Backbone of OWSM v3.1.

### E-Branchformer
- Title: E-Branchformer: Branchformer with Enhanced Merging for Speech Recognition
- Authors: Kwangyoun Kim, Felix Wu, Yifan Peng et al.
- Year: 2022 (IEEE SLT)
- URL: https://arxiv.org/abs/2210.00077
- Summary: Branchformer plus depthwise conv merge plus extra pointwise FFN. SOTA LibriSpeech 1.81% / 3.65% WER.
- Relevance: Used by OWSM v3.1, the strongest open data Whisper style model.

### Squeezeformer
- Title: Squeezeformer: An Efficient Transformer for Automatic Speech Recognition
- Authors: Sehoon Kim et al.
- Year: 2022 (NeurIPS)
- URL: https://arxiv.org/abs/2206.00888
- Summary: Temporal U-Net inside an otherwise Conformer block. Beats Conformer-CTC at same FLOPs.
- Relevance: Another encoder option in the Conformer family.

### Zipformer Unified Streaming
- Year: 2025
- URL: https://arxiv.org/abs/2506.14434
- Summary: Unified streaming and non streaming Zipformer.

### NEST (Self-supervised FastConformer)
- Year: 2024
- URL: https://arxiv.org/abs/2408.13106
- Summary: Self supervised pretraining for FastConformer.

---

## 5. Streaming algorithms

### LocalAgreement-n
- Documented under Machacek et al. 2023 (section 3). Formalised at CUNI-KIT IWSLT 2022 as a streaming policy that converts any offline seq2seq into a simultaneous streaming model without retraining.

### AlignAtt
- Title: AlignAtt: Using Attention based Audio Translation Alignments as a Guide for Simultaneous Speech Translation
- Authors: Sara Papi, Marco Turchi, Matteo Negri (FBK)
- Year: 2023 (Interspeech)
- URL: https://arxiv.org/abs/2305.11408
- Summary: Cross attention argmax determines whether to emit or wait. +2 BLEU at 0.5 to 0.8 s lower latency than prior offline trained policies.
- Relevance: Alternative to LocalAgreement-2. Attention based per token. Future evolution of Off The Record's live panel.

### Simul-Whisper
- Title: Simul-Whisper
- Authors: Wang et al.
- Year: 2024 (Interspeech)
- URL: https://arxiv.org/abs/2406.10052
- Summary: Applies AlignAtt to Whisper plus integrate and fire truncation detector. Only 1.46% absolute WER degradation at 1 s chunk size, beats LocalAgreement at small chunks.
- Relevance: Direct streaming alternative for Whisper.

### Delayed Streams Modeling (Kyutai)
- Title: Streaming Sequence to Sequence Learning with Delayed Streams Modeling
- Authors: Neil Zeghidour, Eugene Kharitonov, Manu Orsini et al. (Kyutai)
- Year: 2025
- URL: https://arxiv.org/abs/2509.08753
- Summary: Decoder only LM over time aligned audio plus text streams with controllable per stream delays. Powers Kyutai STT 1B (0.5 s delay) and 2.6B (2.5 s delay).
- Relevance: Cleaner, more principled streaming framework. Would replace LocalAgreement if browser native Moshi class models ship.

### Moshi
- Title: Moshi: a speech-text foundation model for real-time dialogue
- Authors: Kyutai
- Year: 2024
- URL: https://arxiv.org/abs/2410.00037
- Summary: Full duplex spoken LLM with Mimi codec encoder and 7B scale decoder. ~200 ms latency.
- Relevance: Foundation of Kyutai STT.

### Monotonic Chunkwise Attention (MoChA)
- Authors: Chiu and Raffel
- Year: ICLR 2018
- URL: https://arxiv.org/abs/1712.05382
- Summary: Hard monotonic pointer plus soft attention over a small chunk to the left. Linear time decode, trainable via expectation.
- Relevance: Historical alternative to RNN-T for streaming. Less used today.

### Multi-Head MoChA
- Year: 2020
- URL: https://arxiv.org/abs/2005.00205
- Summary: Multi head extension of MoChA.

### CHAT (Chunk-wise Attention Transducer)
- Year: 2026
- URL: https://arxiv.org/abs/2602.24245
- Summary: Cross attention inside each fixed chunk, strictly monotonic at chunk granularity. Up to 6.3% relative WER reduction over plain RNN-T.

### Streaming Transformer Transducer
- Year: 2020
- URL: https://arxiv.org/abs/2010.11395
- Summary: Streaming variant of Transformer Transducer.

### Two-pass Endpoint Detection
- Year: 2024
- URL: https://arxiv.org/abs/2401.08916
- Summary: Train ASR to emit `<EOW>` token, combine with VAD trailing silence. Lower latency endpointing.

### Improving Endpoint Detection in Streaming ASR
- Year: 2025
- URL: https://arxiv.org/abs/2505.17070
- Summary: Joint VAD + ASR with shared audio encoder.

---

## 6. Edge and efficient models

### Moonshine
- Title: Moonshine: Speech Recognition for Live Transcription and Voice Commands
- Authors: Nat Jeffries et al. (Useful Sensors)
- Year: 2024
- URL: https://arxiv.org/abs/2410.15608
- Summary: 27M / 60M / 190M encoder decoder Transformer. RoPE position embeddings. No zero padding (encoder cost scales with actual audio length). 5x less compute than Whisper-tiny at same WER on 10 s clips.
- Relevance: Strongest candidate for Off The Record's live panel as a Whisper replacement. No padding encoder is a far better fit for streaming than Whisper's fixed 30 s window.

### Moonshine v2 (Ergodic Streaming Encoder)
- Year: 2026
- URL: https://arxiv.org/abs/2602.12241
- Summary: Sliding window self attention, no positional encodings, ergodic in time. Standardised 50 Hz features. 13.1x faster than Whisper-small on Apple M3, 148 ms latency.
- Relevance: Native streaming, native browser.

### Flavors of Moonshine
- Year: 2025
- URL: https://arxiv.org/abs/2509.02523
- Summary: Task specialised tiny variants for specific edge use cases.

---

## 7. Multilingual and multimodal

### Meta MMS
- Title: Scaling Speech Technology to 1,000+ Languages
- Authors: Vineel Pratap et al. (Meta)
- Year: 2023
- URL: https://arxiv.org/abs/2305.13516
- Summary: wav2vec 2.0 pretrained on 1,406 languages. Single multilingual ASR spanning 1,107 languages. 4,017 language LID. Halves Whisper WER on 54 FLEURS languages.
- Relevance: Reference for long tail of languages. **CC-BY-NC-4.0**, non commercial.

### SeamlessM4T
- Title: SeamlessM4T: Massively Multilingual and Multimodal Machine Translation
- Authors: Seamless Communication team (Meta)
- Year: 2023
- URL: https://arxiv.org/abs/2308.11596
- Summary: Single 2.3B model for ASR, S2TT, T2TT, S2ST, T2ST across 100 langs. +20% BLEU on FLEURS.
- Relevance: Multilingual reference. CC-BY-NC.

### SeamlessStreaming
- Title: Seamless: Multilingual Expressive and Streaming Speech Translation
- Authors: Seamless Communication team (Meta)
- Year: 2023
- URL: https://arxiv.org/abs/2312.05187
- Summary: First simultaneous ASR/ST across 96 source langs and 36 target speech langs with Efficient Monotonic Multihead Attention (EMMA).
- Relevance: EMMA is an alternative streaming policy worth comparing to LocalAgreement-2 conceptually.

### Google USM
- Title: Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages
- Authors: Yu Zhang et al. (Google)
- Year: 2023
- URL: https://arxiv.org/abs/2303.01037
- Summary: 2B Conformer encoder, 12M hours unlabeled, 300+ languages. 32.7% relative WER reduction vs Whisper across 18 evaluated langs.
- Relevance: Closed model, benchmark target only.

### OWSM v3.1
- Title: OWSM v3.1: Better and Faster Open Whisper Style Speech Models based on E-Branchformer
- Authors: Yifan Peng et al. (CMU)
- Year: 2024 (Interspeech)
- URL: https://arxiv.org/abs/2401.16658
- Summary: Fully open reproduction of Whisper using public data and ESPnet. E-Branchformer encoder. Matches Whisper WER on most languages with 73k hours (vs Whisper's 438k).
- Relevance: Open data / open code Whisper analogue. Backstop if Whisper licensing ever blocks use.

### Voxtral (Mistral)
- Year: 2025
- URL: https://arxiv.org/abs/2507.13264
- Summary: Voxtral-Mini (Ministral 3B backbone) and Voxtral-Small (Mistral Small 3.1 24B backbone). Audio aware LLMs. Beat GPT-4o-mini-Transcribe and Gemini 2.5 Flash.
- Relevance: Apache-2.0 audio LLM. Too large for browser today.

### Phi-4-Multimodal
- Title: Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture of LoRAs
- Authors: Microsoft Phi team
- Year: 2025
- URL: https://arxiv.org/abs/2503.01743
- Summary: 5.6B unifying text, vision, speech via modality specific LoRA adapters. 460M speech LoRA topped Open ASR LB at release.
- Relevance: MIT licensed audio LLM. Too large for browser today.

### Qwen2-Audio
- Title: Qwen2-Audio Technical Report
- Authors: Yunfei Chu et al. (Alibaba)
- Year: 2024
- URL: https://arxiv.org/abs/2407.10759
- Summary: Audio LLM with two interaction modes (voice chat, audio analysis). Outperforms Gemini 1.5 Pro on AIR Bench.
- Relevance: Apache-2.0 Mandarin first audio LLM.

### Step-Audio
- Title: Step-Audio: Unified Understanding and Generation in Intelligent Speech Interaction
- Authors: Step-Audio Team (StepFun)
- Year: 2025
- URL: https://arxiv.org/abs/2502.11946
- Summary: 130B unified speech text multimodal model. Step-Audio TTS 3B for distilled deployment.
- Relevance: Chinese open audio LLM frontier.

### Step-Audio 2
- Year: 2025
- URL: https://arxiv.org/abs/2507.16632
- Summary: Mini variants Apache-2.0, 8B end to end speech to speech.

### Kimi-Audio
- Title: Kimi-Audio Technical Report
- Authors: Kimi Team (Moonshot AI)
- Year: 2025
- URL: https://arxiv.org/abs/2504.18425
- Summary: Continuous feature encoder plus discrete token decoder LLM, 13M hour pretraining, streaming flow matching detokeniser. SOTA on ASR, audio QA, conversation.
- Relevance: Background. Flow matching detokeniser is interesting for future streaming captioning plus voice.

### GLM-4-Voice
- Title: GLM-4-Voice: Towards Intelligent and Human Like End to End Spoken Chatbot
- Authors: Aohan Zeng et al. (Zhipu AI)
- Year: 2024
- URL: https://arxiv.org/abs/2412.02612
- Summary: Bilingual zh/en spoken chatbot, 175 bps single codebook 12.5 Hz tokenizer, GLM-4-9B base.
- Relevance: Reference for end to end speech chatbots.

### Seed-ASR (closed)
- Year: 2024
- URL: https://arxiv.org/abs/2407.04675
- Summary: ByteDance audio conditioned LLM. 20M hours training, Mandarin plus 13 dialects plus 7 foreign languages.
- Relevance: Benchmark only (closed weights). AISHELL-1 CER 1.52%.

### Seed-TTS
- Year: 2024
- URL: https://arxiv.org/abs/2406.02430
- Summary: ByteDance TTS counterpart. Not open.

---

## 8. Speculative decoding

### Speculative Decoding for Whisper
- Reference: Distil-Whisper paper plus Hugging Face blog https://huggingface.co/blog/whisper-speculative-decoding
- Summary: Pair small draft (Distil-Whisper) with full target (Whisper-large-v3). Draft proposes k tokens, target verifies in one forward pass, accept matching prefix. Mathematically identical to greedy target output. ~2x faster.
- Relevance: Free 2x speedup for batch panel when transformers.js exposes multi pipeline draft / verify.

### Token Map Drafting (Model-free Speculative Decoding)
- Year: 2025
- URL: https://arxiv.org/abs/2507.21522
- Summary: Replaces draft model with lookup table of common n-grams keyed by encoder states. Zero extra GPU memory.
- Relevance: Lighter weight speculative path. Browser friendly because only one model.

---

## 9. Voice Activity Detection

### Silero VAD
- Title: Silero VAD: pre trained enterprise grade Voice Activity Detector
- Authors: Silero Team
- Year: 2021 onward (no formal paper)
- URL: https://github.com/snakers4/silero-vad
- Summary: ~1 MB language agnostic neural VAD. Sub-ms per 30 ms chunk on a CPU thread. 8 and 16 kHz. Apache-2.0.
- Relevance: Pragmatic default browser side VAD. `@ricky0123/vad-web` ships it.

### pyannote.audio 3 (segmentation pipeline)
- Title: pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe
- Authors: Herve Bredin
- Year: 2023 (Interspeech)
- URL: https://www.isca-archive.org/interspeech_2023/bredin23_interspeech.pdf
- Summary: v2.1 pipeline documenting segmentation, embedding, agglomerative clustering.

### Powerset Diarization Loss
- Title: Powerset multi class cross entropy loss for neural speaker diarization
- Authors: Alexis Plaquet, Herve Bredin
- Year: 2023 (Interspeech)
- URL: https://arxiv.org/abs/2310.13025
- Summary: Single label classification over powerset of speaker subsets. Removes the detection threshold hyperparameter, improves overlap handling. Powers pyannote 3.0 and 3.1.

---

## 10. Diarization and speaker recognition

### EEND
- Title: End to End Neural Speaker Diarization with Self Attention
- Authors: Yusuke Fujita, Naoyuki Kanda, Shota Horiguchi, Yawen Xue, Kenji Nagamatsu, Shinji Watanabe (Hitachi)
- Year: 2019 (IEEE ASRU)
- URL: https://arxiv.org/abs/1909.06247
- Earlier paper (Interspeech 2019): https://arxiv.org/abs/1909.05952
- Summary: Permutation invariant multi label frame classification with self attention. Handles overlapping speech, unlike clustering pipelines.

### LS-EEND
- Year: 2024
- URL: https://arxiv.org/abs/2410.06670
- Summary: Long form streaming EEND with online attractor extraction. Real practical streaming diarization.

### TitaNet
- Title: TitaNet: Neural Model for speaker representation with 1D Depth wise separable convolutions and global context
- Authors: Nithin Rao Koluguri, Taejin Park, Boris Ginsburg (NVIDIA)
- Year: 2022 (ICASSP)
- URL: https://arxiv.org/abs/2110.04410
- Summary: 1D depthwise separable conv speaker embedding. 0.68% EER VoxCeleb1. DER 1.73% AMI MixHeadset.

### CAM++
- Title: CAM++: A Fast and Efficient Network for Speaker Verification Using Context Aware Masking
- Authors: Hui Wang et al. (Alibaba)
- Year: 2023 (Interspeech)
- URL: https://arxiv.org/abs/2303.00332
- Summary: Densely connected TDNN backbone with context aware masking pooling. Matches ECAPA-TDNN at lower compute.

### ECAPA-TDNN
- Authors: Desplanques et al.
- Year: 2020 (Interspeech)
- URL: https://arxiv.org/abs/2005.07143
- Summary: Res2Net frame layers, multi layer feature aggregation, channel attention. Dominant pre CAM++.

### 3D-Speaker
- Title: 3D-Speaker-Toolkit: An Open Source Toolkit for Multimodal Speaker Verification and Diarization
- Authors: Yafeng Chen, Siqi Zheng et al. (Alibaba)
- Year: 2024
- URL: https://arxiv.org/abs/2403.19971
- Summary: Acoustic plus semantic plus visual diarization, 10k speaker public dataset.

### Reverb Diarization
- Year: 2024
- URL: https://arxiv.org/abs/2410.03930
- Summary: Open weights diarization from Rev.ai. v1 pyannote 3 architecture, v2 WavLM features. 26k hours labeled training.
- Note: **Non commercial** license.

---

## 11. Datasets and benchmarks

### LibriSpeech
- Title: LibriSpeech: An ASR corpus based on public domain audio books
- Authors: Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur
- Year: 2015 (ICASSP)
- URL: https://www.danielpovey.com/files/2015_icassp_librispeech.pdf
- Summary: 1,000 hours 16 kHz English from LibriVox. The most cited ASR benchmark.

### Common Voice
- Authors: Rosana Ardila et al. (Mozilla)
- Year: 2019
- URL: https://arxiv.org/abs/1912.06670
- Summary: Crowd sourced CC0, 100+ languages.

### FLEURS
- Title: FLEURS: Few shot Learning Evaluation of Universal Representations of Speech
- Authors: Alexis Conneau et al. (Google)
- Year: 2022
- URL: https://arxiv.org/abs/2205.12446
- Summary: N-way parallel speech in 102 languages built on FLoRes-101.

### GigaSpeech
- Title: GigaSpeech: An Evolving, Multi domain ASR Corpus with 10,000 Hours of Transcribed Audio
- Authors: Guoguo Chen et al.
- Year: 2021
- URL: https://arxiv.org/abs/2106.06909
- Summary: 10k hours English filtered from audiobooks, podcasts, YouTube.

### GigaSpeech 2
- Year: 2024
- URL: https://arxiv.org/abs/2406.11546
- Summary: 30k+ hours of automatically transcribed Thai, Indonesian, Vietnamese. 25 to 40% WER reduction vs Whisper-large-v3.

### WenetSpeech
- Title: WenetSpeech: A 10000+ Hours Multi domain Mandarin Corpus
- Authors: Binbin Zhang et al. (WeNet)
- Year: 2021
- URL: https://arxiv.org/abs/2110.03370
- Summary: 10k labelled plus 12k weakly labelled plus 10k unlabelled hours of Mandarin.

### WenetSpeech-Yue
- Year: 2025
- URL: https://arxiv.org/abs/2509.03959
- Summary: Large scale Cantonese corpus.

### People's Speech
- Authors: Daniel Galvez et al. (MLCommons)
- Year: 2021 (NeurIPS Datasets and Benchmarks)
- URL: https://arxiv.org/abs/2111.09344
- Summary: 30k hour CC-BY-SA English ASR corpus from open licensed audio plus transcripts.

### AISHELL-1
- Authors: Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, Hao Zheng
- Year: 2017
- URL: https://arxiv.org/abs/1709.05522
- Summary: 170 hours, 400 speakers, Apache-2.0 Mandarin.

### AISHELL-2
- Authors: Jiayu Du, Xingyu Na, Xuechen Liu, Hui Bu
- Year: 2018
- URL: https://arxiv.org/abs/1808.10583
- Summary: 1,000 hours, three test sets (iOS, Android, Mic).

### VoxLingua107
- Authors: Jorgen Valk, Tanel Alumae
- Year: 2020 (IEEE SLT 2021)
- URL: https://arxiv.org/abs/2011.12998
- Summary: 6,628 hours, 107 languages. Standard public spoken LID benchmark.

### AMI Meeting Corpus
- Authors: Jean Carletta et al.
- Year: 2005 (MLMI)
- URL: https://link.springer.com/chapter/10.1007/11677482_3
- Summary: 100 hour multimodal meeting recordings.

### Open ASR Leaderboard (paper)
- Year: 2025
- URL: https://arxiv.org/abs/2510.06961
- Summary: "Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual and Long Form Speech Recognition Evaluation".
- Relevance: Canonical benchmark paper.

---

## 12. Streaming latency metrics

### Average Lagging (AL) and Length Adaptive AL (LAAL)
- Title: Over Generation Cannot Be Rewarded: Length Adaptive Average Lagging for Simultaneous Speech Translation
- Authors: Sara Papi, Marco Gaido, Matteo Negri, Marco Turchi
- Year: 2022 (AutoSimTrans workshop)
- URL: https://arxiv.org/abs/2206.05807
- Summary: AL under counts latency for over generating systems. LAAL fixes this by using the longer of reference and output as denominator.
- Relevance: Primary latency metric for streaming ASR.

### Average Token Delay (ATD)
- Title: Average Token Delay: A Latency Metric for Simultaneous Translation
- Authors: Yasumasa Kano et al.
- Year: 2023 (Interspeech)
- URL: https://www.isca-archive.org/interspeech_2023/kano23_interspeech.pdf, https://arxiv.org/abs/2311.14353
- Summary: Duration aware per token latency metric.

### Computation Aware Latency (CA*)
- Year: 2024
- URL: https://arxiv.org/abs/2410.16011
- Summary: How to fairly compare latency across systems with very different compute costs.

### Better Late Than Never
- Year: 2025
- URL: https://arxiv.org/abs/2509.17349
- Summary: Latency metric evaluation.

---

## 13. Punctuation, capitalization, normalization

### BERT Punctuation Restoration
- Year: 2021
- URL: https://arxiv.org/abs/2101.07343
- Summary: Automatic punctuation restoration with BERT models.

### Fast and Accurate Capitalization and Punctuation
- Year: 2019
- URL: https://arxiv.org/abs/1908.02404
- Summary: Transformer plus chunk merging for streaming punctuation.

### NeMo Inverse Text Normalization
- Year: 2021
- URL: https://arxiv.org/abs/2104.05055
- Summary: WFST grammars, context aware WFSTs with neural LMs, audio based TN, fully neural TN/ITN.

---

## 14. Toolkits with papers

### ESPnet
- Title: ESPnet: End to End Speech Processing Toolkit
- Authors: Shinji Watanabe et al.
- Year: 2018 (Interspeech)
- URL: https://arxiv.org/abs/1804.00015
- Summary: PyTorch / Chainer toolkit for ASR, TTS, ST, SE. Hosts OWSM v3.1.

---

## 15. What was deliberately left out

- **Pure speech translation work** (SimulMT, MultiSimulST) except where it produced a metric (AL, LAAL, ATD).
- **Diffusion ASR** (Drax 2025: https://arxiv.org/abs/2510.04162). Out of scope for browser deployment in 2026.
- **Discrete token ASR**. Out of scope for browser deployment in 2026.
- **CosyVoice and pure TTS work**. Mentioned only where bundled with SenseVoice (FunAudioLLM).
- **Speaker change detection, overlap detection, target speaker ASR**. Not load bearing for a single speaker live transcription app.

## 16. Source verification

Every arXiv ID, author list, and year was cross checked against the canonical arXiv abstract page. Where a paper has no arXiv preprint (CTC 2006, LibriSpeech 2015, AMI 2005, pyannote 2.1 2023), the canonical venue PDF is linked instead. Whisper-Speculative-Decoding has no standalone arXiv paper; the canonical references are the Distil-Whisper paper and the HF blog post. Silero VAD has no formal whitepaper; the GitHub README and JIT model release are the source of record.

## Changelog

- 2026-05-16. Initial bibliography. 70+ entries, every arXiv ID verified.
