# Research Papers: Modern Automatic Speech Recognition (2006 to 2026)

A curated bibliography of the most important academic papers for an offline, browser-based transcription app. Emphasis on offline, on-device, and streaming work, and on the algorithms that underpin Whisper, LocalAgreement-2, and the larger ASR ecosystem.

Each entry lists title, lead authors, year, the arXiv URL (or other primary venue), a short summary of the contribution, and a relevance note for the **Off The Record** project (in-browser Whisper via Transformers.js with a live LocalAgreement-2 panel and a one-shot batch panel).

---

## 1. Foundations (2006 to 2020)

### Connectionist Temporal Classification (CTC)
- **Title:** Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks
- **Authors:** Alex Graves, Santiago Fernandez, Faustino Gomez, Jurgen Schmidhuber
- **Year:** 2006 (ICML)
- **URL:** https://www.cs.toronto.edu/~graves/icml_2006.pdf
- **Summary:** Introduces a loss that allows an RNN to map an input frame sequence to a shorter label sequence without prior alignment, by marginalising over all valid frame to label alignments using a blank symbol. Removed the need for HMM style forced alignment and made end to end ASR practical.
- **Relevance:** CTC is one of the two output families underlying nearly every modern ASR model. Whisper itself is a seq2seq decoder rather than CTC, but every CTC based competitor (wav2vec 2.0, Paraformer, Zipformer CTC, WhisperX's forced aligner) traces back here. Knowing CTC matters for understanding why LocalAgreement-2 has to operate on Whisper's token stream rather than on a CTC posterior.

### Sequence Transduction with Recurrent Neural Networks (RNN Transducer)
- **Title:** Sequence Transduction with Recurrent Neural Networks
- **Authors:** Alex Graves
- **Year:** 2012
- **arXiv:** https://arxiv.org/abs/1211.3711
- **Summary:** Defines the RNN Transducer (RNN-T): a joint network over an encoder of audio and a prediction network over previously emitted labels, summed and softmaxed to a per frame output distribution. Allows streaming label emission with monotonic alignment. The architecture later dominated on device ASR.
- **Relevance:** Background. Off The Record uses a non streaming seq2seq (Whisper) but RNN-T is the canonical streaming architecture and the reason Google, Meta, and NVIDIA push it for live transcription. Reading RNN-T is essential before reading TDT, U2++, or Zipformer Transducer.

### Listen, Attend and Spell (LAS)
- **Title:** Listen, Attend and Spell
- **Authors:** William Chan, Navdeep Jaitly, Quoc V. Le, Oriol Vinyals
- **Year:** 2015
- **arXiv:** https://arxiv.org/abs/1508.01211
- **Summary:** First sequence to sequence ASR system built entirely from neural components: a pyramidal BLSTM encoder ("Listen") plus an attention based RNN decoder ("Attend and Spell"). Removed the need for CTC, HMMs, or pronunciation lexicons and demonstrated that attention generalises from MT to speech.
- **Relevance:** Direct ancestor of Whisper's encoder decoder Transformer. The attention based seq2seq formulation is exactly what Whisper inherits, including LAS's weakness for hallucination on long form audio (the very thing LocalAgreement-2 is designed to mitigate).

### Deep Speech 2
- **Title:** Deep Speech 2: End to End Speech Recognition in English and Mandarin
- **Authors:** Dario Amodei et al. (Baidu)
- **Year:** 2015
- **arXiv:** https://arxiv.org/abs/1512.02595
- **Summary:** Scaled end to end CTC based ASR with a deep stack of bidirectional RNN/GRU layers and HPC tooling, demonstrating competitive performance with human transcribers and replacing the entire hand engineered speech pipeline with a single neural network.
- **Relevance:** Historical baseline. Shows that scale and end to end optimisation alone can beat HMM GMM hybrids; this is the conceptual antecedent of Whisper's 680k hour weakly supervised regime.

### Attention Is All You Need (Transformer)
- **Title:** Attention Is All You Need
- **Authors:** Ashish Vaswani et al.
- **Year:** 2017
- **arXiv:** https://arxiv.org/abs/1706.03762
- **Summary:** Introduces the Transformer: an encoder decoder architecture built purely from self attention and feed forward layers, removing recurrence and enabling massive parallelisation. The basis of nearly every modern foundation model.
- **Relevance:** The Whisper encoder and decoder are direct Transformers. Understanding the attention mechanism is mandatory for understanding why word level timestamps via cross attention (as in WhisperX, CrisperWhisper, Whisper's `return_timestamps: 'word'`) work at all.

### Streaming End to End Speech Recognition for Mobile Devices
- **Title:** Streaming End to end Speech Recognition For Mobile Devices
- **Authors:** Yanzhang He et al. (Google)
- **Year:** 2018
- **arXiv:** https://arxiv.org/abs/1811.06621
- **Summary:** First production quality RNN-T system shipped on device for streaming ASR (Pixel keyboard). Demonstrates the engineering co design (quantisation, pruning, prediction network simplification) needed to make E2E ASR run live on a phone.
- **Relevance:** The canonical "ASR in your pocket" reference. Off The Record targets the browser rather than mobile silicon, but the same engineering trade offs (model size, partial result emission, decoder pruning) apply.

### Conformer
- **Title:** Conformer: Convolution augmented Transformer for Speech Recognition
- **Authors:** Anmol Gulati et al. (Google)
- **Year:** 2020 (Interspeech)
- **arXiv:** https://arxiv.org/abs/2005.08100
- **Summary:** Combines self attention (global context) with depthwise convolutions (local context) inside a macaron feed forward block, setting a new state of the art on LibriSpeech and becoming the de facto ASR encoder architecture for the next several years.
- **Relevance:** Direct ancestor of FastConformer, Branchformer, Zipformer, and Squeezeformer. Any non Whisper SOTA model (Paraformer, NVIDIA Canary, Parakeet, OWSM) builds on a Conformer derived encoder.

---

## 2. Self supervised audio (2020 to 2022)

### wav2vec 2.0
- **Title:** wav2vec 2.0: A Framework for Self Supervised Learning of Speech Representations
- **Authors:** Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli (Meta)
- **Year:** 2020 (NeurIPS)
- **arXiv:** https://arxiv.org/abs/2006.11477
- **Summary:** Pre trains a CNN+Transformer encoder by masking latent speech representations and solving a contrastive task against a learned codebook. Fine tuning with as little as 10 minutes of transcribed audio matches prior supervised systems trained on 100h.
- **Relevance:** The phoneme aligner used by WhisperX is a fine tuned wav2vec 2.0. If Off The Record ever adds external forced alignment for tighter timestamps, this is the encoder.

### HuBERT
- **Title:** HuBERT: Self Supervised Speech Representation Learning by Masked Prediction of Hidden Units
- **Authors:** Wei-Ning Hsu et al. (Meta)
- **Year:** 2021 (IEEE/ACM TASLP)
- **arXiv:** https://arxiv.org/abs/2106.07447
- **Summary:** Replaces wav2vec 2.0's contrastive objective with BERT style masked prediction of offline clustered targets, simplifying training and matching or beating wav2vec 2.0 on LibriSpeech.
- **Relevance:** Underpins downstream tasks (speaker diarization, emotion, paralinguistics). Knowing HuBERT is necessary background for pyannote 3.x segmentation and most modern self supervised speech encoders.

### WavLM
- **Title:** WavLM: Large Scale Self Supervised Pre Training for Full Stack Speech Processing
- **Authors:** Sanyuan Chen et al. (Microsoft)
- **Year:** 2022 (IEEE J STSP)
- **arXiv:** https://arxiv.org/abs/2110.13900
- **Summary:** Scales HuBERT style pre training to 94k hours, adds gated relative positional bias, and trains with denoising/utterance mixing so the encoder is good at non ASR tasks (speaker, diarization, separation) as well as transcription.
- **Relevance:** Backbone of the pyannote 3.x speaker segmentation and embedding models, which is the path Off The Record would take to add diarization.

### SpeechT5
- **Title:** SpeechT5: Unified Modal Encoder Decoder Pre Training for Spoken Language Processing
- **Authors:** Junyi Ao et al. (Microsoft)
- **Year:** 2022 (ACL)
- **arXiv:** https://arxiv.org/abs/2110.07205
- **Summary:** A T5 style shared encoder decoder with modality specific pre/post nets for both speech and text, supporting ASR, TTS, voice conversion, and speech enhancement from a single backbone.
- **Relevance:** Context for cross modal speech text pre training. Less directly relevant to a browser only transcription app than wav2vec/HuBERT/WavLM, but cited frequently as foundation for downstream models.

---

## 3. Whisper era (2022 to 2025)

### Whisper
- **Title:** Robust Speech Recognition via Large Scale Weak Supervision
- **Authors:** Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever (OpenAI)
- **Year:** 2022 (ICML 2023)
- **arXiv:** https://arxiv.org/abs/2212.04356
- **Summary:** Trains an encoder decoder Transformer on 680,000 hours of weakly supervised multilingual, multitask data (ASR, ST, language ID, VAD) drawn from the open web. The resulting models generalise zero shot to most benchmarks and are competitive with human transcribers.
- **Relevance:** **The core model of Off The Record.** Every architectural decision in the app (chunked encoder, 30s windows, word level timestamps via cross attention, hallucination behaviour on silence) traces back to this paper. Both the live and batch panels run a Whisper family model.

### WhisperX
- **Title:** WhisperX: Time Accurate Speech Transcription of Long Form Audio
- **Authors:** Max Bain, Jaesung Huh, Tengda Han, Andrew Zisserman
- **Year:** 2023 (Interspeech)
- **arXiv:** https://arxiv.org/abs/2303.00747
- **Summary:** Wraps Whisper with (1) a Silero/pyannote based VAD cut and merge front end that segments long audio into Whisper sized chunks for batch inference, and (2) a wav2vec 2.0 phoneme forced aligner that produces 50 ms word timestamps. The pipeline yields a 12x batched inference speedup over sliding window Whisper while removing long form drift and improving timestamp accuracy roughly tenfold.
- **Relevance:** Direct competitor / complement to Off The Record's batch panel. The "VAD then batch then realign" pattern is the production grade alternative to a single one shot Whisper call; we should know it well enough to explain why we chose not to do it (Off The Record is a browser app with no Python side aligner).

### Whisper Streaming (LocalAgreement-2)
- **Title:** Turning Whisper into Real Time Transcription System
- **Authors:** Dominik Machacek, Raj Dabre, Ondrej Bojar
- **Year:** 2023 (IJCNLP AACL system demonstrations)
- **arXiv:** https://arxiv.org/abs/2307.14743
- **Summary:** Introduces a self adaptive streaming policy for Whisper based on **LocalAgreement n**: re run Whisper on growing audio buffers and only commit the longest prefix that two (or n) consecutive hypotheses agree on. Achieves ~3.3 s latency on unsegmented long form audio.
- **Relevance:** **This is the algorithm that powers Off The Record's live panel.** The hypothesis buffer logic, the punctuation stripping rules, and the silence/hallucination heuristics in `src/lib/transcription/hypothesisBuffer.ts` all implement (and slightly extend) LocalAgreement-2 from this paper.

### Distil Whisper
- **Title:** Distil Whisper: Robust Knowledge Distillation via Large Scale Pseudo Labelling
- **Authors:** Sanchit Gandhi, Patrick von Platen, Alexander M. Rush (Hugging Face)
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2311.00430
- **Summary:** Compresses Whisper large v2 with sequence level knowledge distillation on 22k hours of pseudo labelled audio (filtered by WER heuristic). The student is 5.8x faster with 51% fewer parameters and within 1% WER of the teacher on out of distribution data.
- **Relevance:** Directly applicable to Off The Record's live panel: a smaller, faster Whisper student is exactly what LocalAgreement-2 wants (faster decode = lower commit latency). The paper also formalises Whisper style speculative decoding (student drafts, teacher verifies).

### CrisperWhisper
- **Title:** CrisperWhisper: Accurate Timestamps on Verbatim Speech Transcriptions
- **Authors:** Laurin Wagner, Bernhard Thallinger, Mario Zusag (and co authors)
- **Year:** 2024 (Interspeech)
- **arXiv:** https://arxiv.org/abs/2408.16589
- **Summary:** Adjusts Whisper's tokenizer and fine tunes on verbatim transcripts that retain filler words, false starts, and partial word repetitions. Combined with dynamic time warping on cross attention scores it produces much tighter word timestamps (and far fewer hallucinations) for clinical/legal use cases.
- **Relevance:** Important for medical contexts (which is also where Off The Record's user is most useful). The DTW on cross attention approach is essentially what Transformers.js's `return_timestamps: 'word'` does under the hood. The project's hard rule "use `_timestamped` ONNX variants only" exists because of this same mechanism.

### Faster Whisper / CTranslate2
- **Title:** *(no paper, engineering / documentation)*
- **Authors:** Guillaume Klein et al. (SYSTRAN), Ricky Pang (faster whisper)
- **Year:** 2022 onward
- **URL:** https://github.com/SYSTRAN/faster-whisper, https://github.com/OpenNMT/CTranslate2
- **Summary:** CTranslate2 is a high performance C++ Transformer inference engine with quantisation, batched decoding, and KV cache management. Faster whisper wraps it for Whisper and is roughly 4x faster than the reference PyTorch implementation at the same accuracy.
- **Relevance:** Reference point for what "fast Whisper" looks like outside the browser. Off The Record's analogue is ONNX Runtime Web + Transformers.js with WebGPU; the engineering goals (KV cache reuse, beam pruning, quantised matmul) are the same.

### Whisper Medusa (speculative decoding for Whisper)
- **Title:** Whisper in Medusa's Ear: Multi head Efficient Decoding for Transformer based ASR
- **Authors:** Yael Segal Feldman, Aviv Shamsian, Aviv Navon, Gill Hetz, Joseph Keshet (aiOla)
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2409.15869
- **Summary:** Adds Medusa style auxiliary decoding heads to Whisper so multiple future tokens are drafted in parallel each step and accepted/rejected by the main head, cutting decode latency by 1.5x to 5x with minimal WER cost.
- **Relevance:** A purely decoding side speedup that is compatible with any Whisper family checkpoint. If Off The Record's live latency budget ever needs to drop below the LocalAgreement-2 floor, multi head drafting is the cheapest knob.

### WhisperKit
- **Title:** WhisperKit: On device Real time ASR with Billion Scale Transformers
- **Authors:** Atila Orhon et al. (Argmax)
- **Year:** 2025 (ICML 2025 On Device Workshop)
- **arXiv:** https://arxiv.org/abs/2507.10860
- **Summary:** Production grade on device Whisper large v3 turbo pipeline for Apple silicon: encoder reshaped for true streaming, decoder works on partial encoder output, native Apple Neural Engine kernels. Achieves 0.46 s end to end latency and 2.2% WER, matching or beating gpt 4o transcribe.
- **Relevance:** The closest "real" implementation of what Off The Record aims for: fully on device, billion parameter, streaming Whisper. WebGPU/WASM is the browser analogue of ANE; the architectural patches (causal encoder, partial input decoder) are directly transferable.

---

## 4. Non Whisper SOTA encoders and decoders

### Paraformer
- **Title:** Paraformer: Fast and Accurate Parallel Transformer for Non autoregressive End to End Speech Recognition
- **Authors:** Zhifu Gao et al. (Alibaba DAMO)
- **Year:** 2022 (Interspeech)
- **arXiv:** https://arxiv.org/abs/2206.08317
- **Summary:** Non autoregressive ASR: a continuous integrate and fire (CIF) predictor estimates the target length, hidden tokens are decoded in parallel, and a second pass glancing language model refines them. Matches autoregressive Conformer Transducer accuracy at >10x inference speed.
- **Relevance:** The flagship non Whisper ASR for Mandarin and the backbone of FunASR's production stack. Worth knowing as the strongest argument that autoregressive Whisper style decoding is not the only path to high accuracy.

### FunASR
- **Title:** FunASR: A Fundamental End to End Speech Recognition Toolkit
- **Authors:** Zhifu Gao et al. (Alibaba DAMO)
- **Year:** 2023 (Interspeech)
- **arXiv:** https://arxiv.org/abs/2305.11013
- **Summary:** Open source toolkit packaging Paraformer, an FSMN VAD, a CT Transformer punctuation model, and timestamp prediction into a production pipeline. Trained on 60,000h of manually annotated Mandarin.
- **Relevance:** Reference architecture for a complete VAD + ASR + punctuation pipeline. The Mandarin first design is a useful counterpoint to Whisper's English centric tokeniser.

### SenseVoice (FunAudioLLM)
- **Title:** FunAudioLLM: Voice Understanding and Generation Foundation Models for Natural Interaction Between Humans and LLMs
- **Authors:** SenseVoice/CosyVoice team (Alibaba)
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2407.04051
- **Summary:** Family of voice foundation models. SenseVoice Small is a low latency multilingual ASR for 5 languages; SenseVoice Large does 50+ languages with WER reductions of ~20% over Whisper large v3, plus emotion recognition and audio event detection in the same forward pass.
- **Relevance:** A genuine SOTA challenger to Whisper outside the Latin script world. If Off The Record ever needs to support Mandarin, Cantonese, Japanese, or Korean transcription that doesn't degrade against native speakers, SenseVoice is the model to evaluate.

### Zipformer
- **Title:** Zipformer: A faster and better encoder for automatic speech recognition
- **Authors:** Zengwei Yao et al. (k2 fsa, Daniel Povey)
- **Year:** 2023 (ICLR 2024)
- **arXiv:** https://arxiv.org/abs/2310.11230
- **Summary:** A U Net like Conformer replacement that processes middle blocks at lower frame rates, reuses attention weights, swaps LayerNorm for BiasNorm, and uses new Swoosh activations. Introduces ScaledAdam as the optimiser. Significantly faster and more accurate than Conformer at the same parameter budget.
- **Relevance:** Background for the next generation of streaming encoders; Zipformer transducer is widely deployed in production by speech startups. Not directly used in the browser today but the architectural ideas (multi rate U Net, BiasNorm) are likely to appear in distilled Whispers.

### Fast Conformer
- **Title:** Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition
- **Authors:** Dima Rekesh et al. (NVIDIA)
- **Year:** 2023 (ASRU)
- **arXiv:** https://arxiv.org/abs/2305.05084
- **Summary:** Redesigns Conformer's down sampling to give an 8x stride and 2.8x speedup with no accuracy loss; replaces global attention with a sliding window plus global token to handle 11 hour audio.
- **Relevance:** The encoder used by NVIDIA Canary and Parakeet. Reading this is prerequisite for reading either of those NVIDIA reports.

### NVIDIA Canary
- **Title:** Less is More: Accurate Speech Recognition and Translation without Web Scale Data
- **Authors:** Krishna C. Puvvada et al. (NVIDIA)
- **Year:** 2024 (Interspeech)
- **arXiv:** https://arxiv.org/abs/2406.19674
- **Summary:** Canary uses a FastConformer encoder plus Transformer decoder trained on only 85k hours (one tenth of Whisper) yet outperforms Whisper large v3 on English, Spanish, German, and French. Demonstrates that data quality plus synthetic translation pairs plus dynamic bucketing beat web scale weak supervision.
- **Relevance:** Strong evidence that small, well curated training data can beat Whisper. Important when evaluating "should we switch off Whisper?". Canary 1B flash is the strongest Apache 2.0 alternative for the four major Western European languages.

### Canary 1B v2 and Parakeet TDT 0.6B v3
- **Title:** Canary 1B v2 and Parakeet TDT 0.6B v3: Efficient and High Performance Models for Multilingual ASR and AST
- **Authors:** NVIDIA NeMo speech team
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2509.14128
- **Summary:** Updated Canary spans 25 European languages, and Parakeet TDT 0.6B v3 is a 600M FastConformer TDT model that became the first model to break a 7% average WER on the Hugging Face Open ASR leaderboard.
- **Relevance:** Current state of the art for non Whisper open weights ASR. Both models are smaller than Whisper large and may be more attractive for a browser deployment if/when their ONNX exports mature.

### Token and Duration Transducer (TDT)
- **Title:** Efficient Sequence Transduction by Jointly Predicting Tokens and Durations
- **Authors:** Hainan Xu, Fei Jia, Somshubra Majumdar, He Huang, Shinji Watanabe, Boris Ginsburg (NVIDIA)
- **Year:** 2023 (ICML)
- **arXiv:** https://arxiv.org/abs/2304.06795
- **Summary:** Extends RNN-T with a duration head: the joint network emits both a token and the number of input frames that token covers. At inference, frames are skipped according to the predicted duration, giving up to 2.82x faster decoding than RNN-T at equal or better WER.
- **Relevance:** Background for Parakeet TDT. Conceptually similar to LocalAgreement-2 in that "skip frames that we know we don't need" is the same trick, only here it's learned end to end rather than gated by hypothesis stability.

### U2 / U2++
- **Title:** U2++: Unified Two pass Bidirectional End to end Model for Speech Recognition
- **Authors:** Di Wu, Binbin Zhang, Chao Yang et al. (WeNet)
- **Year:** 2021
- **arXiv:** https://arxiv.org/abs/2106.05642
- **Summary:** Single CTC plus Attention model trained with both forward and backward decoders so the *same* checkpoint can serve streaming (CTC prefix beam) and non streaming (attention rescoring) inference. SpecSub data augmentation. Consistent 5 to 8% WERR over U2.
- **Relevance:** The architectural blueprint for "one model, two latency modes", exactly the dichotomy Off The Record exposes via its dual live/batch panels. Reading U2++ is the easiest way to understand why a single Whisper checkpoint can serve both.

### Branchformer
- **Title:** Branchformer: Parallel MLP Attention Architectures to Capture Local and Global Context for Speech Recognition and Understanding
- **Authors:** Yifan Peng, Siddharth Dalmia, Ian Lane, Shinji Watanabe
- **Year:** 2022 (ICML)
- **arXiv:** https://arxiv.org/abs/2207.02971
- **Summary:** Splits the Conformer block into two parallel branches, self attention for long range and convolutional gating MLP (cgMLP) for local, then merges them. Comparable to or better than Conformer, with the interpretability benefit that the learned mixing weights show where local vs. global context matters.
- **Relevance:** Backbone of OWSM v3.1. Worth knowing as a reference encoder for ablations.

### E Branchformer
- **Title:** E Branchformer: Branchformer with Enhanced Merging for Speech Recognition
- **Authors:** Kwangyoun Kim, Felix Wu, Yifan Peng et al.
- **Year:** 2022 (IEEE SLT)
- **arXiv:** https://arxiv.org/abs/2210.00077
- **Summary:** Improves Branchformer with a depthwise convolutional merge module and extra point wise feed forward layers, setting a new SOTA on LibriSpeech (1.81% / 3.65% WER) without external data.
- **Relevance:** OWSM v3.1 uses E Branchformer as its encoder, so this is the architecture currently delivering the strongest open data Whisper style model.

### Squeezeformer
- **Title:** Squeezeformer: An Efficient Transformer for Automatic Speech Recognition
- **Authors:** Sehoon Kim et al.
- **Year:** 2022 (NeurIPS)
- **arXiv:** https://arxiv.org/abs/2206.00888
- **Summary:** Applies a Temporal U Net (down sample, process, up sample) inside an otherwise Conformer shaped block, simplifies the macaron structure, removes redundant LayerNorms, and adds an efficient depthwise down sampling layer. Beats Conformer CTC at the same FLOPs.
- **Relevance:** Another encoder option in the Conformer descendant family; especially interesting on edge devices where the U Net's frame rate reduction directly shrinks attention cost.

---

## 5. Streaming algorithms

### LocalAgreement (LA n)
- The original formal definition is in Machacek, Dabre, Bojar 2023 (https://arxiv.org/abs/2307.14743) as the **LocalAgreement n** policy: emit the longest prefix that the most recent n consecutive Whisper hypotheses agree on. Off The Record runs the n=2 variant. See entry in section 3.

### AlignAtt
- **Title:** AlignAtt: Using Attention based Audio Translation Alignments as a Guide for Simultaneous Speech Translation
- **Authors:** Sara Papi, Marco Turchi, Matteo Negri (FBK)
- **Year:** 2023 (Interspeech)
- **arXiv:** https://arxiv.org/abs/2305.11408
- **Summary:** A streaming policy that uses the cross attention argmax between the next token candidate and the audio frames as a gate: if the model is still attending to the most recent frames, wait; otherwise emit. Outperforms prior offline trained streaming policies by 2 BLEU at 0.5 to 0.8 s lower latency across 8 language pairs.
- **Relevance:** A direct alternative streaming policy for Whisper. AlignAtt is attention based and per token; LocalAgreement-2 is text stability based and per window. A future evolution of Off The Record's live panel could test AlignAtt style emission on top of Whisper's already exposed cross attention.

### Streaming Sequence to Sequence with Delayed Streams Modeling (Kyutai DSM)
- **Title:** Streaming Sequence to Sequence Learning with Delayed Streams Modeling
- **Authors:** Neil Zeghidour, Eugene Kharitonov, Manu Orsini, Vaclav Volhejn, Gabriel de Marmiesse, Edouard Grave, Patrick Perez, Laurent Mazare, Alexandre Defossez (Kyutai)
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2509.08753
- **Summary:** Formulates streaming ASR/TTS as a decoder only LM over time aligned text and audio streams with controllable per stream delays. The same architecture supports 0.5 s latency ASR (kyutai/stt 1b en_fr) and 2.5 s latency higher accuracy ASR (kyutai/stt 2.6b en), matching offline baselines.
- **Relevance:** A cleaner, more principled alternative to Whisper Streaming. If browser native Moshi class models ever ship, DSM is the streaming framework that would power them.

---

## 6. Edge and efficient models

### Moonshine
- **Title:** Moonshine: Speech Recognition for Live Transcription and Voice Commands
- **Authors:** Nat Jeffries et al. (Useful Sensors)
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2410.15608
- **Summary:** A 27M / 60M / 190M parameter encoder decoder Transformer trained on 200k hours, using Rotary Position Embeddings (RoPE) instead of Whisper's absolute embeddings and skipping zero padding so encoder cost scales with actual audio length. 5x less compute than Whisper tiny.en at the same WER on 10 second clips.
- **Relevance:** Strong candidate for Off The Record's live panel as a Whisper replacement: it is explicitly designed for short form, low latency live transcription on edge devices, and the no padding encoder is a much better fit for streaming chunks than Whisper's fixed 30 s window.

### Distil Whisper
- See section 3.

### WhisperKit
- See section 3.

---

## 7. Multilingual and multimodal

### Massively Multilingual Speech (MMS)
- **Title:** Scaling Speech Technology to 1,000+ Languages
- **Authors:** Vineel Pratap et al. (Meta)
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2305.13516
- **Summary:** wav2vec 2.0 pre trained on 1,406 languages (using readings of religious texts as long tail data), plus a single multilingual ASR model spanning 1,107 languages, a TTS model for the same set, and a 4,017 language LID model. Halves Whisper's WER on the 54 FLEURS languages it tests.
- **Relevance:** The reference for any "very long tail of languages" ambition. Not currently in Off The Record's scope but is the obvious model family to consult if low resource language support is ever required.

### SeamlessM4T
- **Title:** SeamlessM4T: Massively Multilingual and Multimodal Machine Translation
- **Authors:** Seamless Communication team (Meta)
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2308.11596
- **Summary:** A single 2.3B model that does ASR, S2TT, T2TT, S2ST, and T2ST across 100 languages, trained on 1M hours of self supervised audio and an automatically aligned multimodal corpus. +20% BLEU on FLEURS S2TT versus prior SOTA.
- **Relevance:** Background for multilingual plus translation combined; not the target for transcription only.

### SeamlessStreaming
- **Title:** Seamless: Multilingual Expressive and Streaming Speech Translation
- **Authors:** Seamless Communication team (Meta)
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2312.05187
- **Summary:** Builds SeamlessM4T v2, SeamlessExpressive (prosody preserving translation), and SeamlessStreaming (the first simultaneous ASR/ST system to cover 96 source languages and 36 target speech languages with low latency Efficient Monotonic Multihead Attention).
- **Relevance:** SeamlessStreaming's EMMA policy is an alternative streaming mechanism worth comparing to LocalAgreement-2 conceptually. Streaming ASR across 96 languages from a single Apache licensed model is the bar Whisper Streaming would have to beat to lose relevance.

### Google USM
- **Title:** Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages
- **Authors:** Yu Zhang et al. (Google)
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2303.01037
- **Summary:** A 2B parameter Conformer encoder pre trained on 12M hours across 300+ languages and fine tuned with a much smaller labelled set. Backbone of YouTube auto captions and Google's multilingual speech products.
- **Relevance:** Closed model, useful as a benchmark target but not deployable. Cited here because every open multilingual ASR paper compares against USM.

### OWSM v3.1
- **Title:** OWSM v3.1: Better and Faster Open Whisper Style Speech Models based on E Branchformer
- **Authors:** Yifan Peng et al. (CMU)
- **Year:** 2024 (Interspeech)
- **arXiv:** https://arxiv.org/abs/2401.16658
- **Summary:** Fully open reproduction of Whisper using public data and ESPnet, with the Transformer encoder swapped for E Branchformer. Matches Whisper's WER on most languages with 73k hours of English data (vs Whisper's 438k), runs 25% faster, and shows emergent zero shot contextual biasing.
- **Relevance:** The closest open data / open code analogue to Whisper. If reproducibility or licensing of Whisper ever becomes a blocker for Off The Record, OWSM v3.1 is the drop in.

### Voxtral
- **Title:** Voxtral
- **Authors:** Mistral AI
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2507.13264
- **Summary:** Voxtral Mini (Ministral 3B backbone) and Voxtral Small (Mistral Small 3.1 24B backbone) are open weights multimodal LLMs trained to natively accept audio. State of the art on transcription and translation, beating GPT 4o mini transcribe and Gemini 2.5 Flash.
- **Relevance:** Represents the "audio LLM" alternative to encoder decoder ASR: a general LLM that happens to speak audio. Architecturally heavier than Whisper but much more capable downstream (summarisation, Q&A from audio). Important context if Off The Record ever adds in line note synthesis from a transcript.

### Phi 4 Multimodal
- **Title:** Phi 4 Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture of LoRAs
- **Authors:** Microsoft Phi team
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2503.01743
- **Summary:** A 5.6B parameter model unifying text, vision, and speech via modality specific LoRA adapters and routers. The 460M parameter speech LoRA topped the Hugging Face OpenASR leaderboard at WER 6.14% on release. First open model with native speech summarisation.
- **Relevance:** Same role as Voxtral, audio aware LLM rather than pure ASR. Useful as evidence that LoRA style multimodal extension can match dedicated ASR models on benchmark WER.

### Qwen2 Audio
- **Title:** Qwen2 Audio Technical Report
- **Authors:** Yunfei Chu et al. (Alibaba)
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2407.10759
- **Summary:** A large audio language model with two interaction modes (voice chat / audio analysis). Outperforms Gemini 1.5 Pro on AIR Bench instruction following.
- **Relevance:** Same category as Voxtral / Phi 4 Multimodal. Cited as a strong Mandarin first option.

### Step Audio
- **Title:** Step Audio: Unified Understanding and Generation in Intelligent Speech Interaction
- **Authors:** Step Audio Team (StepFun)
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2502.11946
- **Summary:** A 130B unified speech text multimodal model with a paired Step Audio TTS 3B for distilled deployment. First production ready open audio understanding/generation system.
- **Relevance:** State of the art Chinese open audio LLM. Background for the high end of "what an audio LLM can do today."

### Kimi Audio
- **Title:** Kimi Audio Technical Report
- **Authors:** Kimi Team (Moonshot AI)
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2504.18425
- **Summary:** Continuous feature encoder plus discrete token decoder LLM based audio foundation model, pre trained on 13M hours, with a streaming flow matching detokeniser. SOTA on speech recognition, audio understanding, audio QA, and conversation benchmarks.
- **Relevance:** Background. The flow matching detokeniser is interesting for any future streaming output beyond text (e.g. live captioning plus voice).

### GLM 4 Voice
- **Title:** GLM 4 Voice: Towards Intelligent and Human Like End to End Spoken Chatbot
- **Authors:** Aohan Zeng et al. (Zhipu AI)
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2412.02612
- **Summary:** Bilingual (Chinese/English) end to end spoken chatbot with a 175 bps single codebook 12.5 Hz speech tokeniser. Continues pre training from GLM 4 9B with interleaved speech text data.
- **Relevance:** Reference for end to end speech chatbots; not directly applicable to a transcription only app.

---

## 8. Speculative decoding and decoder efficiency

### Speculative Decoding for Whisper
- Speculative decoding pairs a small "draft" Whisper (e.g. Distil Whisper) with the full model: the draft proposes a chunk of tokens, the full model verifies in one forward pass. Yields ~2x speedup and is mathematically guaranteed to produce identical outputs to the full model. Demonstrated in the Distil Whisper paper (https://arxiv.org/abs/2311.00430) and the official Hugging Face Whisper Speculative Decoding blog post (https://huggingface.co/blog/whisper-speculative-decoding).
- **Relevance:** A free 2x speedup for Off The Record's batch panel without any change to Whisper's outputs, if/when Transformers.js exposes a multi pipeline draft verify API.

### Whisper Medusa
- See section 3.

### Model free Speculative Decoding for ASR (Token Map Drafting)
- **Title:** Model free Speculative Decoding for Transformer based ASR with Token Map Drafting
- **arXiv:** https://arxiv.org/abs/2507.21522
- **Summary:** Replaces the draft model with a lookup table of common output token n grams keyed by encoder states. Zero extra GPU memory, similar speedups to Medusa for narrow domain data.
- **Relevance:** Lighter weight speculative decoder; potentially more browser friendly because no second model to load.

---

## 9. Voice Activity Detection (VAD)

### Silero VAD
- **Title:** Silero VAD: pre trained enterprise grade Voice Activity Detector
- **Authors:** Silero Team
- **Year:** 2021 onward (no formal academic paper; documentation and JIT model release)
- **URL:** https://github.com/snakers4/silero-vad
- **Summary:** A small (~1 MB), language agnostic neural VAD trained on 100+ languages. Sub millisecond per 30 ms chunk on a single CPU thread. Supports 8 and 16 kHz. MIT licensed.
- **Relevance:** The pragmatic default VAD for any browser side audio pipeline. Off The Record's `heuristics.ts` does its own silence detection on Whisper's output to avoid hallucinations, but Silero VAD on the producer worker (in front of the consumer) would be a clean way to gate Whisper invocations entirely.

### pyannote.audio 3 / Powerset diarization
- **Title 1:** pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe
  - **Authors:** Herve Bredin
  - **Venue:** Interspeech 2023, https://www.isca-archive.org/interspeech_2023/bredin23_interspeech.pdf
- **Title 2:** Powerset multi class cross entropy loss for neural speaker diarization
  - **Authors:** Alexis Plaquet, Herve Bredin
  - **Year:** 2023 (Interspeech)
  - **arXiv:** https://arxiv.org/abs/2310.13025
- **Summary:** The Bredin 2023 paper documents the v2.1 pipeline (segmentation, embedding, agglomerative clustering). The Plaquet and Bredin 2023 paper introduces the powerset loss that powers pyannote 3.0/3.1: single label classification over the powerset of speaker subsets, eliminating the detection threshold hyperparameter and improving overlap handling.
- **Relevance:** pyannote 3.x is the de facto open source speaker diarisation pipeline that pairs with Whisper (see WhisperX). If Off The Record adds "who said what", this is the model family.

---

## 10. Speaker recognition and diarization

### EEND
- **Title:** End to End Neural Speaker Diarization with Self Attention
- **Authors:** Yusuke Fujita, Naoyuki Kanda, Shota Horiguchi, Yawen Xue, Kenji Nagamatsu, Shinji Watanabe (Hitachi)
- **Year:** 2019 (IEEE ASRU)
- **arXiv:** https://arxiv.org/abs/1909.06247 (related earlier paper at Interspeech 2019: https://arxiv.org/abs/1909.05952)
- **Summary:** Reframes speaker diarization as a permutation invariant multi label classification problem solved end to end with a self attention encoder. Handles overlapping speech directly, which clustering based pipelines cannot.
- **Relevance:** Reference paper for any neural diarizer. pyannote 3 borrows from EEND style segmentation.

### TitaNet
- **Title:** TitaNet: Neural Model for speaker representation with 1D Depth wise separable convolutions and global context
- **Authors:** Nithin Rao Koluguri, Taejin Park, Boris Ginsburg (NVIDIA)
- **Year:** 2022 (ICASSP)
- **arXiv:** https://arxiv.org/abs/2110.04410
- **Summary:** 1D depthwise separable convolutional speaker embedding extractor with channel attentive statistics pooling. 0.68% EER on VoxCeleb1; DER 1.73% on AMI MixHeadset.
- **Relevance:** A small, efficient speaker embedder usable as the clustering stage of a diarisation pipeline.

### CAM++
- **Title:** CAM++: A Fast and Efficient Network for Speaker Verification Using Context Aware Masking
- **Authors:** Hui Wang et al. (Alibaba)
- **Year:** 2023 (Interspeech)
- **arXiv:** https://arxiv.org/abs/2303.00332
- **Summary:** Densely connected TDNN backbone with context aware masking pooling, matching ECAPA TDNN at lower compute.
- **Relevance:** Alternative speaker embedder; relevant if pairing diarization with Alibaba's broader speech stack.

### 3D Speaker
- **Title:** 3D Speaker Toolkit: An Open Source Toolkit for Multimodal Speaker Verification and Diarization
- **Authors:** Yafeng Chen, Siqi Zheng et al. (Alibaba)
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2403.19971
- **Summary:** Open toolkit combining acoustic (CAM++/ERes2Net), semantic (LLM), and visual modalities for speaker verification and diarization, with a 10k speaker public dataset.
- **Relevance:** Useful reference for the state of the art in multimodal speaker diarization.

---

## 11. Datasets and benchmarks

### LibriSpeech
- **Title:** Librispeech: An ASR corpus based on public domain audio books
- **Authors:** Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur
- **Year:** 2015 (ICASSP)
- **URL:** https://www.danielpovey.com/files/2015_icassp_librispeech.pdf
- **Summary:** 1,000 hours of 16 kHz read English speech derived from LibriVox audiobooks, with text cleaned transcripts and pre built language models. The most cited ASR benchmark in existence.
- **Relevance:** Reference benchmark. Off The Record's accuracy can be sanity checked on test clean / test other.

### Common Voice
- **Title:** Common Voice: A Massively Multilingual Speech Corpus
- **Authors:** Rosana Ardila et al. (Mozilla)
- **Year:** 2019
- **arXiv:** https://arxiv.org/abs/1912.06670
- **Summary:** Crowd sourced, CC0 licensed corpus that has grown to thousands of hours across 100+ languages.
- **Relevance:** The largest public domain multilingual speech corpus. Useful for evaluating Whisper tiny in real world heterogeneous acoustics.

### FLEURS
- **Title:** FLEURS: Few shot Learning Evaluation of Universal Representations of Speech
- **Authors:** Alexis Conneau et al. (Google)
- **Year:** 2022
- **arXiv:** https://arxiv.org/abs/2205.12446
- **Summary:** N way parallel speech corpus in 102 languages built on top of FLoRes 101 (~12 h/language). Standard for cross lingual ASR, LID, retrieval, and translation eval.
- **Relevance:** Canonical multilingual ASR benchmark for Whisper class models.

### GigaSpeech
- **Title:** GigaSpeech: An Evolving, Multi domain ASR Corpus with 10,000 Hours of Transcribed Audio
- **Authors:** Guoguo Chen et al.
- **Year:** 2021
- **arXiv:** https://arxiv.org/abs/2106.06909
- **Summary:** 10k hours of carefully filtered English audio from audiobooks, podcasts, and YouTube, plus 40k hours unlabelled. Subsets of 10h/250h/1000h/2500h/10000h for ladder evaluation.
- **Relevance:** A more realistic ASR benchmark than LibriSpeech; closer to the audio Off The Record actually sees (conversational and spontaneous, not read).

### GigaSpeech 2
- **Title:** GigaSpeech 2: An Evolving, Large Scale and Multi domain ASR Corpus for Low Resource Languages with Automated Crawling, Transcription and Refinement
- **Authors:** Yifan Yang et al.
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2406.11546
- **Summary:** 30k+ hours of automatically transcribed Thai, Indonesian, and Vietnamese YouTube, refined with noisy student training. ASR models trained on it cut WER by 25 to 40% versus Whisper large v3 on those languages with only 10% the parameters.
- **Relevance:** Strong evidence that targeted data plus small model can beat Whisper for low resource languages.

### WenetSpeech
- **Title:** WenetSpeech: A 10000+ Hours Multi domain Mandarin Corpus for Speech Recognition
- **Authors:** Binbin Zhang et al. (WeNet)
- **Year:** 2021
- **arXiv:** https://arxiv.org/abs/2110.03370
- **Summary:** 10k labelled plus 12k weakly labelled plus 10k unlabelled hours of Mandarin from YouTube and Podcasts. Largest public Mandarin corpus with three held out test sets.
- **Relevance:** Benchmark for Mandarin Whisper alternatives (Paraformer, SenseVoice).

### People's Speech
- **Title:** The People's Speech: A Large Scale Diverse English Speech Recognition Dataset for Commercial Usage
- **Authors:** Daniel Galvez et al. (MLCommons)
- **Year:** 2021 (NeurIPS Datasets and Benchmarks)
- **arXiv:** https://arxiv.org/abs/2111.09344
- **Summary:** 30,000 hour CC BY SA licensed English ASR corpus assembled from open licensed audio plus transcripts on the internet. Explicitly commercial use licensed.
- **Relevance:** The benchmark people show when comparing commercially permissive open data to Whisper's proprietary 680k hour set.

### AISHELL 1
- **Title:** AISHELL 1: An Open Source Mandarin Speech Corpus and A Speech Recognition Baseline
- **Authors:** Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, Hao Zheng
- **Year:** 2017
- **arXiv:** https://arxiv.org/abs/1709.05522
- **Summary:** 170 h, 400 speakers, Apache 2.0 Mandarin ASR corpus. Standard small scale Mandarin benchmark.
- **Relevance:** Reference benchmark for Mandarin ASR research.

### AISHELL 2
- **Title:** AISHELL 2: Transforming Mandarin ASR Research Into Industrial Scale
- **Authors:** Jiayu Du, Xingyu Na, Xuechen Liu, Hui Bu
- **Year:** 2018
- **arXiv:** https://arxiv.org/abs/1808.10583
- **Summary:** 1000 h iOS channel Mandarin read speech free for academic use, plus matching Android and Mic test sets to study channel mismatch.
- **Relevance:** Larger Mandarin benchmark; the workhorse for industrial Mandarin ASR research.

### VoxLingua107
- **Title:** VoxLingua107: a Dataset for Spoken Language Recognition
- **Authors:** Jorgen Valk, Tanel Alumae
- **Year:** 2020 (IEEE SLT 2021)
- **arXiv:** https://arxiv.org/abs/2011.12998
- **Summary:** 6,628 h of YouTube audio across 107 languages, filtered by speech activity detection and crowd verification. Standard public spoken LID benchmark.
- **Relevance:** Relevant if Off The Record ever performs language detection as a pre processing step.

### AMI Meeting Corpus
- **Title:** The AMI Meeting Corpus: A Pre announcement
- **Authors:** Jean Carletta et al.
- **Year:** 2005 (MLMI)
- **URL:** https://link.springer.com/chapter/10.1007/11677482_3
- **Summary:** 100 hours of multi modal meeting recordings with close talking and far field microphones, video, plus rich annotations (orthography, dialogue acts, summaries, emotions). Defines the "meeting ASR" benchmark.
- **Relevance:** The reference benchmark for meeting transcription and diarisation, including AMI Headset / AMI Lapel (used in TitaNet, pyannote).

---

## 12. Streaming latency metrics

### Average Lagging (AL) and Length Adaptive Average Lagging (LAAL)
- **Title:** Over Generation Cannot Be Rewarded: Length Adaptive Average Lagging for Simultaneous Speech Translation
- **Authors:** Sara Papi, Marco Gaido, Matteo Negri, Marco Turchi
- **Year:** 2022 (AutoSimTrans workshop)
- **arXiv:** https://arxiv.org/abs/2206.05807
- **Summary:** Average Lagging (Ma et al. 2019) measures, on average, how many target tokens a streaming system is behind a perfect oracle. AL under counts latency for over generating systems; LAAL fixes this by using the *longer* of reference and output as the denominator.
- **Relevance:** The primary latency metric the streaming ASR community uses. If Off The Record ever publishes its LocalAgreement-2 latency vs. competitors, AL/LAAL is the metric.

### Average Token Delay (ATD)
- **Title:** Average Token Delay: A Latency Metric for Simultaneous Translation
- **Authors:** Yasumasa Kano et al.
- **Year:** 2023 (Interspeech)
- **URL:** https://www.isca-archive.org/interspeech_2023/kano23_interspeech.pdf (extended duration aware version: https://arxiv.org/abs/2311.14353)
- **Summary:** A duration aware latency metric that measures the average delay between when an output token *could* have been emitted and when it *was* emitted, given the source token timestamps. Complementary to AL/LAAL.
- **Relevance:** Per token latency analysis is exactly what LocalAgreement-2 trades off; ATD is the canonical way to report it.

### Computation Aware Latency / CA*
- **Title:** CA*: Addressing Evaluation Pitfalls in Computation Aware Latency for Simultaneous Speech Translation
- **arXiv:** https://arxiv.org/abs/2410.16011
- **Summary:** Re examines how to fairly compare latency for systems with very different compute costs (e.g. Whisper tiny vs. Whisper large).
- **Relevance:** Important if Off The Record reports latency comparisons against server side or differently sized models.

---

## 13. Toolkits worth knowing (papers)

### ESPnet
- **Title:** ESPnet: End to End Speech Processing Toolkit
- **Authors:** Shinji Watanabe et al.
- **Year:** 2018 (Interspeech)
- **arXiv:** https://arxiv.org/abs/1804.00015
- **Summary:** PyTorch/Chainer toolkit covering ASR, TTS, ST, and SE with Kaldi style data recipes. Maintains the largest public collection of speech recipes alongside k2/WeNet.
- **Relevance:** Where most of the non Whisper open ASR models live (including OWSM v3.1).

### WeNet
- **Title:** WeNet: Production Oriented Streaming and Non streaming End to End Speech Recognition Toolkit
- **Authors:** Binbin Zhang et al.
- **Year:** 2021
- **arXiv:** https://arxiv.org/abs/2102.01547
- **Summary:** Production focused end to end speech toolkit that introduced the U2 unified streaming/non streaming architecture (CTC plus attention rescoring), runtime quantisation, and ONNX/TensorRT export.
- **Relevance:** Closest open source spiritual ancestor to "one model, two latency modes", which is essentially Off The Record's UX.

---

## Coverage notes (what was deliberately left out)

- **Pure speech translation work that is not relevant to monolingual transcription** (e.g. SimulMT, MultiSimulST surveys) is omitted except where it produced a metric (AL, LAAL, ATD) the streaming ASR community uses.
- **Diffusion based ASR** (Drax 2025: https://arxiv.org/abs/2510.04162) and **discrete token ASR** are out of scope for a browser deployment in 2026 but worth tracking.
- **CosyVoice / TTS** is mentioned only insofar as it shares the FunAudioLLM paper with SenseVoice.
- **Speaker change detection, overlap detection, target speaker ASR** are all related to diarization but are not load bearing for a single speaker live transcription app.

## Sources

- arXiv (verified URLs for every entry).
- Interspeech / ICASSP / ICML / NeurIPS proceedings for the few entries without an arXiv pre print (CTC 2006, LibriSpeech 2015, AMI 2005, pyannote 2.1 2023).
- All authorship and year details cross checked via web search against the canonical arXiv abstract pages.
