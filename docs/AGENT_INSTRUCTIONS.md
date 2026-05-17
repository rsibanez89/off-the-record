# AGENT_INSTRUCTIONS, Research Plan

> Standardised research plan for the "state of the art in offline speech-to-text transcription" investigation that produced the docs in this folder. Reusable as a `PLAN.md`-style brief for future research passes.

## Mission

Document, in depth, the full landscape of **offline / on-device / browser-runnable** automatic speech recognition (ASR) systems, the algorithms that make them work in a streaming setting, and the academic literature underneath them, so that an engineer building a browser-side transcription app (Off The Record) can:

1. Pick the right model for any quality, latency, or size tradeoff.
2. Understand the streaming algorithms (LocalAgreement-2 and alternatives).
3. Know which runtimes and quantizations actually deploy in a 2026 browser.
4. Anchor every choice to a primary source (paper, repo, model card).

Off-the-record specifics that bias the scope:

- Runs entirely in the **browser**. WebGPU when present, WASM otherwise.
- Uses **transformers.js v3** and **ONNX Runtime Web**.
- Streams using **LocalAgreement-2** over Whisper.
- Provides a **batch vs streaming side-by-side comparison** UI.
- Targets desktop Chrome and Edge primarily, Firefox and Safari secondarily.

## Output contract

All research artefacts live in `docs/` at the repo root.

| File | Purpose |
|---|---|
| `docs/AGENT_INSTRUCTIONS.md` | This file. The plan. |
| `docs/state-of-the-art.md` | Final synthesised, readable overview. Single source of truth. |
| `docs/repositories.md` | Catalogue of every public repo or code release for offline ASR. |
| `docs/papers.md` | Annotated bibliography of every relevant paper. |
| `docs/research-whisper.md` | Working notes, Whisper ecosystem. |
| `docs/research-chinese.md` | Working notes, Chinese and Asian models. |
| `docs/research-western.md` | Working notes, Western non-Whisper models. |
| `docs/research-browser.md` | Working notes, browser / WebGPU / runtime stack. |
| `docs/research-streaming.md` | Working notes, streaming algorithms, VAD, diarization. |
| `docs/research-papers.md` | Working notes, academic literature. |

The `research-*.md` files are raw working notes from parallel research agents. `state-of-the-art.md`, `repositories.md`, and `papers.md` are the **synthesised, polished outputs** that should be read first.

## Methodology

### 1. Decomposition into parallel research streams

Six independent topics, each large enough to justify its own agent, small enough to avoid overlap:

1. **Whisper ecosystem**. OpenAI Whisper and every documented derivative.
2. **Chinese and Asian models**. Alibaba, Tencent, ByteDance, StepFun, Moonshot, iFlyTek, Kaldi next-gen, Japanese, Korean.
3. **Western non-Whisper**. NVIDIA NeMo, Meta MMS and Seamless, Mistral Voxtral, Kyutai, UsefulSensors Moonshine, Rev, IBM, Microsoft, Google, AssemblyAI and Deepgram references.
4. **Browser and edge runtime stack**. transformers.js v3, ONNX Runtime Web, WebGPU, WebNN, WASM, audio capture, storage.
5. **Streaming algorithms and supporting tech**. LocalAgreement-2, VAD (Silero, TEN, WebRTC, pyannote), diarization, forced alignment, speculative decoding, hallucination mitigation.
6. **Academic papers**. Annotated bibliography, 2016 to 2026.

Each stream is researched by a dedicated `general-purpose` sub-agent firing `WebSearch` and `WebFetch` aggressively. Streams are explicitly scoped so they do not duplicate work.

### 2. Source priority

For each model or system, sources are weighted in this order:

1. Original paper on arXiv (or venue PDF).
2. Official GitHub repository.
3. Hugging Face or ModelScope model card.
4. Official blog post or technical report.
5. Open ASR Leaderboard or another independent benchmark.
6. Third-party reproduction (community notebooks, benchmarks).

Each agent is required to cite raw URLs. Marketing copy without a primary source is flagged as such.

### 3. Per-entry data shape

For every model, capture:

- Repo URL and license.
- HuggingFace or ModelScope IDs.
- Sizes, parameter counts, VRAM.
- Languages.
- Architecture family (encoder-decoder Transformer, Conformer, FastConformer, Zipformer, RNN-T, TDT, CTC, non-autoregressive).
- WER on at least one of: LibriSpeech test-clean and test-other, CommonVoice, FLEURS, AISHELL, WenetSpeech, Open ASR Leaderboard average.
- Real-time factor (RTF) or tokens per second where available.
- Quantization availability (FP16, INT8, Q4, GGUF, ONNX).
- Browser, WebGPU, or WASM availability.
- Streaming support.
- Maintenance status.
- Notable paper(s).

For every paper, capture:

- Title, lead author, year, venue, arxiv ID.
- Two or three sentence summary.
- Relevance to offline browser transcription.

### 4. Quality bar

- No hallucinations. Every claim that cites a number or capability links to a source.
- License is captured for every release. Non-commercial or research-only restrictions are flagged explicitly because they are blockers for a deployed app.
- Chinese and Asian sources are weighted as heavily as Western ones. They are a genuine frontier (SenseVoice claims 15x over Whisper, Paraformer is the canonical non-autoregressive ASR, Zipformer powers Sherpa-ONNX).
- The browser stack section is opinionated. It must reflect what actually ships in Chrome stable as of May 2026.

### 5. Synthesis

After all six working note files are populated, the synthesiser:

1. Builds `state-of-the-art.md` as a readable narrative for an engineer. Architecture eras, current frontier, browser-specific options, recommendation matrix.
2. Builds `repositories.md` as a flat catalogue grouped by family, every entry with URL, license, and one-line description.
3. Builds `papers.md` as a chronological annotated bibliography.

The synthesised docs are the audience facing artefacts. The `research-*.md` files are kept verbatim for traceability.

## Running this plan

Replay the parallel research at any time by re-issuing the six prompts in the conversation that produced these docs. Each prompt is fully self-contained and writes to its own `research-*.md` file, so re-runs do not collide. Synthesis step is a separate pass once all six files exist.

## Why this shape

- **Parallel sub-agents.** Six independent streams complete in roughly the time of one. They share no files, so concurrent writes are safe.
- **Working notes vs synthesis.** Keeping the raw research in `research-*.md` means the final synthesis is auditable. If `state-of-the-art.md` claims something, the supporting note has the citations.
- **Scoped to deployment reality.** The host project ships in a browser. License restrictions, ONNX export availability, and WebGPU compatibility carry as much weight as raw WER.
- **Standing artefact.** This file (`AGENT_INSTRUCTIONS.md`) is the standing brief. The investigation can be re-run, extended, or audited against it without re-derivation.

## Update protocol

When new models, papers, or runtimes appear:

1. Add a row to the appropriate `research-*.md` file with full source links.
2. Update the synthesised `state-of-the-art.md`, `repositories.md`, and `papers.md` to reflect the new entry.
3. If a new top-of-leaderboard model appears, update the "current frontier" section of `state-of-the-art.md` and the recommendation matrix.
4. Date stamp the change in a `## Changelog` section at the bottom of the affected file.
