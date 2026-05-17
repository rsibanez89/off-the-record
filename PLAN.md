# Off The Record, Build Plan

## Your role

You are an expert in offline speech recognition and Whisper. You know transformers.js v3, WebGPU/ONNX runtime, Web Audio API, AudioWorklet, IndexedDB (Dexie), Cache API, and browser-side streaming transcription heuristics. Build a clean, minimal, working POC. Favour clarity over cleverness. No backend. Everything runs in the browser.

## Goal

A web app that records the mic and shows **two transcripts at once** for direct comparison:

- 🟢 **Live transcript** powered by a streaming LocalAgreement-2 over Whisper.
- 🔵 **Batch transcript** produced by a single Whisper pass on the full audio, triggered when the user presses Stop.

Each panel has its own model picker, so the same audio can be transcribed by two different models (or the same model with two different algorithms).

The app prefers WebGPU when available and falls back to WASM. Audio capture starts on the main thread, raw samples are sent through an AudioWorklet, and chunking/transcription are handled by three independent Web Workers. The producer and live consumer coordinate through IndexedDB for audio payloads, with `BroadcastChannel('new-chunk')` used only as a wake-up signal. The batch worker is triggered by a direct message from the main thread on Stop.

## Architecture (four parts)

- **Producer worker** (`producer.worker.ts`)
  - Receives raw context-rate `Float32Array` frames from the main thread.
  - Linearly resamples to 16 kHz mono and buffers into 1-second chunks (typed ring buffer).
  - Runs **Silero VAD v5** (`src/lib/vad/`) on every 512-sample (32 ms) 16 kHz frame as samples arrive. Tracks `max(speechProbability)` for the outgoing 1-second chunk. VAD load is lazy and fire-and-forget on first Record; chunks written before VAD is ready carry `speechProbability: undefined` and the consumer falls back to RMS.
  - Writes each chunk to **two** IDB tables in one transaction:
    - `db.chunks`: live, evictable (the live consumer deletes as the anchor advances). Carries `speechProbability?: number`.
    - `db.audioArchive`: kept until next Record or Clear (input to the batch worker). Same shape.
  - Posts `BroadcastChannel('new-chunk')` as a wake-up notification only. Audio payload never leaves IDB after this write.
  - Acknowledges `stop` with `stopped` so the main thread can wait before flushing the consumer and before triggering the batch run.

- **Live consumer worker** (`consumer.worker.ts`)
  - Lazy-loads the Whisper pipeline using transformers.js via `whisperAdapter.createPipeline`.
  - Detects WebGPU, otherwise uses WASM (`whisperAdapter.detectBackend`).
  - Uses fp32 encoder for non-turbo checkpoints, fp16 encoder for `large-v3-turbo` on WebGPU, and q4 merged decoder.
  - On wake-up: collects chunks from `committedAudioStartS` onward and runs inference.
  - Feeds Whisper output to the `HypothesisBuffer` (LocalAgreement-2).
  - Reconciles `db.transcript` from the buffer's committed and tentative queues. Writes new rows before deleting excess rows so the UI does not flash through an empty/short table on every inference tick.
  - Deletes audio chunks from `db.chunks` older than the committed audio anchor. `db.audioArchive` is untouched.
  - Acknowledges `reset` with `reset-done` and `flush` with `flush-done`.

- **Batch worker** (`batch.worker.ts`)
  - Lazy-loads its own Whisper pipeline (same helpers as the live consumer; independent model selection).
  - On `transcribe { sessionId }`:
    - Loads the entire `db.audioArchive` table.
    - Concatenates chunks into one `Float32Array`.
    - Runs Whisper once with `return_timestamps: 'word'` and `chunk_length_s: 30` (transformers.js handles longer audio via overlapping windows).
    - Posts `transcribe-done { sessionId, tokens, durationS, inferenceMs }`.
  - Session IDs let the main thread ignore late results from a superseded batch run (e.g. the user clicked Record again while batch was still going).

- **Main thread (React)**
  - Calls `getUserMedia`, owns `AudioContext`, loads `/audio-worklet.js`, and forwards worklet frames to the producer worker.
  - Spawns and supervises three workers: producer, live consumer, batch.
  - Renders two `TranscriptPanel`s side by side, each with its own model picker.
  - Header has only Record / Stop / Clear. Backend pill and model picker are inside each `TranscriptPanel`.
  - Waveform: separate `AnalyserNode` on the live mic stream. Not via IDB. Visual must be zero-lag.
  - Stop enters a `stopping` state, waits for producer `stopped` and live consumer `flush-done`, then triggers the batch worker. Batch runs in the background and updates the right panel when it finishes.

## Streaming transcript strategy: LocalAgreement-2

The live consumer uses the **LocalAgreement-2** algorithm (Macháček, Dabre, Bojar 2023; the reference implementation is `ufal/whisper_streaming`). Whisper is run with `return_timestamps: 'word'` so we have per-word timestamps.

- Whisper is not used as a true streaming model. The consumer repeatedly transcribes a growing audio window.
- Minimum live window: 0 seconds (run on the first chunk for fast feedback). Stop drain window: 0 seconds (transcribe whatever is left on Stop). Maximum window: 24 seconds.
- `committedAudioStartS` anchors the left edge of audio that still needs transcription. **Anchor advancement is conservative: it preserves intra-sentence acoustic context for Whisper.**
  - Trim only when the latest committed word ends a sentence (`.`, `?`, `!`). Target: `sentence_end_time - CONTEXT_LOOKBACK_S` (default 5 s).
  - Secondary cap: if the window grows past `FAST_TRIM_THRESHOLD_S` (default 10 s) without a sentence end, trim anyway to `last_committed_end - CONTEXT_LOOKBACK_S`. Prevents inference from falling behind real-time on long unpunctuated speech.
  - MAX_WINDOW_S force-slide still fires as the absolute safety net (commits whatever is in `tentative`, trims to `audio.t1 - CONTEXT_LOOKBACK_S`).
  - Rationale: aggressive per-word trimming (anchor to last-committed-word end time) strips context Whisper needs for accuracy, especially on `tiny.en` and `base.en`. Verified empirically against the batch panel.
- The `HypothesisBuffer` class (`src/lib/transcription/hypothesisBuffer.ts`) holds two queues:
  - `committed`: permanently confirmed words. Render as `isFinal:1` (black). Never revert.
  - `tentative`: the surviving tail of the previous iteration's hypothesis. Render as `isFinal:0` (grey).
- On each inference tick:
  1. Drop words from the new hypothesis whose end is before the last committed boundary (timestamp jitter tolerance: 0.1 s).
  2. Run 5-gram tail dedup between the committed transcript's tail and the new hypothesis's head to strip re-emitted overlap.
  3. Walk the new hypothesis and `tentative` in parallel. Every word that matches (case- and punctuation-normalised) graduates to `committed`. Stop at the first mismatch. The surviving tail of the new hypothesis becomes the next `tentative`.
- **Once a word is committed it is monotonically stable.** No flicker back to grey.
- Standalone dash artefacts and known hallucination fillers are filtered at the word and line level.
- **Silence gate: Silero VAD first, RMS fallback.** For each window the consumer collects per-chunk `speechProbability` and computes the window's verdict as `max(...)`. If the verdict is below `VAD_SILENCE_THRESHOLD` (0.3, in `src/lib/config.ts`), the window is silent: force-commit any tentative buffer, advance the anchor past the silent region, discard those chunks. If ANY chunk in the window lacks a VAD verdict (VAD was still loading, or has failed for this tab), the window falls back to the RMS check in `heuristics.ts::isSilent`. Treating partial coverage as "max over verdicted chunks only" would silently drop real speech in the unverdicted chunks, so the all-or-nothing rule is load-bearing.
- Full-line hallucinations are detected AFTER the silence gate, so by definition they fire on non-silent audio. This means the model has failed (small model + short context), not that the audio is empty. The handler does NOT advance the anchor and does NOT discard chunks; it just defers the tick and lets the audio buffer grow so the next inference has more context. If hallucinations persist long enough that the window exceeds `MAX_WINDOW_S`, the force-slide safety net commits and trims.
- A `MAX_WINDOW_S` safety net force-slides if the window grows beyond 24 s without any natural commit (continuous unstable speech).
- On `stop`, the consumer drains available chunks through LocalAgreement-2 until the state settles or the safety cap fires, force-commits any remaining tentative buffer, rewrites the transcript as final, clears chunks, and posts `flush-done`.

## Batch transcript strategy

The batch worker runs **only on Stop**. It takes the entire `audioArchive` table, concatenates the chunks, and runs Whisper once. No streaming, no buffer, no agreement protocol. The result is a single rendered token list that populates the right panel.

- Same `return_timestamps: 'word'` and same `runWhisper` adapter as the live consumer.
- `chunk_length_s: 30` lets transformers.js handle longer-than-30s audio via internal windowing.
- All output tokens are `isFinal: 1` (black). There is no notion of tentative in a one-shot run.
- A `sessionId` accompanies each transcribe request. The main thread bumps it on Clear / new Record, so any late result from a superseded run is dropped.

## Session lifecycle

- **Record**: wipes both transcripts, resets the live consumer's in-memory state via `reset` + `reset-done`, calls `clearAll()` (drops `chunks`, `transcript`, `audioArchive`), bumps the batch session ID, then starts capture.
- **Stop**: stops capture, awaits producer `stopped`, awaits live consumer `flush-done`, sends `transcribe { sessionId }` to the batch worker (does not await; batch updates the right panel asynchronously).
- **Clear**: same reset flow as Record but without starting capture.
- Each Record session is independent. Live and batch transcripts are scoped to one recording.

## Tech stack

- Vite plus React 19 plus TypeScript.
- Tailwind. Minimalist: two transcript panels side by side, small waveform strip, dev panel below.
- `@huggingface/transformers` (v3+) for Whisper inference.
- Dexie for IndexedDB. `lucide-react` for icons.
- No router, no state library. Dexie `liveQuery` plus React state is enough.

## Storage rules

- **Model weights live in the Cache API**, not IDB. IDB silently fails past ~1 GB on some browsers; Whisper shards exceed that. transformers.js handles this automatically. Do NOT override its caching.
- **`db.chunks`**: 1-second audio chunks for the live consumer. Evicted as the committed-audio anchor advances.
- **`db.audioArchive`**: 1-second audio chunks kept until next Record or Clear. Input to the batch worker. Each chunk is small (<200 KB at 16 kHz mono Float32, 1 s).
- **`db.transcript`**: live transcript tokens. Reconciled on each inference tick.
- `clearAll()` drops all three IDB tables.
- Cache API weights are shared across panels when the model ID matches. Different models means two parallel downloads and two pipelines loaded in memory.

## Model picker

- Three options, dropdown in each panel:
  - `onnx-community/whisper-tiny.en_timestamped` (fast, weaker)
  - `onnx-community/whisper-base.en_timestamped` (default for both panels)
  - `onnx-community/whisper-large-v3-turbo_timestamped` (slower first load, best quality)
- All entries are `_timestamped` ONNX exports. The non-timestamped variants do not include cross-attention outputs, which transformers.js needs to compute word-level timestamps via DTW. LocalAgreement-2 depends on those word timestamps.
- Selection persisted in `localStorage`. Two keys: `off-the-record:model` (live) and `off-the-record:batch-model` (batch).
- Changing a model: terminate then respawn the matching worker.
- Show download progress on first load (transformers.js exposes a progress callback).

## File layout

```
src/
├── main.tsx, App.tsx
├── components/
│   ├── TranscriptPanel.tsx   transcript + per-panel model picker + status badge
│   ├── Waveform.tsx
│   ├── ModelPicker.tsx
│   └── DevPanel.tsx
├── workers/
│   ├── producer.worker.ts    audio capture, resample, dual-table writes
│   ├── consumer.worker.ts    live LocalAgreement-2 streaming
│   └── batch.worker.ts       one-shot Whisper on Stop
└── lib/
    ├── db.ts                  Dexie v4 schema (chunks + audioArchive carry speechProbability), clearAll
    ├── audio.ts               models, sample rate, isMultilingual, isTurbo
    ├── config.ts              VAD_SILENCE_THRESHOLD and centralised tunables
    ├── transcription/
    │   ├── hypothesisBuffer.ts   LocalAgreement-2 (pure algorithm)
    │   ├── heuristics.ts          hallucination detection + RMS silence fallback
    │   └── whisperAdapter.ts      pipeline creation, backend detection, runWhisper
    └── vad/                   in-house Silero VAD v5 (VadEngine interface, SileroVad, Framer,
                                LinearResampler, VadStateMachine, NoopVadEngine, createVadEngine)

public/
├── audio-worklet.js
└── models/                    silero_vad_v5.onnx (sha256-pinned, fetched by scripts/fetch-models.mjs)

scripts/
└── fetch-models.mjs           predev/prebuild provisioner for the Silero ONNX
```

Plus root: `index.html`, `vite.config.ts` (COOP/COEP headers + ortRuntime plugin serving `/ort/*`), `package.json`, `tsconfig.json`, `tailwind.config.ts`.

## Critical gotchas (do not relearn)

- **COOP/COEP headers required** for `SharedArrayBuffer` (transformers.js needs it). Vite dev server must set `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`.
- **q4 on the encoder destroys accuracy** in noise/accents. Use `q4` on `decoder_model_merged` only. Keep the encoder at `fp32` except for `large-v3-turbo` on WebGPU, where a published `fp16` encoder is used.
- **WebGPU adapter check** on startup. Fall back to WASM with a visible warning.
- **Producer and live consumer must stay decoupled for audio payloads.** IDB is the audio contract; `BroadcastChannel` is only a wake-up.
- **Batch worker is triggered by direct message**, not via `BroadcastChannel`. The wake-up channel is exclusive to the live pipeline.
- **Do not allow Record during stop flush.** The UI must wait for producer `stopped` and live consumer `flush-done`, otherwise old and new transcript rewrites can flicker.
- **`HypothesisBuffer` owns the LocalAgreement-2 algorithm.** Keep it pure: no Whisper, no Dexie, no async, no timers. The consumer worker is the orchestrator.
- **`return_timestamps: 'word'` is required** for LocalAgreement-2. Whisper words without timestamps are dropped (their position is unreliable).
- **Use `_timestamped` model variants only.** The default `onnx-community/whisper-*` ONNX exports do not include cross-attention outputs, and `return_timestamps: 'word'` throws `Model outputs must contain cross attentions to extract timestamps`.
- **Batch worker uses `sessionId` to ignore late results.** If the user clicks Record while a batch transcribe is in flight, the new session ID makes the eventual response a no-op.
- **VAD is the primary silence gate; RMS is the fallback only.** `collectAudioFrom` returns `speechProbability: undefined` for any window containing a chunk without a VAD verdict, forcing the consumer onto RMS. Do not "improve" this by max'ing over the verdicted subset; it silently drops real speech in the unverdicted chunks.
- **VAD load is fire-and-forget, recording must never block on it.** Init failure is allowed; the producer logs once per session and chunks ship with `speechProbability: undefined`.
- **ORT runtime is served by Vite from `node_modules`, never from `public/`.** Files in `public/` cannot be dynamically `import()`-ed; ORT's threaded loader uses `import()`. See `vite.config.ts::ortRuntime` and `src/lib/vad/silero/sileroVad.ts` (`ort.env.wasm.wasmPaths = '/ort/'`).

## Verification

- Smoke test: record 10 s of speech. The live panel streams words grey, then commits to black as LA-2 confirms. The right panel stays empty until Stop, then renders the batch transcript.
- Stop a recording with > 3 s of audio: batch panel shows `done · Xms / Ys` in the status badge.
- Compare panels: pick different models per panel, record once, and observe how the algorithms diverge on the same audio.
- Clear while stopped and while recording: both transcripts and the consumer's in-memory state reset.
- Record, stop, Record: a new Record wipes the previous session (live, batch, and audioArchive). The two transcripts are scoped to one recording.
- Swap a panel's model: only that panel's worker respawns. The other panel and any in-progress recording are unaffected.

## Out of scope

Diarization, language UI, transcript export, mobile layout, auth, sync. POC only.
