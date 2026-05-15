# Off The Record, Build Plan

## Your role

You are an expert in offline speech recognition and Whisper. You know transformers.js v3, WebGPU/ONNX runtime, Web Audio API, AudioWorklet, IndexedDB (Dexie), Cache API, and browser-side streaming transcription heuristics. Build a clean, minimal, working POC. Favour clarity over cleverness. No backend. Everything runs in the browser.

## Goal

A web app that records the mic and shows a live transcript powered by Whisper running locally in the browser. The app prefers WebGPU when available and falls back to WASM. Audio capture starts on the main thread, raw samples are sent through an AudioWorklet, and chunking/transcription are handled by two independent Web Workers. The producer and consumer coordinate through IndexedDB for audio payloads, with `BroadcastChannel('new-chunk')` used only as a wake-up signal.

## Architecture (three layers)

- **Producer worker** (`producer.worker.ts`)
  - Receives raw context-rate `Float32Array` frames from the main thread.
  - Linearly resamples to 16 kHz mono and buffers into 1-second chunks.
  - Flushes each chunk to IDB: `db.chunks.add({ startedAt, samples })`.
  - Posts `BroadcastChannel('new-chunk')` as a wake-up notification only. Audio payload never leaves IDB after this write.
  - Acknowledges `stop` with `stopped` so the main thread can wait before flushing the consumer.

- **Consumer worker** (`consumer.worker.ts`)
  - Lazy-loads the Whisper pipeline using transformers.js.
  - Detects WebGPU, otherwise uses WASM.
  - Uses fp32 encoder for non-turbo checkpoints, fp16 encoder for `large-v3-turbo` on WebGPU, and q4 merged decoder.
  - On wake-up: collects chunks from `committedAudioStartS` onward and runs inference once enough audio is available.
  - Applies inline longest-common-prefix stability against the previous hypothesis for final-vs-tentative colouring.
- Reconciles `db.transcript` from in-memory `committedWords`, stable words, and volatile words. It writes new rows before deleting excess rows so the UI does not flash through an empty/short table on every inference tick.
  - Deletes audio chunks older than the committed audio anchor.
  - Acknowledges `reset` with `reset-done` and `flush` with `flush-done`.

- **Main thread (React)**
  - Calls `getUserMedia`, owns `AudioContext`, loads `/audio-worklet.js`, and forwards worklet frames to the producer worker.
  - Renders display tokens sent directly by the consumer worker, with `db.transcript` via `dexie-react-hooks` `useLiveQuery` as the initial/fallback source.
  - Renders one `<span>` per token. `isFinal:true` becomes strong black. `isFinal:false` becomes soft grey.
  - Waveform: separate `AnalyserNode` on the live mic stream. Not via IDB. Visual must be zero-lag.
  - Model picker plus Record/Stop/Clear.
  - Stop enters a `stopping` state and waits for producer `stopped` and consumer `flush-done` before allowing another recording.

## Streaming transcript strategy

- Whisper is not used as a true streaming model. The consumer repeatedly transcribes a growing audio window.
- Minimum live window: 3 seconds. Stop drain window: 1.5 seconds. Maximum window: 24 seconds.
- `committedAudioStartS` anchors the left edge of audio that still needs transcription.
- `committedWords` stores permanently committed transcript words across recording sessions.
- `prevHypothesis` stores the previous inference result for the current audio region.
- The consumer computes a longest common prefix between `prevHypothesis` and the current hypothesis.
- Prefix words are written as `isFinal:1`; the rest are written as `isFinal:0`.
- If the whole hypothesis is stable for two consecutive ticks, the consumer commits it, advances the audio anchor to the end of that window, clears consumed chunks, and starts fresh.
- Whenever words are permanently committed, the consumer de-duplicates overlap by comparing the tail of `committedWords` with the head of the new hypothesis and appending only the non-overlapping suffix.
- At the maximum window, the consumer commits the whole current hypothesis and restarts the audio window after it as a fallback. Whisper does not provide word timestamps in this mode, so the app avoids guessing how to split words by audio fraction.
- Standalone leading dash artifacts from Whisper, such as `- Absolutely`, are ignored during tokenization/stability checks so formatting jitter does not prevent a stable commit.
- Silence and common Whisper filler/hallucination outputs commit the previous hypothesis, advance the anchor, and drop consumed chunks.
- On `stop`, the consumer drains available chunks, folds the final hypothesis into `committedWords`, rewrites the transcript as final, clears chunks, and posts `flush-done`.

## Tech stack

- Vite plus React 19 plus TypeScript.
- Tailwind. Minimalist: large transcript area, small waveform strip, one model dropdown.
- `@huggingface/transformers` (v3+) for Whisper inference.
- Dexie for IndexedDB. `lucide-react` for icons.
- No router, no state library. Dexie `liveQuery` plus React state is enough.

## Storage rules

- **Model weights live in the Cache API**, not IDB. IDB silently fails past ~1 GB on some browsers; Whisper shards exceed that. transformers.js handles this automatically. Do NOT override its caching.
- **Audio chunks live in IDB** (Dexie). Each chunk is small (<200 KB at 16 kHz mono Float32, 1 s).
- **Transcript tokens live in IDB**.
- Drop chunks older than the current window's tail to bound storage.

## Model picker

- Three options minimum, dropdown in UI:
  - `onnx-community/whisper-tiny.en` (fast, weaker)
  - `onnx-community/whisper-base.en`
  - `onnx-community/whisper-large-v3-turbo` (slower first load, best quality)
- Selection persisted in `localStorage`. Changing model: terminate then respawn consumer worker.
- Show download progress on first load (transformers.js exposes a progress callback).

## File layout

```
src/
├── main.tsx, App.tsx
├── components/   Transcript.tsx, Waveform.tsx, ModelPicker.tsx
├── workers/      producer.worker.ts, consumer.worker.ts
└── lib/          db.ts (Dexie), audio.ts

public/
└── audio-worklet.js
```

Plus root: `index.html`, `vite.config.ts` (COOP/COEP headers), `package.json`, `tsconfig.json`, `tailwind.config.ts`.

## Critical gotchas (do not relearn)

- **COOP/COEP headers required** for `SharedArrayBuffer` (transformers.js needs it). Vite dev server must set `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`.
- **q4 on the encoder destroys accuracy** in noise/accents. Use `q4` on `decoder_model_merged` only. Keep the encoder at `fp32` except for `large-v3-turbo` on WebGPU, where a published `fp16` encoder is used.
- **WebGPU adapter check** on startup. Fall back to WASM with a visible warning.
- **Producer and consumer must stay decoupled for audio payloads.** IDB is the audio contract; `BroadcastChannel` is only a wake-up.
- **Do not allow Record during stop flush.** The UI must wait for producer `stopped` and consumer `flush-done`, otherwise old and new transcript rewrites can flicker.
- **No separate stability module.** The longest-common-prefix stability logic currently lives inline in `consumer.worker.ts`; do not extract a helper unless there is a real reuse need.

## Verification

- Smoke test: record 10 s of speech. Soft grey text settles into strong black once consecutive hypotheses agree.
- Record, stop, record, stop: all words stay in one transcript, with no flicker between previous-only/current-only and combined output.
- Clear while stopped and while recording: transcript and worker in-memory state both reset.
- Swap model: consumer respawns and starts with a fresh transcript/chunk state.

## Out of scope

Diarization, language UI, export, mobile layout, auth, sync. POC only.
