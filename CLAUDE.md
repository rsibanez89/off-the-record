# CLAUDE.md

Project rules for agents working on **Off The Record**. Read `README.md` for what the app is, `PLAN.md` for the design spec.

## 🧭 Where to find things

- `README.md`: what the app is, how to run it.
- `PLAN.md`: design spec and behavioural contract.
- `src/workers/`:
  - `producer.worker.ts`: audio capture, resample to 16 kHz, **Silero VAD per frame**, dual-table IDB writes with per-chunk `speechProbability`.
  - `consumer.worker.ts`: live LocalAgreement-2 streaming. Silence gate prefers VAD verdict, falls back to RMS.
  - `batch.worker.ts`: one-shot Whisper run on Stop.
- `src/lib/db.ts`: Dexie schema (v4), `chunks` and `audioArchive` carry `speechProbability?: number`, plus `transcript`.
- `src/lib/audio.ts`: model list, sample-rate constants, `isMultilingual`, `isTurbo`.
- `src/lib/config.ts`: `VAD_SILENCE_THRESHOLD` and other centralised live-transcription tunables.
- `src/lib/vad/`: in-house Silero VAD v5. `VadEngine` interface, `SileroVad`, `Framer`, `LinearResampler`, `VadStateMachine`, `NoopVadEngine`, `createVadEngine()`. See `src/lib/vad/README.md`.
- `src/lib/transcription/`:
  - `hypothesisBuffer.ts`: LocalAgreement-2 (pure algorithm, no I/O).
  - `heuristics.ts`: hallucination filter, sentence-end check, RMS fallback (`rms`, `isSilent`).
  - `whisperAdapter.ts`: pipeline creation, backend detection, `runWhisper`.
- `src/components/TranscriptPanel.tsx`: one transcript panel (transcript + model picker + status badge).
- `public/audio-worklet.js`: mic capture worklet.
- `scripts/fetch-models.mjs`: build-time provisioner for the Silero ONNX (sha256-pinned). Wired to `predev` and `prebuild`.
- `vite.config.ts::ortRuntime`: Vite plugin that serves `/ort/*` from `node_modules/onnxruntime-web/dist/` in dev and copies to `dist/ort/` at build.

## 🛑 Hard rules

- **`PLAN.md` is the behavioural contract.** Don't silently change streaming, commit, or window logic without updating it first.
- **`HypothesisBuffer` owns LocalAgreement-2.** Keep it pure: no Whisper, no Dexie, no async, no timers. The consumer worker is the orchestrator.
- **`whisperAdapter.ts` owns transformers.js setup.** Both the live consumer and the batch worker call `createPipeline` and `detectBackend` from here. Don't recreate them locally.
- **Use `_timestamped` model variants only.** The non-timestamped ONNX exports lack cross-attention outputs and break `return_timestamps: 'word'`. LocalAgreement-2 depends on word timestamps.
- **Don't touch the punctuation-stripping codepoint sets** in `src/lib/transcription/hypothesisBuffer.ts` (`STRIP_PUNCT_CODEPOINTS`) and `src/lib/transcription/heuristics.ts` (`isHallucinationWord`). They reference U+2013 (en-dash) and U+2014 (em-dash) as numeric codepoints because Whisper emits those characters literally in transcripts.
- **Model weights go in the Cache API, never IDB.** Transformers.js handles this. Do not override its caching.
- **Audio payloads go in IDB.** `BroadcastChannel('new-chunk')` is wake-up only. The batch worker is triggered by a direct message from the main thread, not via the wake-up channel.
- **`audioArchive` is the batch input.** The producer writes every chunk to both `chunks` (evictable) and `audioArchive` (kept). Do not evict `audioArchive` from any worker except via `clearAll()` on Record/Clear.
- **Encoder dtype: `fp32` everywhere except `large-v3-turbo` on WebGPU (uses `fp16`).** Decoder stays `q4`. `q4` on the encoder destroys accuracy.
- **Silero VAD is the primary silence gate.** The producer worker writes a `speechProbability` on every chunk. The consumer worker treats `speechProbability < VAD_SILENCE_THRESHOLD` as silence and bypasses Whisper. RMS (`isSilent` in `heuristics.ts`) is the fallback ONLY when the window contains any chunk without a VAD verdict (e.g. the first half-second while Silero loads). Do not regress to RMS as the primary gate.
- **One undefined chunk poisons the whole window.** `collectAudioFrom` returns `speechProbability: undefined` if any chunk in the window lacks a verdict, even if other chunks do. Treating partial coverage as "max over the verdicted subset" would silently drop real speech in the unverdicted chunks. Preserve this all-or-nothing rule.
- **VAD failure must not block recording.** `loadVad` in `producer.worker.ts` is fire-and-forget; on init failure, recording continues with `speechProbability: undefined` per chunk and the consumer falls back to RMS. A `console.warn` per session reminds the user that VAD is disabled.
- **ORT runtime is served from `node_modules` via `vite.config.ts::ortRuntime`.** Do not vendor ORT WASM/MJS into `public/`. Vite blocks dynamic `import()` from `public/`, which breaks ORT's threaded loader.
- **COOP/COEP headers are required** for `SharedArrayBuffer`. Vite sets them; any other host must too.
- **Don't enable Record during stop flush.** UI must wait for producer `stopped` and consumer `flush-done` before allowing the next Record.
- **Each Record starts a fresh session.** The live and batch transcripts are scoped to one recording. Don't re-introduce cross-session accumulation without a new feature request.
- **Batch worker uses `sessionId` to ignore late results.** Preserve this when changing the batch flow. If you remove it, a stale transcribe-done can stomp the current panel state.

## ⚙️ Commands

- `npm run dev`: dev server.
- `npm run typecheck`: TS check, no emit. Run before claiming a change is done.
- `npm run build`: production build to `dist/`.

## ✅ Smoke test

- Record 10 s of speech. Grey volatile text in the live panel settles into black committed text within ~1 to 2 inference ticks.
- Press Stop. The batch panel transitions: ready → transcribing → done · `<ms>` / `<s>`.
- Pick a different model in one panel (or both). The matching worker respawns and shows download progress; the other panel keeps working.
- Clear while idle and while recording. Both transcripts and the live consumer's in-memory state reset; `audioArchive` is wiped.
- Record again immediately after a previous Stop. Both panels reset and the new session is fully independent.

## ✍️ Communication style

- **Short sentences. Bullet points. Write for busy humans scanning the page, not reading paragraphs.**
- Use icons in section headers and log prefixes when they aid scanning. Don't sprinkle them randomly.
- **No em-dash (U+2014) or en-dash (U+2013) in prose.** Use a colon, comma, or period. The only exception is the punctuation codepoint sets in the transcription library (see Hard rules above).
- **No AI co-authoring signatures.** No "Co-Authored-By", "Generated with Claude", 🤖, or similar. Not in code, not in comments, not in commits.

## 🧠 Useful context

- The resampler in `producer.worker.ts` carries `resamplePos` and `prevTail` across frames. If you touch it, preserve the seam-continuity invariant.
- Renaming the Dexie DB, bumping the schema, or changing `localStorage` keys orphans browser state. Bump the Dexie version cleanly and document the migration.
- `MIN_WINDOW_S` is currently `0` in the consumer worker for fast user feedback (a grey wrong word at 1 s is preferable to 3 s of empty UI).
- The producer writes to `chunks` and `audioArchive` in one Dexie transaction. Keep them together so a partial write cannot orphan a chunk in one table only.

## 🎯 Live transcription design invariants

- **Audio anchor advances conservatively.** Only trim on sentence boundaries (`.`, `?`, `!`) or when the window grows past `FAST_TRIM_THRESHOLD_S`. Always keep `CONTEXT_LOOKBACK_S` of already-transcribed audio in the window. Aggressive per-word trimming kills accuracy on smaller models. The dual-panel comparison made this visible; do not regress it.
- **Silence gate order: VAD first, RMS as fallback.** Silero VAD's per-chunk `speechProbability` is the trusted gate when present. The window's verdict is `max(speechProbability)` across its chunks. If any chunk lacks a verdict, the window falls back to RMS so VAD-loading-induced gaps cannot silently drop speech.
- **Hallucination on non-silent audio is deferred, not discarded.** The silence gate runs before Whisper; anything reaching the hallucination check has real energy. Whisper outputting garbage on real audio is a model failure, not empty input. Wait for more context. `MAX_WINDOW_S` force-slide is the safety net.
- **Silence is the only path that drops audio without committing.** Keep it that way: silence is the one signal we can fully trust about "no transcribable content here".
