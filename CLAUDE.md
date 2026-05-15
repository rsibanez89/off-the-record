# CLAUDE.md

Project rules for agents working on **Off The Record**. Read `README.md` for what the app is, `PLAN.md` for the design spec.

## 🧭 Where to find things

- `README.md`: what the app is, how to run it.
- `PLAN.md`: design spec and behavioural contract.
- `src/workers/`:
  - `producer.worker.ts`: audio capture, resample to 16 kHz, dual-table IDB writes.
  - `consumer.worker.ts`: live LocalAgreement-2 streaming.
  - `batch.worker.ts`: one-shot Whisper run on Stop.
- `src/lib/db.ts`: Dexie schema, including `chunks`, `transcript`, and `audioArchive` tables.
- `src/lib/audio.ts`: model list, sample-rate constants, `isMultilingual`, `isTurbo`.
- `src/lib/transcription/`:
  - `hypothesisBuffer.ts`: LocalAgreement-2 (pure algorithm, no I/O).
  - `heuristics.ts`: silence and hallucination detection.
  - `whisperAdapter.ts`: pipeline creation, backend detection, `runWhisper`.
- `src/components/TranscriptPanel.tsx`: one transcript panel (transcript + model picker + status badge).
- `public/audio-worklet.js`: mic capture worklet.

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
- `MIN_WINDOW_S` is currently `0` in the consumer worker for fast user feedback (a grey wrong word at 1 s is preferable to 3 s of empty UI). The hallucination filter catches obvious junk before it commits.
- The producer writes to `chunks` and `audioArchive` in one Dexie transaction. Keep them together so a partial write cannot orphan a chunk in one table only.
