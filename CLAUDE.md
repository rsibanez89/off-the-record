# CLAUDE.md

Project rules for agents working on **Off The Record**. Read `README.md` for what the app is, `PLAN.md` for the design spec.

## 🧭 Where to find things

- `README.md`: what the app is, how to run it.
- `PLAN.md`: design spec and behavioural contract.
- `src/workers/`: producer (audio capture, resample, IDB writes) and consumer (Whisper inference, transcript reconciliation).
- `src/lib/db.ts`: Dexie schema.
- `src/lib/audio.ts`: model list, sample-rate constants.
- `public/audio-worklet.js`: mic capture worklet.

## 🛑 Hard rules

- **`PLAN.md` is the behavioural contract.** Don't silently change streaming, commit, or window logic without updating it first.
- **Don't touch the regex character classes in `src/workers/consumer.worker.ts:28` and `:50`.** Those en-dash (U+2013) and em-dash (U+2014) literals match real Whisper output. Removing them breaks hallucination filtering.
- **Model weights go in the Cache API, never IDB.** Transformers.js handles this. Do not override its caching.
- **Audio payloads go in IDB.** `BroadcastChannel('new-chunk')` is wake-up only.
- **Encoder dtype: `fp32` everywhere except `large-v3-turbo` on WebGPU (uses `fp16`).** Decoder stays `q4`. `q4` on the encoder destroys accuracy.
- **COOP/COEP headers are required** for `SharedArrayBuffer`. Vite sets them; any other host must too.
- **Don't enable Record during stop flush.** UI must wait for producer `stopped` and consumer `flush-done`.

## ⚙️ Commands

- `npm run dev`: dev server.
- `npm run typecheck`: TS check, no emit. Run before claiming a change is done.
- `npm run build`: production build to `dist/`.

## ✅ Smoke test

- Record 10 s of speech. Grey volatile text should settle into black committed text after a couple of inference ticks.
- Record, stop, record, stop. All words should stay in one transcript with no flicker.
- Clear while idle and while recording. Both should reset transcript and worker state.
- Swap model. Consumer respawns, transcript resets, chunks cleared.

## ✍️ Communication style

- **Short sentences. Bullet points. Write for busy humans scanning the page, not reading paragraphs.**
- Use icons in section headers and log prefixes when they aid scanning. Don't sprinkle them randomly.
- **No em-dash (U+2014) or en-dash (U+2013) in prose.** Use a colon, comma, or period. The only exception is regex character classes that match real punctuation (see Hard rules above).
- **No AI co-authoring signatures.** No "Co-Authored-By", "Generated with Claude", 🤖, or similar. Not in code, not in comments, not in commits.

## 🧠 Useful context

- The streaming-transcript heuristics (longest-common-prefix stability, silence/hallucination commit, force-slide at MAX_WINDOW_S) live inline in `consumer.worker.ts` on purpose. Don't extract them into a helper module unless there is a real reuse need.
- The resampler in `producer.worker.ts` carries `resamplePos` and `prevTail` across frames. If you touch it, preserve the seam-continuity invariant.
- Renaming the Dexie DB or the `localStorage` model key orphans browser state. Bump the schema or document the migration.
