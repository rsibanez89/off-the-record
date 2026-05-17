# Autonomous improvement loop

This document is the operating manual for an agent that improves the
streaming-transcription pipeline autonomously, using the integration
benchmark as the regression gate.

The mechanism is simple: pick an improvement, implement it, run the
benchmark, and either commit (if quality improved) or revert (if it
regressed). The agent never needs human approval inside a loop iteration,
but it MUST follow the protocol below exactly.

## Pre-flight (once per loop run)

1. `git status` must show a clean working tree. If not, stop and report.
2. `npm run typecheck` must pass.
3. `npm test` must pass (unit tests).
4. `tests/integration/baselines.json` must exist. If it does not, run
   `npm run bench` then `npm run bench:promote` to bootstrap.

## Loop body (repeat up to 5 times or until no more improvements left)

### Step 1: pick an improvement

Read `../improvements.md` (kept outside the repo by design; it tracks intended
follow-up work without coupling product roadmap to source history). Choose
the highest-payoff item that:

- has not been attempted in this run,
- can be implemented in a single coherent change (no multi-day refactors),
- is not crossed off as "Done".

Record the choice in the loop log (see Step 7).

### Step 2: implement

Make ONLY the changes needed for that improvement. Keep edits minimal and
scoped. Do not refactor adjacent code unless the improvement requires it.

Update the `CHANGELOG` section in `../improvements.md` for the item:
status, files touched, brief notes.

### Step 3: verify locally

Run, in order:

```
npm run typecheck     # must exit 0
npm test              # all unit tests must pass
```

If either fails: fix the change. If you cannot fix it within a small number
of edits, `git restore .` and go to Step 1 with a different improvement.

### Step 4: run the benchmark

```
npm run bench
```

This:

1. Wipes `tests/integration/.bench/`.
2. Runs `vitest run --config vitest.integration.config.ts` (the slow path,
   ~5 minutes wall-clock).
3. Calls `scripts/bench-compare.mjs`, which writes
   `tests/integration/.bench/comparison.json` and prints a Markdown table.

If the test run itself fails (exit non-zero from vitest), the comparator
does not run. Treat that as a regression: `git restore .` and try another
improvement.

### Step 5: read the verdict

Open `tests/integration/.bench/comparison.json`. The `verdict` field is
one of:

- `"improvement"`: at least one fixture got better, none got worse.
- `"neutral"`: nothing meaningfully changed.
- `"regression"`: at least one fixture got worse beyond the tolerance.

The decision rule is in `BENCHMARKS.md` and replicated in
`scripts/bench-compare.mjs` at the top.

### Step 6: act on the verdict

**Improvement.** Promote and commit:

```
npm run bench:promote     # updates tests/integration/baselines.json + BENCHMARKS.md
git add -A
git commit -m "<improvement-id>: WER -Xpp on <fixture>, no regressions

<short description of the change>

WER fixtures (live, before -> after):
  jfk:           X.XX% -> X.XX%
  synth:         X.XX% -> X.XX%
  apollo11:      X.XX% -> X.XX%
  long:          X.XX% -> X.XX%
  jfk-inaugural: X.XX% -> X.XX%"
```

**Neutral.** Use judgement:

- If the change was a structural improvement (refactor, dead-code removal,
  better separation), keep it: `git add -A && git commit`.
- If it was a speculative optimisation that did not pay off, revert it:
  `git restore .`.

**Regression.** Revert immediately:

```
git restore .
git clean -fd
```

Then record what regressed and why in the loop log so future iterations
do not retry the same approach blindly.

### Step 7: append to the loop log

Append a row to `tests/integration/loop-log.md` (create if missing):

```
## <ISO timestamp> | <improvement id from improvements.md>
- Verdict: <improvement | neutral | regression>
- Action: <committed | kept-no-promote | reverted>
- WER deltas per fixture: <pasted from comparison.json>
- Notes: <one or two sentences>
```

This is the audit trail. A human reviewer can read it after the loop
finishes to understand what was tried.

### Step 8: continue or stop

- If improvements remain and you have made fewer than 5 attempts in this
  loop run: go to Step 1.
- If 3 consecutive iterations end in `regression` or `neutral-revert`:
  stop. Report.
- If wall-clock time exceeds 2 hours: stop. Report.
- If the backlog is empty: stop. Report.

## Hard constraints

The agent **must not**:

- Modify `tests/integration/baselines.json` directly. Promote only via
  `npm run bench:promote`.
- Modify `tests/integration/*.test.ts` thresholds to make a failing
  candidate pass. Thresholds reflect product expectations. If a threshold
  legitimately needs to move, that is a human decision, not a regression
  fix.
- Modify `tests/fixtures/*.wav`, `tests/fixtures/*.txt`, or
  `tests/fixtures/*.json`. The audio and ground-truth strings are the
  reference.
- Skip `npm run typecheck` or `npm test`.
- `git push` anything. Commits are local; the human decides when to push.
- `git rebase`, `git reset --hard`, `git push --force`, or any
  history-rewriting operation.
- Edit `docs/AUTONOMOUS-LOOP.md` (this file).
- Edit `BENCHMARKS.md` by hand (regenerated by `bench:promote`).

## What counts as an "improvement" from the backlog

`../improvements.md` ranks items by expected payoff. Examples that fit
the loop well (small, scoped, measurable):

- 3.2 pipeline `dispose()` cycle
- 5.2 pipeline next-tick during current inference
- 3.5 `writeTranscript` diff-only
- Adjustments to `CONTEXT_LOOKBACK_S`, `FAST_TRIM_THRESHOLD_S`,
  `VAD_SILENCE_THRESHOLD`
- `stripCommittedTailOverlap` k-gram tuning
- `dropAlreadyCovered` jitter constant adaptation

Items that do NOT fit the loop (require human judgement or are too large):

- 1.x model picker additions (those require download, license review)
- 2.x large refactors (LiveTranscriptionLoop is already extracted; 2.5
  WhisperEngine, 2.6 message-union split, 2.7 feature-flag centralisation
  are structural decisions for a human)
- 5.4 AlignAtt policy (research-grade)
- 5.6 KV cache reuse (blocked on upstream transformers.js)
- 5.7 speculative decoding (research)
- 1.7 sherpa-onnx engine (second engine; architectural)

Stick to the small, scoped items.

## Bootstrap (only once, by a human)

When this protocol is set up the first time:

```
npm run typecheck
npm test
npm run test:integration       # produces .bench/*.json
npm run bench:promote          # writes baselines.json and BENCHMARKS.md
git add tests/integration/baselines.json BENCHMARKS.md docs/AUTONOMOUS-LOOP.md
git commit -m "bootstrap benchmark baseline"
```

After that the loop can run.
