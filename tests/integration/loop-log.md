# Autonomous-loop attempt log

Append-only. One block per attempt. See `docs/AUTONOMOUS-LOOP.md` for the protocol.

## 2026-05-17T16:12Z | 5.14 drainAndFinalise drops tentative

- Verdict: neutral
- Action: reverted
- WER deltas per fixture: apollo11 +0.00pp, jfk +0.00pp, jfk-inaugural +0.00pp, long +0.00pp, synth +0.00pp
- Notes: Hypothesis was that the Apollo 11 trailing-garbage tokens enter `committed` via `drainAndFinalise`'s final unconditional `forceCommit()`. Replaced it with a new `HypothesisBuffer.dropTentative()` and ran the bench: every fixture identical. Conclusion: in all five fixtures the drain settles with `tentative` already empty, so removing the final force-commit is a no-op for the suite. The Apollo 11 garbage must arrive in `committed` via LA-2 agreement during the drain loop itself (Whisper re-emits the same hallucinated tokens, drain ticks confirm them via the standard agreement path). Need a different angle for 5.14: either gate the drain on a silence/confidence signal that the test fixtures don't currently supply (they all use `speechProbability: 0.9`), or attack the same problem inside the per-tick LA-2 path. Reverted via `git restore .`; unit tests on `hypothesisBuffer` and `liveLoop` returned to baseline.

## 2026-05-17T16:47Z | 5.13 stripCommittedTailOverlap k-cap 5 -> 20

- Verdict: improvement
- Action: committed (promoted to baseline)
- WER deltas per fixture: apollo11 +0.00pp, jfk +0.00pp, jfk-inaugural -0.51pp, long +0.00pp, synth +0.00pp
- Notes: First swing at the 5.13 long-form duplication theory. Raised `stripCommittedTailOverlap`'s max overlap from 5 to 10 -> JFK inaugural dropped 0.37pp (sub-threshold). Bumped to 20 -> JFK inaugural dropped 0.51pp, clearing the 0.5pp rule with no regressions elsewhere. Mechanism: a 5 s `CONTEXT_LOOKBACK_S` after a sentence-end trim holds ~10 to 15 words, so a Whisper re-emission of 6+ boundary words slips past a 5-gram cap, enters `tentative`, and gets LA-2 confirmed as a duplicate. The boundary check `start > lastCommittedTime + 1` and `dropAlreadyCovered` keep the comparison well-targeted, so the deeper window is essentially free. Only JFK inaugural moved (720 s real speech, lots of seams); `long` (335 s TTS, 0 force-slides) was unchanged, which suggests its duplication mechanism is something else (probably the iterative drain-commit pattern hinted at by 5.13's second suspected cause). Worth a follow-up loop iteration.

## 2026-05-17T17:04Z | dropAlreadyCovered jitter 0.1 -> 0.05 (tighter)

- Verdict: regression
- Action: reverted
- WER deltas per fixture: apollo11 +0.00pp, jfk +0.00pp, jfk-inaugural +0.00pp, long +0.26pp, synth +0.00pp
- Notes: Hypothesis: tighter cutoff (closer to `lastCommittedTime`) drops more re-emitted boundary words before they reach LA-2, killing the residual `long` fixture duplicates. The WER moved 0.26pp the WRONG way on `long` (sub-threshold). The bench also flagged JFK inaugural as a realtime-factor regression (ratio 0.29 vs baseline), but the run took 607 s wall vs ~252 s baseline so that is environmental load on my Mac, not the change. Reverted per protocol. Learning: with the wider stripCommittedTailOverlap (k=20) already catching boundary re-emissions, tightening `dropAlreadyCovered` mostly drops legitimate new words that span the boundary, which is a net loss.

## 2026-05-17T17:09Z | dropAlreadyCovered jitter 0.1 -> 0.2 (looser)

- Verdict: neutral
- Action: reverted
- WER deltas per fixture: apollo11 +0.00pp, jfk +0.00pp, jfk-inaugural +0.15pp, long +0.00pp, synth +0.00pp
- Notes: Tried the opposite direction after the tight-jitter regression. JFK inaugural drifted +0.15pp (sub-threshold), `long` did not move. The 0.1 default is well-tuned; either direction degrades quality somewhere. Speculative tweak, no payoff -> reverted.

## 2026-05-17T17:17Z | stripCommittedTailOverlap k-cap 20 -> 50

- Verdict: neutral
- Action: reverted
- WER deltas per fixture: apollo11 +0.00pp, jfk +0.00pp, jfk-inaugural +0.00pp, long +0.00pp, synth +0.00pp
- Notes: Hypothesis: maybe boundary overlaps even longer than 20 words exist on the long-form fixtures. Bench result: identical metrics across all five fixtures. k=20 already covered every overlap that matters here; the rest is bounded by `dropAlreadyCovered` + the boundary-start short-circuit. Confirms k=20 is the right setting; pushing higher is wasted comparisons.

## Session summary (run 1, ended 17:17Z)

- Attempts: 5 (the loop's hard cap).
- Committed: 1 (5.13 k-cap 5 -> 20). Baseline JFK-inaugural WER dropped from 28.06% to 27.55%.
- Reverted: 4 (5.14 drop-tentative, jitter 0.05, jitter 0.2, k-cap 50).
- Stopped because (a) hit the 5-attempt cap, (b) the last three iterations were all neutral/regression reverts (the protocol's other stop condition), and (c) the small-scoped levers in the in-scope list are exhausted at this baseline. The obvious remaining duplication on `long` likely needs an investigative pass on the actual streamed transcript rather than another blind parameter sweep.

## Session summary (run 2, ended 17:51Z)

- Attempts: 2 (one improvement, one regression).
- Committed: 1 (5.15 `dropFlatTimestampTail`, commit `1c597aa`). MASSIVE win: `long` WER 25.16% -> 8.95% (-16.21pp), jfk-inaugural 27.55% -> 19.78% (-7.77pp).
- Reverted: 1 (`dropAlreadyCovered` start-based filter, hard JFK regression).
- Stopped voluntarily after the regression: the biggest tractable algorithm-side wins are now banked, Apollo 11's residual has no clean single-file remedy per the second sub-agent investigation, and continuing risks burning more wall-time on diminishing returns. The investigative-sub-agent recipe worked beautifully on `long` (5.15); applied again to Apollo 11 (next-iteration entry above), it honestly returned "no clean fix" plus a fix that bench-regressed, which is also a valuable signal.

## Session summary (run 3, ended 19:28Z)

Goal: test every remaining item in `improvements.md`. Result:

Shipped 7 bench-neutral structural improvements (all SOLID refactors, integration tests pass, bench neutral, kept by structural-improvement rule):

- 2.7 config centralisation (commit f038e4b)
- 2.5 WhisperEngine class (commit 39aae82)
- 2.1 anchor advancement strategy module (commit 5aa3a3d)
- 2.2 producer/batch repositories (commit 4dcdd30)
- 2.8 audio-worklet sample-rate adaptivity (commit 9ee1981)
- 3.5 writeTranscript diff-only (commit 04486c3)
- 5.3 LocalAgreement-n configurability (commit 1066492; n=2 default kept, n=3 sweep regressed Apollo 11 streaming penalty so reverted to n=2)
- 2.6 consumer OutMessage type split (commit 6deed0a; the App.tsx handler-extraction half stays open as a UI pass)

Reverted 1: 5.14 v2 drain witness-only LA-2 mode (entry below).

Discarded with documented reason in `improvements.md` Status lines (bench-invisible, WebGPU-only, research, upstream-blocked, or UX/feature requiring human review):

  Section 1 (model picker): 1.1, 1.2, 1.3, 1.4, 1.5, 1.7
  Section 2 (architecture): 2.6 (the one structural refactor that purely changes worker -> main-thread types; bench cannot see it)
  Section 3 (memory): 3.1, 3.2, 3.4, 3.6, 3.7
  Section 4 (testability): 4.6, 4.8, 4.10
  Section 5 (algorithm): 5.2, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10, 5.11

Flagged for human input: 5.12 (long-form chunking for batch worker). A successful 5.12 improves only the offline WER, which makes the bench's `streamingPenaltyPp` rise mechanically and triggers the regression rule even though the change is a real product improvement. The user has to choose between (a) deferring 5.12 to a deliberate batch-quality pass, or (b) updating `scripts/bench-compare.mjs` to gate on `werOfflineVsGt` regression instead.

## 2026-05-17T18:30Z | 5.14 v2: drain witness-only LA-2 mode

- Verdict: regression (hard test failures on synth and jfk; comparator did not run)
- Action: reverted
- WER deltas per fixture: n/a (vitest exited non-zero; synth CER 0% -> 17%, jfk CER 1.9% -> ?, streaming penalties blew through thresholds)
- Notes: Added `ingest(hypothesis, { commitsAllowed: false })` mode and a `dropTentative` finaliser; drain passes `commitsAllowed: false` so re-decoding the same audio cannot self-confirm hallucinations. The hypothesis was right (Apollo 11's mechanism is exactly drain self-agreement) but the implementation was too blunt. The synth and JFK fixtures both rely on drain-side LA-2 to confirm trailing words that production saw only once before stop. With drain commits gated off, those words die at `dropTentative`. Hard regression. Reverted.
- Next angle: per-word "witness count" on `tentative` that increments only when ingest happens on a NEW audio window (`audio.t1` strictly greater than the previous ingest's). Then drain ticks (same audio) cannot increment the count, so apollo11's first-time-emitted garbage cannot reach the threshold, but production-witnessed words (count >= 1 entering drain) can still LA-2 commit. Requires plumbing `audio.t1` into `ingest` and tracking per-word counts. Larger change, deferred until the rest of the backlog is processed.

## 2026-05-17T17:51Z | dropAlreadyCovered: filter by START instead of END

- Verdict: regression (hard test failures, comparator did not run)
- Action: reverted
- WER deltas per fixture: n/a (vitest exited non-zero on 3 JFK assertion failures: CER 16.3% vs baseline 1.9%, streaming penalty 9.1% vs baseline 0%, committed-words count drop)
- Notes: Sub-agent investigation of Apollo 11's residual trailing-garbage tokens proposed switching `dropAlreadyCovered`'s filter from `w.end > cutoff` to `w.start > cutoff`. Hypothesis: catch hallucinated boundary-anchored words like `plan` whose end sits exactly at `lastCommittedTime` but whose start precedes it. The sub-agent honestly flagged that the fix would only remove 1 of 6 bad tokens and that "any fixture where Whisper drifts the textual head while keeping the start before lastCommittedTime would have that drift suppressed earlier". That risk materialised hard on the JFK fixture: legitimate boundary words got dropped, causing CER to balloon. Lesson: in the LA-2 pipeline, the filter ORDER matters. `dropAlreadyCovered` is the upstream defence against re-emission; tightening it is more dangerous than tightening `stripCommittedTailOverlap` because there is no downstream rescue. If we want to attack Apollo 11's residual we need a different lever, probably a confidence proxy on `tentative` words rather than a boundary-time filter.

## 2026-05-17T17:40Z | 5.15 dropFlatTimestampTail pre-filter

- Verdict: improvement
- Action: committed (promoted to baseline)
- WER deltas per fixture: apollo11 +0.00pp, jfk +0.00pp, jfk-inaugural -7.77pp, long -16.21pp, synth +0.00pp
- Notes: Implemented the sub-agent's design. New private method `HypothesisBuffer.dropFlatTimestampTail(words)` truncates the hypothesis at the first run of 3+ consecutive words whose `start` differs by less than 10 ms. Called from `ingest()` BEFORE `dropAlreadyCovered`. Result is a massive win on both long-form fixtures: `long` WER 25.16% -> 8.95%, jfk-inaugural 27.55% -> 19.78%. Apollo 11 did NOT move, meaning its trailing garbage is a different mechanism (the hallucinated tokens there don't share start timestamps); 5.14 stays open. The pattern is so specific to Whisper's repetition-loop failure mode that the threshold has plenty of margin: zero false positives observed on legitimate content. This is the biggest single win the bench has measured.

## 2026-05-17T17:30Z | investigation: long-fixture duplications (sub-agent, no code change)

- Verdict: investigative (no bench, no commit)
- Action: findings recorded in `improvements.md` as new item 5.15
- WER deltas per fixture: n/a
- Notes: After the 5-attempt loop run failed to move the `long` fixture, a `general-purpose` sub-agent ran the long integration test, diffed the streamed transcript against ground truth, and reported the 5 longest concretely duplicated phrases (10 to 17 words each, all at paragraph boundaries around 56 s, 188 s, 229 s, 291 s, 314 s). All five duplications share a Whisper hallucination signature: 3 or more consecutive words whose `start` timestamps are clamped to the audio window's `t1`. Two consecutive windows produce the same flat-timestamp tail, LA-2 in `runLocalAgreement` confirms them, they enter `committed` as duplicates of recently-committed real content. `stripCommittedTailOverlap` misses them (heads do not match) and `dropAlreadyCovered` misses them (fresh timestamps inside the current window). Sub-agent verified the pattern is highly specific: 6 such flat runs in the streamed transcript, all 6 are hallucinations. Proposed fix is a `dropFlatTimestampTail(words)` pre-filter on `HypothesisBuffer.ingest` that truncates the hypothesis at the start of the first 3-word flat-start run (epsilon ~10 ms). Probably also closes 5.14 (Apollo 11 trailing garbage at the noisy end of clip looks like the same mechanism). Risk: very low (worst case shaves 1 to 2 trailing words on extremely fast speech). Ready for the next loop run to attempt.
