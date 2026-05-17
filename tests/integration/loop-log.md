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

## Session summary

- Attempts: 5 (the loop's hard cap).
- Committed: 1 (5.13 k-cap 5 -> 20). Baseline JFK-inaugural WER dropped from 28.06% to 27.55%.
- Reverted: 4 (5.14 drop-tentative, jitter 0.05, jitter 0.2, k-cap 50).
- Stopped because (a) hit the 5-attempt cap, (b) the last three iterations were all neutral/regression reverts (the protocol's other stop condition), and (c) the small-scoped levers in the in-scope list are exhausted at this baseline. The obvious remaining duplication on `long` likely needs an investigative pass on the actual streamed transcript rather than another blind parameter sweep.

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
