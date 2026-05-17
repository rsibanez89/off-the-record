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
