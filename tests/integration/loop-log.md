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
