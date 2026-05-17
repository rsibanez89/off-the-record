# Benchmarks

Captured 2026-05-17 with `onnx-community/whisper-tiny.en_timestamped` on Node + onnxruntime-node CPU.

See `docs/AUTONOMOUS-LOOP.md` for the regression-driven improvement protocol.

Run `npm run bench` to refresh the candidate snapshot, `npm run bench:compare` to diff against this baseline, `npm run bench:promote` to update it.

## Per-fixture metrics

| Fixture | Duration | Ticks | WER vs gt | CER vs gt | WER offline | Streaming penalty | Hallucinations | Realtime ×
|---|---|---|---|---|---|---|---|---|
| jfk | 11 s | 11 | 4.55% | 1.92% | 4.55% | 0.00pp | 0 | 4.46× |
| synth | 5 s | 5 | 0.00% | 0.00% | 0.00% | 0.00pp | 0 | 3.83× |
| apollo11 | 25 s | 25 | 28.13% | 23.45% | 15.63% | 12.50pp | 0 | 5.13× |
| long | 335 s | 335 | 8.95% | 7.16% | 47.47% | -38.52pp | 0 | 5.14× |
| jfk-inaugural | 720 s | 720 | 19.78% | 15.71% | 58.39% | -38.61pp | 0 | 5.02× |

## Long-session stability (jfk-inaugural fixture)

- Tick count: 720 (712 inference, 7 force-slide, 1 hallucination-defer)
- Latency drift Q4/Q1: 1.00× (above 1.2× would suggest per-tick state growth or a memory leak)
- Realtime factor: 5.02×

## Decision rule

A candidate run is a **regression** if ANY fixture shows:
- `werVsGt` rose by more than 0.5pp
- `hallucinationStrings` increased
- `streamingPenaltyPp` rose by more than 2.0pp
- `latencyDriftQ4Q1` rose by more than 0.2× (long-session only)
- `realtimeFactor` dropped below 70% of baseline

A candidate is an **improvement** if at least one fixture's `werVsGt` dropped by ≥ 0.5pp AND no fixture regressed by the rule above.

Otherwise: **neutral**.
