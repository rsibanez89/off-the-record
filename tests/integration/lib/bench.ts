// Records per-fixture benchmark metrics to disk so an external script
// (`scripts/bench-compare.mjs`) can compare a candidate run against the
// committed baseline.
//
// Each speech-fixture integration test calls `recordFixtureMetrics()` from
// its `beforeAll`. The metrics dictionary captures everything the
// autonomous-improvement loop needs to decide "better, worse, or
// equivalent" without re-running the tests.
//
// Output: `tests/integration/.bench/<fixtureId>.json` (gitignored).
// The aggregator reads all of them into a single snapshot.

import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = fileURLToPath(new URL('.', import.meta.url));
const BENCH_DIR = join(HERE, '..', '.bench');

export interface FixtureMetrics {
  /** Stable id for the fixture (matches `<id>` in baselines.json). */
  fixtureId: string;
  /** Whisper model used to produce the metrics. */
  model: string;
  /** Audio duration of the fixture in seconds (post-trim, post-pad). */
  durationS: number;

  /** Total number of `tick()` calls the harness made. */
  ticks: number;
  /** Of those, how many ran Whisper inference. */
  inferenceTicks: number;
  /** Force-slide safety-net activations. */
  forceSlideTicks: number;
  /** Ticks that deferred due to `isHallucinationLine`. */
  hallucinationDeferTicks: number;

  // Quality.

  /** WER between ground truth and the LIVE streaming transcript. */
  werVsGt: number;
  /** CER between ground truth and the LIVE streaming transcript. */
  cerVsGt: number;
  /** WER between ground truth and the OFFLINE (one-shot) Whisper transcript. */
  werOfflineVsGt: number;
  /**
   * Streaming penalty in percentage points, per Macháček et al. 2023:
   * `werVsGt - werOfflineVsGt`. Negative means streaming beat offline.
   */
  streamingPenaltyPp: number;
  /** Count of known-hallucination substrings in the LIVE transcript. */
  hallucinationStrings: number;

  // Performance.

  /** Sum of `inferenceMs` across all ticks. */
  totalInferenceMs: number;
  /** `durationS * 1000 / totalInferenceMs`. >1 means faster than real time. */
  realtimeFactor: number;
  /**
   * Long-session-only: mean Q4 inference / mean Q1 inference. >1 means
   * later ticks are slower than earlier ticks (memory leak canary).
   * Null on short fixtures where the metric is meaningless.
   */
  latencyDriftQ4Q1: number | null;
}

export function recordFixtureMetrics(metrics: FixtureMetrics): void {
  mkdirSync(BENCH_DIR, { recursive: true });
  const path = join(BENCH_DIR, `${metrics.fixtureId}.json`);
  writeFileSync(path, JSON.stringify(metrics, null, 2) + '\n');
}
