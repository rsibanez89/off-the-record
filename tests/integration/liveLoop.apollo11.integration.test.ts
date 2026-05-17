// Integration test: drives `LiveTranscriptionLoop` on the Apollo 11 liftoff
// countdown (25 s of narration over launch-control radio and Saturn V engine
// audio). NASA-produced, public domain. This is the "speech plus background
// noise" fixture in the suite.
//
// Reports end-to-end quality AND streaming penalty (vs offline reference).
// See `liveLoop.jfk.integration.test.ts` for the rationale on those two
// metrics.

import { beforeAll, describe, expect, it } from 'vitest';
import { join } from 'node:path';
import type { ModelId } from '../../src/lib/audio';
import { createNodeWhisperPipeline } from './lib/whisperNode';
import { chunkIntoSeconds, loadGroundtruth, loadWav16kMono } from './lib/fixture';
import { runLiveOnFixture, type RunResult } from './lib/runLiveOnFixture';
import { runOfflineOnFixture, type OfflineResult } from './lib/runOfflineOnFixture';
import {
  cer,
  countHallucinationMatches,
  normalizeForScoring,
  wer,
} from './lib/metrics';
import { recordFixtureMetrics } from './lib/bench';

const MODEL_ID: ModelId = 'onnx-community/whisper-tiny.en_timestamped';
const FIXTURE_DIR = join(__dirname, '..', 'fixtures');

describe('Apollo 11 liftoff clip (tiny.en, Node)', () => {
  let liveRun: RunResult;
  let offline: OfflineResult;
  let groundtruth: ReturnType<typeof loadGroundtruth>;

  beforeAll(async () => {
    const pipeline = await createNodeWhisperPipeline(MODEL_ID);
    const fixture = loadWav16kMono(join(FIXTURE_DIR, 'apollo11-liftoff.wav'), 'apollo11-liftoff');
    groundtruth = loadGroundtruth(join(FIXTURE_DIR, 'apollo11-liftoff.json'));

    offline = await runOfflineOnFixture({
      pipeline,
      modelId: MODEL_ID,
      audio: fixture.samples,
    });

    const chunks = chunkIntoSeconds(fixture.samples, fixture.sampleRateHz);
    liveRun = await runLiveOnFixture({ pipeline, modelId: MODEL_ID, chunks });

    const e2eWer = wer(groundtruth.transcript, liveRun.liveTranscript);
    const e2eCer = cer(groundtruth.transcript, liveRun.liveTranscript);
    const offlineWer = wer(groundtruth.transcript, offline.transcript);
    const halluc = countHallucinationMatches(liveRun.liveTranscript);

    /* eslint-disable no-console */
    console.log('\n  ===== Apollo 11 / tiny.en =====');
    console.log(`  ground truth         : ${normalizeForScoring(groundtruth.transcript)}`);
    console.log(`  offline (model)      : ${normalizeForScoring(offline.transcript)}`);
    console.log(`  streaming (live)     : ${normalizeForScoring(liveRun.liveTranscript)}`);
    console.log('  --- end-to-end (model + algorithm) ---');
    console.log(`  WER (gt vs live)     : ${(e2eWer.rate * 100).toFixed(2)}%  (${e2eWer.edits}/${e2eWer.referenceLength})`);
    console.log(`  CER (gt vs live)     : ${(e2eCer.rate * 100).toFixed(2)}%`);
    console.log('  --- model only (informational) ---');
    console.log(`  WER (gt vs offline)  : ${(offlineWer.rate * 100).toFixed(2)}%`);
    console.log('  --- streaming penalty (algorithm only) ---');
    console.log(`  penalty (pp)         : ${((e2eWer.rate - offlineWer.rate) * 100).toFixed(2)}`);
    console.log(`  hallucinations       : ${halluc}`);
    console.log(`  ticks                : ${liveRun.ticks.length}, total inference: ${liveRun.totalInferenceMs.toFixed(0)} ms`);
    /* eslint-enable no-console */

    const histogram: Record<string, number> = {};
    for (const t of liveRun.ticks) histogram[t.kind] = (histogram[t.kind] ?? 0) + 1;
    recordFixtureMetrics({
      fixtureId: 'apollo11',
      model: MODEL_ID,
      durationS: liveRun.totalChunks,
      ticks: liveRun.ticks.length,
      inferenceTicks: histogram['inference'] ?? 0,
      forceSlideTicks: histogram['force-slide'] ?? 0,
      hallucinationDeferTicks: histogram['hallucination'] ?? 0,
      werVsGt: e2eWer.rate,
      cerVsGt: e2eCer.rate,
      werOfflineVsGt: offlineWer.rate,
      streamingPenaltyPp: (e2eWer.rate - offlineWer.rate) * 100,
      hallucinationStrings: halluc,
      totalInferenceMs: liveRun.totalInferenceMs,
      realtimeFactor: (liveRun.totalChunks * 1000) / liveRun.totalInferenceMs,
      latencyDriftQ4Q1: null,
    });
  }, 120_000);

  describe('end-to-end quality (model + streaming, vs ground truth)', () => {
    it('produces a non-empty committed transcript', () => {
      expect(liveRun.committedWords.length).toBeGreaterThan(15);
    });

    // Baseline 2026-05-17: WER 28.13%, CER 23.45%, 0 hallucinations.
    //
    // tiny.en has consistent quirks on this clip:
    //   "Ignition sequence start" gets a trailing "s" ("starts").
    //   "Lift off on Apollo 11" mishears as "Lift up on Apollo 11".
    //   "Tower cleared" becomes "Power cleared".
    //
    // Additionally, the streaming pipeline commits extra trailing tokens
    // that the offline pass drops (e.g. "plan to a gotta roll program"),
    // likely because `drainAndFinalise()` force-commits whatever is in
    // the tentative buffer at end-of-clip, including garbage. Some of it
    // (the "roll program" portion) is real Houston/Apollo audio under
    // the engine noise that the offline path filters out. The rest is
    // hallucination that the drain commits without an LA-2 confirmation
    // tick. Worth investigating; see handoff notes.
    it('WER stays under 35%', () => {
      expect(wer(groundtruth.transcript, liveRun.liveTranscript).rate).toBeLessThan(0.35);
    });

    it('CER stays under 30%', () => {
      expect(cer(groundtruth.transcript, liveRun.liveTranscript).rate).toBeLessThan(0.30);
    });

    it('emits zero known-hallucination substrings', () => {
      expect(countHallucinationMatches(liveRun.liveTranscript)).toBe(0);
    });
  });

  describe('streaming penalty (algorithm only)', () => {
    // penalty = WER(gt, live) - WER(gt, offline). See JFK test for
    // rationale. On noisy real-world audio the streaming penalty is
    // non-trivial (baseline 12.50pp on 2026-05-17) because the drain
    // force-commits a garbage tentative buffer at end-of-clip. The
    // offline path does not have this failure mode since it sees the
    // whole audio at once. Threshold gives ~5pp safety margin over the
    // baseline to detect further regressions.
    it('streaming penalty stays under 18 percentage points', () => {
      const liveWer = wer(groundtruth.transcript, liveRun.liveTranscript).rate;
      const offlineWer = wer(groundtruth.transcript, offline.transcript).rate;
      expect(liveWer - offlineWer).toBeLessThan(0.18);
    });
  });
});
