// Integration test: drives `LiveTranscriptionLoop` on the JFK clip with a
// real `tiny.en_timestamped` Whisper pipeline (via `onnxruntime-node`).
//
// Reports TWO metrics:
//
//   1. End-to-end quality: `WER(ground truth, streaming)`. Captures "model
//      plus our streaming code" combined. Useful for product decisions
//      like "should we use tiny.en or base.en for live?".
//
//   2. Streaming penalty: `WER(offline, streaming)`. Subtracts model
//      accuracy and isolates what LocalAgreement-2 / anchor /
//      silence-gate / drain code adds on top. Useful for regression
//      detection on OUR code.

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

describe('JFK clip (tiny.en, Node)', () => {
  let liveRun: RunResult;
  let offline: OfflineResult;
  let groundtruth: ReturnType<typeof loadGroundtruth>;

  beforeAll(async () => {
    const pipeline = await createNodeWhisperPipeline(MODEL_ID);
    const fixture = loadWav16kMono(join(FIXTURE_DIR, 'jfk.16k.wav'), 'jfk');
    groundtruth = loadGroundtruth(join(FIXTURE_DIR, 'jfk.json'));

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
    const streamingPenalty = wer(offline.transcript, liveRun.liveTranscript);
    const halluc = countHallucinationMatches(liveRun.liveTranscript);

    /* eslint-disable no-console */
    console.log('\n  ===== JFK / tiny.en =====');
    console.log(`  ground truth         : ${normalizeForScoring(groundtruth.transcript)}`);
    console.log(`  offline (model)      : ${normalizeForScoring(offline.transcript)}`);
    console.log(`  streaming (live)     : ${normalizeForScoring(liveRun.liveTranscript)}`);
    console.log('  --- end-to-end (model + algorithm) ---');
    console.log(`  WER (gt vs live)     : ${(e2eWer.rate * 100).toFixed(2)}%  (${e2eWer.edits}/${e2eWer.referenceLength})`);
    console.log(`  CER (gt vs live)     : ${(e2eCer.rate * 100).toFixed(2)}%`);
    console.log('  --- model only (informational) ---');
    console.log(`  WER (gt vs offline)  : ${(offlineWer.rate * 100).toFixed(2)}%`);
    console.log('  --- streaming penalty (algorithm only) ---');
    console.log(`  WER (offline vs live): ${(streamingPenalty.rate * 100).toFixed(2)}%  (${streamingPenalty.edits}/${streamingPenalty.referenceLength})`);
    console.log(`  hallucinations       : ${halluc}`);
    console.log(`  ticks                : ${liveRun.ticks.length}, total inference: ${liveRun.totalInferenceMs.toFixed(0)} ms`);
    /* eslint-enable no-console */

    const histogram: Record<string, number> = {};
    for (const t of liveRun.ticks) histogram[t.kind] = (histogram[t.kind] ?? 0) + 1;
    recordFixtureMetrics({
      fixtureId: 'jfk',
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
      expect(liveRun.committedWords.length).toBeGreaterThan(5);
      expect(liveRun.liveTranscript.trim().length).toBeGreaterThan(0);
    });

    // Baseline 2026-05-17: 4.55% (1 edit, "for" became "to"). Threshold
    // has 2x headroom for natural variance across ORT versions.
    it('WER stays under 10%', () => {
      expect(wer(groundtruth.transcript, liveRun.liveTranscript).rate).toBeLessThan(0.10);
    });

    it('CER stays under 5%', () => {
      expect(cer(groundtruth.transcript, liveRun.liveTranscript).rate).toBeLessThan(0.05);
    });

    it('emits zero known-hallucination substrings', () => {
      expect(countHallucinationMatches(liveRun.liveTranscript)).toBe(0);
    });

    it('every tick except the first runs faster than 5s', () => {
      const tail = liveRun.ticks.slice(2);
      for (const t of tail) {
        expect(
          t.inferenceMs,
          `tick ${t.index} took ${t.inferenceMs.toFixed(0)} ms on a ${t.windowDurationS.toFixed(1)} s window`,
        ).toBeLessThan(5000);
      }
    });
  });

  describe('streaming penalty (algorithm only)', () => {
    // The "streaming penalty" is the extra WER our streaming code adds
    // over a one-shot Whisper pass on the same audio. Following Macháček
    // et al. (whisper_streaming, 2023): penalty = WER(gt, live) - WER(gt, offline).
    // Positive means streaming made things worse; near-zero means our code
    // adds nothing on top of the model; negative means streaming was
    // actually MORE accurate than offline (e.g. when transformers.js
    // chunked long-form handling drops content).
    //
    // Baseline 2026-05-17: 0.00% (both offline and streaming reproduce the
    // same single tiny.en error on "for"/"to"). Threshold catches a
    // streaming-side regression of more than ~1 extra word on 22.
    it('streaming penalty stays under 5 percentage points', () => {
      const liveWer = wer(groundtruth.transcript, liveRun.liveTranscript).rate;
      const offlineWer = wer(groundtruth.transcript, offline.transcript).rate;
      expect(liveWer - offlineWer).toBeLessThan(0.05);
    });
  });
});
