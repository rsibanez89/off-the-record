// Integration test: drives `LiveTranscriptionLoop` on the synthesized
// `synth.16k.wav` clip with a real `tiny.en_timestamped` Whisper pipeline.
// Distinct from the JFK clip in voice (female American TTS), cadence, and
// vocabulary so the harness is not biased to a single speaker.
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

describe('synth clip (tiny.en, Node)', () => {
  let liveRun: RunResult;
  let offline: OfflineResult;
  let groundtruth: ReturnType<typeof loadGroundtruth>;

  beforeAll(async () => {
    const pipeline = await createNodeWhisperPipeline(MODEL_ID);
    const fixture = loadWav16kMono(join(FIXTURE_DIR, 'synth.16k.wav'), 'synth');
    groundtruth = loadGroundtruth(join(FIXTURE_DIR, 'synth.json'));

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
    console.log('\n  ===== synth / tiny.en =====');
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
      fixtureId: 'synth',
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
    });

    // Baseline 2026-05-17: WER 0.00%, CER 0.00%. Clean TTS audio.
    it('WER stays under 5%', () => {
      expect(wer(groundtruth.transcript, liveRun.liveTranscript).rate).toBeLessThan(0.05);
    });

    it('CER stays under 2%', () => {
      expect(cer(groundtruth.transcript, liveRun.liveTranscript).rate).toBeLessThan(0.02);
    });

    it('emits zero known-hallucination substrings', () => {
      expect(countHallucinationMatches(liveRun.liveTranscript)).toBe(0);
    });
  });

  describe('streaming penalty (algorithm only)', () => {
    // penalty = WER(gt, live) - WER(gt, offline). See JFK test for
    // rationale. Baseline 2026-05-17: 0% (offline and streaming both
    // word-perfect on the clean synth audio).
    it('streaming penalty stays under 5 percentage points', () => {
      const liveWer = wer(groundtruth.transcript, liveRun.liveTranscript).rate;
      const offlineWer = wer(groundtruth.transcript, offline.transcript).rate;
      expect(liveWer - offlineWer).toBeLessThan(0.05);
    });
  });
});
