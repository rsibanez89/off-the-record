// Integration test: drives `LiveTranscriptionLoop` on a ~5.5 minute synth
// clip with a real `tiny.en_timestamped` Whisper pipeline. Exercises the
// long-form streaming behaviour that the JFK (11 s) and synth (5 s)
// fixtures cannot:
//
//   - Hundreds of ticks, so the anchor must trim repeatedly to stay under
//     MAX_WINDOW_S.
//   - Sentence-end trim path fires many times.
//   - LA-2 stability over a long session (no flicker, no flapping).
//   - Memory pressure on the in-memory repos (bounded growth as chunks
//     are evicted past the anchor).
//
// Reports end-to-end quality AND streaming penalty (vs offline reference).
// On long-form clips the two diverge a lot, which is itself useful signal:
// it isolates the LocalAgreement-2 long-form duplication weakness from
// the underlying model accuracy.

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

describe('long clip (tiny.en, Node, ~5.5 min)', () => {
  let liveRun: RunResult;
  let offline: OfflineResult;
  let groundtruth: ReturnType<typeof loadGroundtruth>;

  beforeAll(async () => {
    const pipeline = await createNodeWhisperPipeline(MODEL_ID);
    const fixture = loadWav16kMono(join(FIXTURE_DIR, 'long.16k.wav'), 'long');
    groundtruth = loadGroundtruth(join(FIXTURE_DIR, 'long.json'));

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

    const histogram: Record<string, number> = {};
    for (const t of liveRun.ticks) histogram[t.kind] = (histogram[t.kind] ?? 0) + 1;

    /* eslint-disable no-console */
    console.log('\n  ===== long / tiny.en =====');
    console.log(`  duration             : ${liveRun.totalChunks} s of audio in ${liveRun.ticks.length} ticks`);
    console.log('  --- end-to-end (model + algorithm) ---');
    console.log(`  WER (gt vs live)     : ${(e2eWer.rate * 100).toFixed(2)}%  (${e2eWer.edits}/${e2eWer.referenceLength})`);
    console.log(`  CER (gt vs live)     : ${(e2eCer.rate * 100).toFixed(2)}%`);
    console.log('  --- model only (informational) ---');
    console.log(`  WER (gt vs offline)  : ${(offlineWer.rate * 100).toFixed(2)}%  (offline word count: ${offline.wordCount})`);
    console.log('  --- streaming penalty (algorithm only) ---');
    console.log(`  WER (offline vs live): ${(streamingPenalty.rate * 100).toFixed(2)}%  (${streamingPenalty.edits}/${streamingPenalty.referenceLength})`);
    console.log(`  hallucinations       : ${halluc}`);
    console.log(`  total inference      : ${liveRun.totalInferenceMs.toFixed(0)} ms  (real time ${liveRun.totalChunks * 1000} ms)`);
    console.log(`  realtime factor      : ${(liveRun.totalChunks * 1000 / liveRun.totalInferenceMs).toFixed(2)}x`);
    console.log(`  tick kinds           : ${JSON.stringify(histogram)}`);
    const norm = normalizeForScoring(liveRun.liveTranscript);
    console.log(`  live head            : "${norm.slice(0, 200)}..."`);
    console.log(`  live tail            : "...${norm.slice(-200)}"`);
    /* eslint-enable no-console */

    recordFixtureMetrics({
      fixtureId: 'long',
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
  }, 600_000);

  describe('end-to-end quality (model + streaming, vs ground truth)', () => {
    it('produces hundreds of committed words', () => {
      expect(liveRun.committedWords.length).toBeGreaterThan(500);
    });

    // Baseline 2026-05-17 with the current audio: WER 25.16%, CER 19.71%.
    // Much higher than short fixtures because of the LA-2 long-form
    // duplication issue (see streaming-penalty test below). Tighten when
    // that bug is fixed.
    it('WER stays under 35%', () => {
      expect(wer(groundtruth.transcript, liveRun.liveTranscript).rate).toBeLessThan(0.35);
    });

    it('CER stays under 30%', () => {
      expect(cer(groundtruth.transcript, liveRun.liveTranscript).rate).toBeLessThan(0.30);
    });

    it('emits zero known-hallucination substrings', () => {
      expect(countHallucinationMatches(liveRun.liveTranscript)).toBe(0);
    });

    it('the anchor trim path fires often (sentence-end driven)', () => {
      const windowsAtMax = liveRun.ticks.filter((t) => t.windowDurationS >= 23.5).length;
      expect(windowsAtMax / liveRun.ticks.length).toBeLessThan(0.1);
    });

    it('the loop keeps up with real time on average', () => {
      expect(liveRun.totalInferenceMs).toBeLessThan(liveRun.totalChunks * 1000);
    });
  });

  describe('streaming penalty (algorithm only)', () => {
    // penalty = WER(gt, live) - WER(gt, offline). See JFK test for
    // rationale.
    //
    // Baseline 2026-05-17:
    //   WER(gt vs offline) = 47.47% (transformers.js chunked long-form
    //     pass drops half the content; offline produces ~423 words out of
    //     a 771-word ground truth).
    //   WER(gt vs live)    = 25.16%.
    //   penalty            = -22.31% (streaming is BETTER than offline).
    //
    // Surprising result, worth a note: our LA-2 streaming preserves more
    // content than the one-shot offline pass on this clip. The LA-2
    // long-form duplication issue we observed before is real and
    // increases the absolute WER, but the offline path has its own (worse)
    // failure mode on this length of audio. We can detect future
    // regressions by asserting the streaming penalty does not creep above
    // a small positive value.
    it('streaming penalty stays under 5 percentage points', () => {
      const liveWer = wer(groundtruth.transcript, liveRun.liveTranscript).rate;
      const offlineWer = wer(groundtruth.transcript, offline.transcript).rate;
      expect(liveWer - offlineWer).toBeLessThan(0.05);
    });
  });
});
