// Long-session stress test: drives `LiveTranscriptionLoop` on a 12-minute
// JFK Inaugural Address with a real `tiny.en_timestamped` Whisper pipeline.
// This is the longest fixture in the suite (~720 ticks). It exists to
// surface failure modes that only appear at scale:
//
//   - Memory growth (transformers.js issue #860, WebGPU/native session
//     leaks). On Node + CPU we cannot trigger the WebGPU leak directly,
//     but cumulative session-state growth still shows up as latency drift.
//   - Inference-loop endurance: any state that grows linearly with tick
//     count (HypothesisBuffer.committed, transcript repo rows, anchor
//     chunk eviction lag) becomes visible here.
//   - Per-tick latency drift: if late ticks are markedly slower than
//     early ticks, something is leaking or accumulating.
//   - Realtime-factor stability: a session that starts at 5x realtime
//     but drops below 1x by tick 500 is a regression worth catching.
//   - Hallucination accumulation over many ticks: even a single
//     hallucination string emerging once in 720 ticks is a real product
//     concern.
//
// The end-to-end WER threshold is loose here on purpose: the long fixture
// inherits the LA-2 long-form duplication issue plus the trimmed-final-
// paragraphs (~5-10pp) of the canonical transcript that are absent from
// the audio. The streaming penalty is the cleaner regression signal.

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

describe('JFK inaugural long-session stress (tiny.en, Node, 12 min)', () => {
  let liveRun: RunResult;
  let offline: OfflineResult;
  let groundtruth: ReturnType<typeof loadGroundtruth>;

  beforeAll(async () => {
    const pipeline = await createNodeWhisperPipeline(MODEL_ID);
    const fixture = loadWav16kMono(join(FIXTURE_DIR, 'jfk-inaugural.16k.wav'), 'jfk-inaugural');
    groundtruth = loadGroundtruth(join(FIXTURE_DIR, 'jfk-inaugural.json'));

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

    const histogram: Record<string, number> = {};
    for (const t of liveRun.ticks) histogram[t.kind] = (histogram[t.kind] ?? 0) + 1;

    // Latency-drift signal: split ticks into first/last quartiles and
    // compare the mean inference time. A healthy run keeps these close.
    const inferenceTicks = liveRun.ticks.filter((t) => t.kind === 'inference');
    const quartile = Math.floor(inferenceTicks.length / 4);
    const firstQ = inferenceTicks.slice(0, quartile);
    const lastQ = inferenceTicks.slice(-quartile);
    const meanFirstQ = firstQ.reduce((s, t) => s + t.inferenceMs, 0) / firstQ.length;
    const meanLastQ = lastQ.reduce((s, t) => s + t.inferenceMs, 0) / lastQ.length;

    /* eslint-disable no-console */
    console.log('\n  ===== JFK inaugural / tiny.en (long-session stress) =====');
    console.log(`  duration              : ${liveRun.totalChunks} s of audio in ${liveRun.ticks.length} ticks`);
    console.log('  --- end-to-end (model + algorithm) ---');
    console.log(`  WER (gt vs live)      : ${(e2eWer.rate * 100).toFixed(2)}%  (${e2eWer.edits}/${e2eWer.referenceLength})`);
    console.log(`  CER (gt vs live)      : ${(e2eCer.rate * 100).toFixed(2)}%`);
    console.log('  --- model only (informational) ---');
    console.log(`  WER (gt vs offline)   : ${(offlineWer.rate * 100).toFixed(2)}%  (offline word count: ${offline.wordCount})`);
    console.log('  --- streaming penalty (algorithm only) ---');
    console.log(`  penalty (pp)          : ${((e2eWer.rate - offlineWer.rate) * 100).toFixed(2)}`);
    console.log(`  hallucinations        : ${halluc}`);
    console.log('  --- stress / stability ---');
    console.log(`  total inference       : ${liveRun.totalInferenceMs.toFixed(0)} ms (real time ${liveRun.totalChunks * 1000} ms)`);
    console.log(`  realtime factor       : ${(liveRun.totalChunks * 1000 / liveRun.totalInferenceMs).toFixed(2)}x`);
    console.log(`  tick kinds            : ${JSON.stringify(histogram)}`);
    console.log(`  mean inference Q1     : ${meanFirstQ.toFixed(1)} ms  (first ${quartile} inference ticks)`);
    console.log(`  mean inference Q4     : ${meanLastQ.toFixed(1)} ms  (last  ${quartile} inference ticks)`);
    console.log(`  latency drift Q4/Q1   : ${(meanLastQ / meanFirstQ).toFixed(2)}x`);
    const norm = normalizeForScoring(liveRun.liveTranscript);
    console.log(`  live head             : "${norm.slice(0, 200)}..."`);
    console.log(`  live tail             : "...${norm.slice(-200)}"`);
    /* eslint-enable no-console */

    recordFixtureMetrics({
      fixtureId: 'jfk-inaugural',
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
      latencyDriftQ4Q1: meanLastQ / meanFirstQ,
    });
  }, /* model load + 12 min audio + drain budget */ 900_000);

  describe('end-to-end quality (informational, vs ground truth)', () => {
    // The full transcript runs ~14 min; our audio is the first 12 min,
    // so roughly the last 5-10% of the transcript is missing from the
    // audio. End-to-end WER inherits that as a floor, plus the LA-2
    // long-form duplication issue. Loose threshold; baseline filled in
    // on first run.
    it('produces a non-empty committed transcript with many words', () => {
      expect(liveRun.committedWords.length).toBeGreaterThan(1000);
    });

    it('WER stays under 50%', () => {
      expect(wer(groundtruth.transcript, liveRun.liveTranscript).rate).toBeLessThan(0.50);
    });

    it('emits zero known-hallucination substrings (no creep over hundreds of ticks)', () => {
      expect(countHallucinationMatches(liveRun.liveTranscript)).toBe(0);
    });
  });

  describe('streaming penalty (algorithm only, vs offline)', () => {
    // The same trimmed-transcript issue affects offline too, so the
    // delta between offline and live is a clean signal. Threshold leaves
    // room for the known LA-2 long-form behaviours.
    it('streaming penalty stays under 25 percentage points', () => {
      const liveWer = wer(groundtruth.transcript, liveRun.liveTranscript).rate;
      const offlineWer = wer(groundtruth.transcript, offline.transcript).rate;
      expect(liveWer - offlineWer).toBeLessThan(0.25);
    });
  });

  describe('stress / stability (this fixture\'s primary purpose)', () => {
    it('completes all 720 ticks without throwing', () => {
      expect(liveRun.ticks.length).toBeGreaterThanOrEqual(720);
    });

    it('no tick reports an error kind', () => {
      const errors = liveRun.ticks.filter((t) => t.kind === 'error');
      expect(errors, `errors at ticks: ${errors.map((t) => t.index).join(', ')}`).toEqual([]);
    });

    it('keeps up with real time on average (realtime factor > 1)', () => {
      expect(liveRun.totalInferenceMs).toBeLessThan(liveRun.totalChunks * 1000);
    });

    it('per-tick latency does not drift drastically across the session', () => {
      // If late ticks are >3x slower than early ticks, something is
      // leaking or accumulating per-tick state. This is the canary for
      // transformers.js issue #860 (WebGPU) or local state growth.
      const inf = liveRun.ticks.filter((t) => t.kind === 'inference');
      const quartile = Math.floor(inf.length / 4);
      const meanFirstQ = inf.slice(0, quartile).reduce((s, t) => s + t.inferenceMs, 0) / quartile;
      const meanLastQ = inf.slice(-quartile).reduce((s, t) => s + t.inferenceMs, 0) / quartile;
      const drift = meanLastQ / meanFirstQ;
      expect(
        drift,
        `Q4 (${meanLastQ.toFixed(0)} ms) vs Q1 (${meanFirstQ.toFixed(0)} ms) = ${drift.toFixed(2)}x`,
      ).toBeLessThan(3.0);
    });

    it('the safety-net force-slide fires at most a few times', () => {
      // Force-slide at MAX_WINDOW_S is the LAST-RESORT trim. On a clean
      // 12-min monologue the sentence-end and fast-trim paths should
      // handle anchor advancement. A high count here means the trim
      // logic is failing on long-form audio.
      const forceSlide = liveRun.ticks.filter((t) => t.kind === 'force-slide').length;
      expect(forceSlide).toBeLessThan(20);
    });
  });
});
