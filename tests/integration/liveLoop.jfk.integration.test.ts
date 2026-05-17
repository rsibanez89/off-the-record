// Integration test: drives `LiveTranscriptionLoop` on the JFK clip with a
// real `tiny.en_timestamped` Whisper pipeline (via `onnxruntime-node`) and
// scores the resulting live transcript against ground truth.
//
// First run downloads ~40 MB of int8 model weights into `.cache/transformers`
// (~10 s). Subsequent runs are ~5 s of inference on this 11-second clip.
//
// This file is OPT IN. The default `npm test` only runs unit tests from
// `src/**/*.test.ts`. Run integration tests with `npm run test:integration`.

import { beforeAll, describe, expect, it } from 'vitest';
import { join } from 'node:path';
import type { ModelId } from '../../src/lib/audio';
import { createNodeWhisperPipeline } from './lib/whisperNode';
import { chunkIntoSeconds, loadGroundtruth, loadWav16kMono } from './lib/fixture';
import { runLiveOnFixture, type RunResult } from './lib/runLiveOnFixture';
import {
  cer,
  countHallucinationMatches,
  normalizeForScoring,
  wer,
} from './lib/metrics';

const MODEL_ID: ModelId = 'onnx-community/whisper-tiny.en_timestamped';
const FIXTURE_DIR = join(__dirname, '..', 'fixtures');

describe('LiveTranscriptionLoop on JFK clip (tiny.en, Node)', () => {
  let run: RunResult;
  let groundtruth: ReturnType<typeof loadGroundtruth>;

  // Load the pipeline once for the whole describe block. tiny.en weights
  // are cached after the first download; loading the in-memory pipeline
  // still takes ~10 s of session-init the first time.
  beforeAll(async () => {
    const pipeline = await createNodeWhisperPipeline(MODEL_ID);
    const fixture = loadWav16kMono(join(FIXTURE_DIR, 'jfk.16k.wav'), 'jfk');
    groundtruth = loadGroundtruth(join(FIXTURE_DIR, 'jfk.json'));
    const chunks = chunkIntoSeconds(fixture.samples, fixture.sampleRateHz);
    run = await runLiveOnFixture({
      pipeline,
      modelId: MODEL_ID,
      chunks,
    });
    // Always print the baseline so contributors can see what changed.
    const w = wer(groundtruth.transcript, run.liveTranscript);
    const c = cer(groundtruth.transcript, run.liveTranscript);
    const h = countHallucinationMatches(run.liveTranscript);
    /* eslint-disable no-console */
    console.log('\n  ===== JFK / tiny.en baseline =====');
    console.log(`  ground truth  : ${normalizeForScoring(groundtruth.transcript)}`);
    console.log(`  live          : ${normalizeForScoring(run.liveTranscript)}`);
    console.log(`  WER           : ${(w.rate * 100).toFixed(2)}%  (${w.edits} edits / ${w.referenceLength} ref words)`);
    console.log(`  CER           : ${(c.rate * 100).toFixed(2)}%  (${c.edits} edits / ${c.referenceLength} ref chars)`);
    console.log(`  hallucinations: ${h}`);
    console.log(`  ticks         : ${run.ticks.length}, total inference: ${run.totalInferenceMs.toFixed(0)} ms`);
    /* eslint-enable no-console */
  }, /* model load + inference budget */ 120_000);

  it('produces a non-empty committed transcript', () => {
    expect(run.committedWords.length).toBeGreaterThan(5);
    expect(run.liveTranscript.trim().length).toBeGreaterThan(0);
  });

  // Baseline at write time (2026-05-17, tiny.en_timestamped, fp32 enc + q4 dec):
  //   WER 4.55% (1 edit), CER 1.92% (2 edits), 0 hallucinations.
  // Thresholds below leave ~5x headroom so legitimate improvement does not
  // flake the test, while still flagging meaningful regressions (e.g., a
  // change that doubles WER would fail).
  it('WER stays under 10% (baseline 4.55% on 2026-05-17)', () => {
    const score = wer(groundtruth.transcript, run.liveTranscript);
    expect(score.rate).toBeLessThan(0.10);
  });

  it('CER stays under 5% (baseline 1.92% on 2026-05-17)', () => {
    const score = cer(groundtruth.transcript, run.liveTranscript);
    expect(score.rate).toBeLessThan(0.05);
  });

  it('emits zero known-hallucination substrings on real speech', () => {
    expect(countHallucinationMatches(run.liveTranscript)).toBe(0);
  });

  it('every tick except the first few runs faster than real-time', () => {
    // 1-second chunks; the consumer should keep up. The first tick pays the
    // ORT session warm-up; skip it. After that no tick should exceed 5x
    // wall-clock relative to its chunk (tiny.en on CPU is comfortably under).
    const tail = run.ticks.slice(2);
    for (const t of tail) {
      expect(
        t.inferenceMs,
        `tick ${t.index} took ${t.inferenceMs.toFixed(0)} ms on a ${t.windowDurationS.toFixed(1)} s window`,
      ).toBeLessThan(5000);
    }
  });
});
