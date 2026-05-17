// Hallucination torture tests. tiny.en is known to emit "Thanks for
// watching", "(phone beeping)", "[BLANK_AUDIO]", or " >>" on non-speech
// audio. Our defences are layered:
//
//   1. Silero VAD per chunk: if `speechProbability < VAD_SILENCE_THRESHOLD`
//      the consumer never calls Whisper for that window.
//   2. RMS fallback: when VAD has no verdict, an all-quiet window is also
//      treated as silent.
//   3. `heuristics::isHallucinationLine` filter: if Whisper IS called and
//      its output matches a known hallucination pattern, the loop defers
//      (no commit, no anchor advance) so the next inference has more
//      context.
//
// Each test below exercises one of these layers.

import { beforeAll, describe, expect, it } from 'vitest';
import type { ModelId } from '../../src/lib/audio';
import { createNodeWhisperPipeline } from './lib/whisperNode';
import { runLiveOnFixture, type RunResult } from './lib/runLiveOnFixture';
import { countHallucinationMatches, normalizeForScoring } from './lib/metrics';

const MODEL_ID: ModelId = 'onnx-community/whisper-tiny.en_timestamped';
const SR = 16_000;

function silenceChunk(): Float32Array {
  return new Float32Array(SR);
}

// Low-amplitude 440 Hz tone: loud enough to pass an RMS-only silence check,
// but exactly the kind of audio tiny.en is known to hallucinate on (the
// project's earlier smoke test showed "(phone beeping)" on this input).
function toneChunk(): Float32Array {
  const out = new Float32Array(SR);
  for (let i = 0; i < SR; i++) out[i] = 0.1 * Math.sin((2 * Math.PI * 440 * i) / SR);
  return out;
}

// Build a callable-shaped pipeline that counts how many times the loop
// actually invoked Whisper. Lets us assert that the silence gate
// short-circuited inference entirely.
function instrumentedPipeline(realPipeline: Awaited<ReturnType<typeof createNodeWhisperPipeline>>) {
  let calls = 0;
  const wrapper = (async (samples: Float32Array, opts: Record<string, unknown>) => {
    calls++;
    return (realPipeline as unknown as (s: Float32Array, o: Record<string, unknown>) => Promise<unknown>)(
      samples,
      opts,
    );
  }) as unknown as typeof realPipeline;
  // Preserve the tokenizer so `runWhisper` can encode prompts.
  wrapper.tokenizer = realPipeline.tokenizer;
  return { pipeline: wrapper, getCalls: () => calls };
}

describe('Hallucination defences', () => {
  // Load the real pipeline once. The silence tests do not actually need
  // inference (the gate should fire first), but having a real pipeline
  // ready means the wrapper falls back to it correctly if the gate misses.
  let basePipeline: Awaited<ReturnType<typeof createNodeWhisperPipeline>>;
  beforeAll(async () => {
    basePipeline = await createNodeWhisperPipeline(MODEL_ID);
  }, 120_000);

  describe('VAD silence gate (probability below threshold)', () => {
    let run: RunResult;
    let whisperCalls = 0;

    beforeAll(async () => {
      const { pipeline, getCalls } = instrumentedPipeline(basePipeline);
      const chunks = Array.from({ length: 5 }, () => silenceChunk());
      run = await runLiveOnFixture({
        pipeline,
        modelId: MODEL_ID,
        chunks,
        // VAD says silence (well below VAD_SILENCE_THRESHOLD=0.3).
        probabilityFor: () => 0.01,
      });
      whisperCalls = getCalls();
    }, 120_000);

    it('never invokes Whisper when every chunk is VAD-classified silent', () => {
      expect(whisperCalls).toBe(0);
    });

    it('produces an empty committed transcript', () => {
      expect(run.committedWords).toEqual([]);
      expect(run.liveTranscript.trim()).toBe('');
    });

    it('reports every tick as kind = silence (or idle for the empty repo case)', () => {
      for (const t of run.ticks) {
        expect(['silence', 'idle']).toContain(t.kind);
      }
    });
  });

  describe('RMS fallback when VAD verdict is undefined', () => {
    let run: RunResult;
    let whisperCalls = 0;

    beforeAll(async () => {
      const { pipeline, getCalls } = instrumentedPipeline(basePipeline);
      const chunks = Array.from({ length: 5 }, () => silenceChunk());
      run = await runLiveOnFixture({
        pipeline,
        modelId: MODEL_ID,
        chunks,
        // Simulate VAD not yet loaded: no probability set. The loop must
        // fall back to RMS for the silence check.
        probabilityFor: () => undefined,
      });
      whisperCalls = getCalls();
    }, 120_000);

    it('never invokes Whisper when the RMS fallback says silent', () => {
      expect(whisperCalls).toBe(0);
    });

    it('produces an empty committed transcript', () => {
      expect(run.liveTranscript.trim()).toBe('');
    });
  });

  describe('isHallucinationLine filter on non-speech audio that passes the silence gate', () => {
    let run: RunResult;

    beforeAll(async () => {
      // A 440 Hz tone has RMS well above SILENCE_RMS_THRESHOLD, AND we tell
      // the loop VAD is uncertain (undefined) so it falls back to RMS, which
      // says "not silent". This forces Whisper to run on non-speech audio.
      // The defence here is the line-level hallucination filter: tiny.en
      // returns things like "(phone beeping)" or " >>" on tones, all of
      // which are caught by `isHallucinationLine` and deferred.
      const chunks = Array.from({ length: 5 }, () => toneChunk());
      run = await runLiveOnFixture({
        pipeline: basePipeline,
        modelId: MODEL_ID,
        chunks,
        probabilityFor: () => undefined, // RMS path; RMS on tone is non-silent.
      });
      /* eslint-disable no-console */
      console.log('\n  ===== tone / tiny.en hallucination defence =====');
      console.log(`  committed text   : "${normalizeForScoring(run.liveTranscript)}"`);
      console.log(`  hallucination hits: ${countHallucinationMatches(run.liveTranscript)}`);
      console.log(`  ticks            : ${run.ticks.map((t) => t.kind).join(', ')}`);
      /* eslint-enable no-console */
    }, 120_000);

    it('does not commit any known-hallucination substring to the live transcript', () => {
      // Even if Whisper hallucinates internally, the loop must defer and
      // never commit those words. If this asserts > 0, the
      // `isHallucinationLine` filter has regressed.
      expect(countHallucinationMatches(run.liveTranscript)).toBe(0);
    });

    it('produces an empty or extremely short committed transcript on pure tone', () => {
      // On a 5-second tone the model has no real speech to commit; the only
      // path that could populate the buffer is a non-hallucination filter
      // miss. Allow up to 2 committed words as a safety margin for model
      // jitter, but flag anything more.
      expect(run.committedWords.length).toBeLessThanOrEqual(2);
    });
  });
});
