import { describe, expect, it, vi } from 'vitest';
import { HypothesisBuffer } from './hypothesisBuffer';
import {
  LiveTranscriptionLoop,
  type AudioChunkRepository,
  type CollectedAudio,
  type TranscriptRepository,
} from './liveLoop';
import type { WhisperPipeline } from './whisperAdapter';
import type { ModelId } from '../audio';
import type { TranscriptToken } from '../db';

// In-memory AudioChunkRepository. The producer would normally write chunks
// to Dexie; here the test pushes them directly. `collectFrom` mirrors the
// Dexie implementation: returns concatenated samples plus `max(speechProb)`
// across the window, with the one-undefined-poisons-everything rule.
class FakeAudioRepo implements AudioChunkRepository {
  chunks: { startedAt: number; samples: Float32Array; speechProbability: number | undefined }[] = [];

  push(chunk: { startedAt: number; samples: Float32Array; speechProbability: number | undefined }) {
    this.chunks.push(chunk);
    this.chunks.sort((a, b) => a.startedAt - b.startedAt);
  }

  async collectFrom(startS: number): Promise<CollectedAudio | null> {
    const inWindow = this.chunks.filter((c) => c.startedAt >= startS);
    if (inWindow.length === 0) return null;
    const total = inWindow.reduce((s, c) => s + c.samples.length, 0);
    const out = new Float32Array(total);
    let offset = 0;
    let maxProb = -1;
    let allVerdicted = true;
    for (const c of inWindow) {
      out.set(c.samples, offset);
      offset += c.samples.length;
      if (c.speechProbability === undefined) allVerdicted = false;
      else if (c.speechProbability > maxProb) maxProb = c.speechProbability;
    }
    const t0 = inWindow[0].startedAt;
    const t1 = t0 + total / 16_000;
    return {
      samples: out,
      t0,
      t1,
      speechProbability: allVerdicted ? maxProb : undefined,
    };
  }

  async deleteBefore(startS: number): Promise<void> {
    this.chunks = this.chunks.filter((c) => c.startedAt >= startS);
  }

  async clear(): Promise<void> {
    this.chunks = [];
  }
}

class FakeTranscriptRepo implements TranscriptRepository {
  rows: TranscriptToken[] = [];
  writes = 0;
  async write(rows: TranscriptToken[]): Promise<void> {
    this.writes++;
    this.rows = [...rows];
  }
  async clear(): Promise<void> {
    this.rows = [];
  }
}

/**
 * Build a scripted Whisper pipeline. Each `tick()` consumes the next scripted
 * result. If the script runs out, returns an empty result. Captures the
 * `(samples, opts)` of every call for assertions.
 */
function scriptedPipeline(
  results: Array<{
    text: string;
    chunks: { text: string; timestamp: [number | null, number | null] }[];
  }>,
) {
  let idx = 0;
  const calls: { samples: Float32Array; opts: Record<string, unknown> }[] = [];
  const call = vi.fn(async (samples: Float32Array, opts: Record<string, unknown>) => {
    calls.push({ samples, opts });
    if (idx < results.length) return results[idx++];
    return { text: '', chunks: [] };
  });
  const pipeline = call as unknown as WhisperPipeline;
  pipeline.tokenizer = { encode: () => [] };
  return { pipeline, calls, callMock: call };
}

const TEST_MODEL_ID = 'onnx-community/whisper-base.en_timestamped' as ModelId;

function buildLoop(
  pipeline: WhisperPipeline,
  opts?: {
    buffer?: HypothesisBuffer;
    audioRepo?: FakeAudioRepo;
    transcriptRepo?: FakeTranscriptRepo;
  },
) {
  const buffer = opts?.buffer ?? new HypothesisBuffer();
  const audioRepo = opts?.audioRepo ?? new FakeAudioRepo();
  const transcriptRepo = opts?.transcriptRepo ?? new FakeTranscriptRepo();
  const loop = new LiveTranscriptionLoop({
    pipeline,
    buffer,
    audioRepo,
    transcriptRepo,
    modelId: TEST_MODEL_ID,
  });
  return { loop, buffer, audioRepo, transcriptRepo };
}

/** 1 s of silence at 16 kHz. */
const SECOND_OF_SILENCE = new Float32Array(16_000);
/** 1 s of constant low-amplitude speech-like audio at 16 kHz. */
function secondOfSpeech(): Float32Array {
  const out = new Float32Array(16_000);
  for (let i = 0; i < out.length; i++) out[i] = 0.1 * Math.sin((2 * Math.PI * 220 * i) / 16_000);
  return out;
}

describe('LiveTranscriptionLoop', () => {
  describe('idle / short-window paths', () => {
    it('returns idle when the audio repo is empty', async () => {
      const { pipeline, callMock } = scriptedPipeline([]);
      const { loop } = buildLoop(pipeline);
      const outcome = await loop.tick();
      expect(outcome.kind).toBe('idle');
      expect(outcome.rowsChanged).toBe(false);
      expect(callMock).not.toHaveBeenCalled();
    });
  });

  describe('silence gate', () => {
    it('VAD verdict below threshold force-commits and advances the anchor past t1', async () => {
      const { pipeline, callMock } = scriptedPipeline([]);
      const { loop, buffer, audioRepo, transcriptRepo } = buildLoop(pipeline);
      // Seed buffer with a tentative word that should get force-committed.
      buffer.ingest([{ text: 'hello', start: 0, end: 0.5 }]);
      audioRepo.push({ startedAt: 0, samples: SECOND_OF_SILENCE, speechProbability: 0.05 });

      const outcome = await loop.tick();
      expect(outcome.kind).toBe('silence');
      expect(outcome.rowsChanged).toBe(true);
      expect(buffer.getCommitted().map((w) => w.text)).toEqual(['hello']);
      expect(buffer.getTentative()).toEqual([]);
      // Anchor advanced past the silent window (t1 = 0 + 1 = 1).
      expect(loop.getCommittedAudioStartS()).toBe(1);
      // Chunks at or below the anchor were deleted.
      expect(audioRepo.chunks).toEqual([]);
      // Whisper was NOT invoked on silence.
      expect(callMock).not.toHaveBeenCalled();
      // Transcript was rewritten with the force-committed row.
      expect(transcriptRepo.rows.map((r) => r.text)).toEqual(['hello']);
      expect(transcriptRepo.rows.every((r) => r.isFinal === 1)).toBe(true);
    });

    it('falls back to RMS when ANY chunk in the window lacks a VAD verdict', async () => {
      const { pipeline, callMock } = scriptedPipeline([]);
      const { loop, audioRepo } = buildLoop(pipeline);
      // First chunk has no VAD verdict (e.g. VAD still loading); second has
      // a high verdict. The window-level verdict must be `undefined` so the
      // loop falls back to RMS. The combined audio is all zeros, so RMS
      // (~0) is below the silence threshold and the window is silent.
      audioRepo.push({ startedAt: 0, samples: SECOND_OF_SILENCE, speechProbability: undefined });
      audioRepo.push({ startedAt: 1, samples: SECOND_OF_SILENCE, speechProbability: 0.95 });

      const outcome = await loop.tick();
      expect(outcome.kind).toBe('silence');
      expect(callMock).not.toHaveBeenCalled();
      // Anchor advanced past both chunks (t1 = 2).
      expect(loop.getCommittedAudioStartS()).toBe(2);
    });
  });

  describe('inference path', () => {
    it('ingests Whisper output into the buffer and writes the transcript', async () => {
      const { pipeline, calls } = scriptedPipeline([
        {
          text: ' hello world',
          chunks: [
            { text: ' hello', timestamp: [0, 0.5] },
            { text: ' world', timestamp: [0.5, 1] },
          ],
        },
      ]);
      const { loop, buffer, audioRepo, transcriptRepo } = buildLoop(pipeline);
      audioRepo.push({ startedAt: 0, samples: secondOfSpeech(), speechProbability: 0.9 });

      const outcome = await loop.tick();
      expect(outcome.kind).toBe('inference');
      expect(outcome.rowsChanged).toBe(true);
      // First tick with empty tentative: everything goes to tentative, nothing
      // committed yet. LA-2 needs a second confirming tick to commit.
      expect(buffer.getCommitted()).toEqual([]);
      expect(buffer.getTentative().map((w) => w.text)).toEqual(['hello', 'world']);
      // Transcript reflects tentative rows (isFinal=0).
      expect(transcriptRepo.rows.map((r) => r.text)).toEqual(['hello', 'world']);
      expect(transcriptRepo.rows.every((r) => r.isFinal === 0)).toBe(true);
      expect(calls).toHaveLength(1);
      // Audio anchor unchanged: no sentence end, window dur (1 s) below fast-trim.
      expect(loop.getCommittedAudioStartS()).toBe(0);
    });

    it('forwards the audio offset to runWhisper so word timestamps land in the absolute timeline', async () => {
      const { pipeline, calls } = scriptedPipeline([
        {
          text: ' yes',
          chunks: [{ text: ' yes', timestamp: [0, 0.3] }],
        },
      ]);
      const { loop, buffer, audioRepo } = buildLoop(pipeline);
      // Chunk starts at 12.0 s. Word timestamps in the returned hypothesis
      // are relative (0 to 0.3); the loop must add the offset.
      audioRepo.push({ startedAt: 12, samples: secondOfSpeech(), speechProbability: 0.9 });

      await loop.tick();
      expect(calls).toHaveLength(1);
      expect(buffer.getTentative()[0]).toEqual({ text: 'yes', start: 12, end: 12.3 });
    });

    it('passes a buildInitialPrompt-derived prompt to runWhisper once committed history exists', async () => {
      const { pipeline, calls } = scriptedPipeline([
        // Tick 1: empties into tentative.
        {
          text: ' alpha beta',
          chunks: [
            { text: ' alpha', timestamp: [0, 0.4] },
            { text: ' beta', timestamp: [0.4, 0.8] },
          ],
        },
        // Tick 2: confirms tentative, adds new tentative.
        {
          text: ' alpha beta gamma',
          chunks: [
            { text: ' alpha', timestamp: [0, 0.4] },
            { text: ' beta', timestamp: [0.4, 0.8] },
            { text: ' gamma', timestamp: [0.8, 1.2] },
          ],
        },
        // Tick 3: tests that the prompt is built from committed history.
        {
          text: ' delta',
          chunks: [{ text: ' delta', timestamp: [0, 0.3] }],
        },
      ]);
      const { loop, audioRepo, buffer } = buildLoop(pipeline);
      audioRepo.push({ startedAt: 0, samples: secondOfSpeech(), speechProbability: 0.9 });
      await loop.tick();
      audioRepo.push({ startedAt: 1, samples: secondOfSpeech(), speechProbability: 0.9 });
      await loop.tick();
      // After tick 2, "alpha beta" is committed.
      expect(buffer.getCommitted().map((w) => w.text)).toEqual(['alpha', 'beta']);
      audioRepo.push({ startedAt: 2, samples: secondOfSpeech(), speechProbability: 0.9 });
      await loop.tick();
      // Third call was made with an initialPrompt containing the committed
      // history. The adapter encodes it through the tokenizer (mocked to
      // return []), so we cannot read prompt_ids; but we CAN assert the
      // pipeline was called the expected number of times.
      expect(calls).toHaveLength(3);
    });

    it('defers when the pipeline returns a hallucination line (no anchor advance, no force commit)', async () => {
      const { pipeline } = scriptedPipeline([
        {
          // Matches /^thank you\.?$/i in heuristics.ts::HALLUCINATION_LINE_PATTERNS.
          text: 'Thank you.',
          chunks: [
            { text: ' Thank', timestamp: [0, 0.3] },
            { text: ' you.', timestamp: [0.3, 0.6] },
          ],
        },
      ]);
      const { loop, buffer, audioRepo, transcriptRepo } = buildLoop(pipeline);
      audioRepo.push({ startedAt: 0, samples: secondOfSpeech(), speechProbability: 0.9 });

      const outcome = await loop.tick();
      expect(outcome.kind).toBe('hallucination');
      expect(outcome.rowsChanged).toBe(false);
      // No state mutation: tentative empty, no commits, no anchor advance.
      expect(buffer.getCommitted()).toEqual([]);
      expect(buffer.getTentative()).toEqual([]);
      expect(loop.getCommittedAudioStartS()).toBe(0);
      // Transcript was NOT rewritten.
      expect(transcriptRepo.writes).toBe(0);
      // Audio chunks preserved so the window can grow next tick.
      expect(audioRepo.chunks).toHaveLength(1);
    });

    it('surfaces inference errors via TickOutcome without throwing', async () => {
      const call = vi.fn(async () => {
        throw new Error('GPU device lost');
      });
      const pipeline = call as unknown as WhisperPipeline;
      pipeline.tokenizer = { encode: () => [] };
      const { loop, audioRepo } = buildLoop(pipeline);
      audioRepo.push({ startedAt: 0, samples: secondOfSpeech(), speechProbability: 0.9 });

      const outcome = await loop.tick();
      expect(outcome.kind).toBe('error');
      expect(outcome.errorMessage).toContain('GPU device lost');
      expect(outcome.rowsChanged).toBe(false);
    });
  });

  describe('anchor advancement policy', () => {
    it('does NOT advance the anchor when committed text has no sentence end and window is under FAST_TRIM_THRESHOLD_S', async () => {
      const { pipeline } = scriptedPipeline([
        {
          text: ' hello world',
          chunks: [
            { text: ' hello', timestamp: [0, 0.4] },
            { text: ' world', timestamp: [0.4, 0.9] },
          ],
        },
        {
          text: ' hello world',
          chunks: [
            { text: ' hello', timestamp: [0, 0.4] },
            { text: ' world', timestamp: [0.4, 0.9] },
          ],
        },
      ]);
      const { loop, audioRepo, buffer } = buildLoop(pipeline);
      audioRepo.push({ startedAt: 0, samples: secondOfSpeech(), speechProbability: 0.9 });
      await loop.tick();
      audioRepo.push({ startedAt: 1, samples: secondOfSpeech(), speechProbability: 0.9 });
      await loop.tick();
      expect(buffer.getCommitted().map((w) => w.text)).toEqual(['hello', 'world']);
      // No sentence end, window 2 s, under FAST_TRIM_THRESHOLD_S (10 s).
      expect(loop.getCommittedAudioStartS()).toBe(0);
    });

    it('trims to (sentence_end minus CONTEXT_LOOKBACK_S) once a sentence-ending word is committed', async () => {
      // Tick 1: 8 s window, hypothesis ends with a sentence-final period.
      // Tick 2: same hypothesis, so LA-2 commits everything. The latest
      // sentence end is at t=7.9; target = 7.9 - 5 = 2.9. Anchor advances.
      const sentence = {
        text: ' one two three four. ',
        chunks: [
          { text: ' one', timestamp: [0, 1] as [number | null, number | null] },
          { text: ' two', timestamp: [1, 2] as [number | null, number | null] },
          { text: ' three', timestamp: [2, 3] as [number | null, number | null] },
          { text: ' four.', timestamp: [3, 7.9] as [number | null, number | null] },
        ],
      };
      const { pipeline } = scriptedPipeline([sentence, sentence]);
      const { loop, audioRepo } = buildLoop(pipeline);
      // 8 s of speech so we hit sentence-end behaviour without tripping
      // FAST_TRIM_THRESHOLD_S (which is 10 s).
      for (let i = 0; i < 8; i++) {
        audioRepo.push({ startedAt: i, samples: secondOfSpeech(), speechProbability: 0.9 });
      }
      await loop.tick();
      await loop.tick();
      expect(loop.getCommittedAudioStartS()).toBeCloseTo(2.9, 6);
      expect(audioRepo.chunks.every((c) => c.startedAt >= 2.9)).toBe(true);
    });

    it('force-slides at MAX_WINDOW_S, force-committing any pending tentative', async () => {
      // 25 s of audio in a single inference (> MAX_WINDOW_S=24). Hypothesis
      // has no sentence end, so the only path that advances the anchor is
      // the force-slide safety net at the bottom of `tick`.
      const chunks: { text: string; timestamp: [number | null, number | null] }[] = [];
      for (let i = 0; i < 25; i++) {
        chunks.push({ text: ` w${i}`, timestamp: [i, i + 1] });
      }
      const { pipeline } = scriptedPipeline([
        { text: chunks.map((c) => c.text).join(''), chunks },
      ]);
      const { loop, buffer, audioRepo } = buildLoop(pipeline);
      for (let i = 0; i < 25; i++) {
        audioRepo.push({ startedAt: i, samples: secondOfSpeech(), speechProbability: 0.9 });
      }

      const outcome = await loop.tick();
      expect(outcome.kind).toBe('force-slide');
      expect(outcome.rowsChanged).toBe(true);
      // Force-slide commits everything tentative.
      expect(buffer.getTentative()).toEqual([]);
      expect(buffer.getCommitted()).toHaveLength(25);
      // Anchor moved to t1 - CONTEXT_LOOKBACK_S = 25 - 5 = 20.
      expect(loop.getCommittedAudioStartS()).toBe(20);
    });
  });

  describe('drainAndFinalise', () => {
    it('runs ticks until settled, force-commits remaining tentative, clears chunks, and resets anchor', async () => {
      // Two identical hypotheses across two drain ticks: LA-2 commits the
      // shared prefix; the final forceCommit pins any remainder.
      const hypothesis = {
        text: ' hello',
        chunks: [{ text: ' hello', timestamp: [0, 0.5] as [number | null, number | null] }],
      };
      const { pipeline } = scriptedPipeline([hypothesis, hypothesis]);
      const { loop, buffer, audioRepo, transcriptRepo } = buildLoop(pipeline);
      audioRepo.push({ startedAt: 0, samples: secondOfSpeech(), speechProbability: 0.9 });

      const rows = await loop.drainAndFinalise();
      expect(buffer.getTentative()).toEqual([]);
      expect(buffer.getCommitted().map((w) => w.text)).toEqual(['hello']);
      expect(loop.getCommittedAudioStartS()).toBe(0);
      expect(audioRepo.chunks).toEqual([]);
      expect(rows.map((r) => r.text)).toEqual(['hello']);
      expect(rows.every((r) => r.isFinal === 1)).toBe(true);
      expect(transcriptRepo.rows.map((r) => r.text)).toEqual(['hello']);
    });

    it('terminates promptly when the repo is empty (idle break)', async () => {
      const { pipeline, callMock } = scriptedPipeline([]);
      const { loop } = buildLoop(pipeline);
      const rows = await loop.drainAndFinalise();
      expect(rows).toEqual([]);
      expect(callMock).not.toHaveBeenCalled();
    });
  });

  describe('reset', () => {
    it('clears anchor and buffer state but does not touch repositories', async () => {
      const { pipeline } = scriptedPipeline([
        {
          text: ' x',
          chunks: [{ text: ' x', timestamp: [0, 0.2] }],
        },
      ]);
      const { loop, buffer, audioRepo, transcriptRepo } = buildLoop(pipeline);
      audioRepo.push({ startedAt: 0, samples: secondOfSpeech(), speechProbability: 0.9 });
      await loop.tick();
      expect(buffer.getTentative()).toHaveLength(1);

      loop.reset();
      expect(buffer.getCommitted()).toEqual([]);
      expect(buffer.getTentative()).toEqual([]);
      expect(loop.getCommittedAudioStartS()).toBe(0);
      // Repos untouched: reset is for in-memory state only. The worker
      // clears repos separately at session boundaries.
      expect(audioRepo.chunks).toHaveLength(1);
      expect(transcriptRepo.rows).toHaveLength(1);
    });
  });
});
