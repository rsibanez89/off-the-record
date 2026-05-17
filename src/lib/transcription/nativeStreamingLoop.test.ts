import { describe, expect, it, vi } from 'vitest';
import { NativeStreamingLoop } from './nativeStreamingLoop';
import type { ModelId } from '../audio';
import type { TranscriptToken } from '../db';
import type {
  AudioChunkRepository,
  CollectedAudio,
  TranscriptRepository,
} from './liveLoop';
import type { WhisperPipeline } from './whisperAdapter';

// `moonshine-base-ONNX` is the only `streamingPolicy: 'native'` model in
// MODELS today. Using its id keeps the loop's policy-mismatch warning
// silent so the test output stays clean.
const MOONSHINE: ModelId = 'onnx-community/moonshine-base-ONNX';

class FakeAudioRepo implements AudioChunkRepository {
  chunks: { startedAt: number; samples: Float32Array; speechProbability: number | undefined }[] = [];
  push(c: { startedAt: number; samples: Float32Array; speechProbability: number | undefined }) {
    this.chunks.push(c);
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
    return { samples: out, t0, t1, speechProbability: allVerdicted ? maxProb : undefined };
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
  async write(rows: TranscriptToken[]): Promise<void> {
    this.rows = [...rows];
  }
  async writeIncremental(diff: TranscriptToken[], totalCount: number): Promise<void> {
    for (const row of diff) this.rows[row.tokenId] = row;
    if (this.rows.length > totalCount) this.rows.length = totalCount;
  }
  async clear(): Promise<void> {
    this.rows = [];
  }
}

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
  // Tokenizer that pretends to encode whatever text it gets. Lets the
  // adapter's prompt-ids path run end to end in tests.
  pipeline.tokenizer = {
    encode: (text: string) => text.split(/\s+/).filter((s) => s.length > 0).map((_, i) => i + 1),
  };
  return { pipeline, calls };
}

const SECOND_OF_SILENCE = new Float32Array(16_000);
function secondOfSpeech(): Float32Array {
  const out = new Float32Array(16_000);
  for (let i = 0; i < out.length; i++) {
    out[i] = 0.05 * Math.sin((2 * Math.PI * 220 * i) / 16_000);
  }
  return out;
}

describe('NativeStreamingLoop', () => {
  it('returns short-window until enough audio is collected (NATIVE_STREAMING_WINDOW_S)', async () => {
    const audioRepo = new FakeAudioRepo();
    const transcriptRepo = new FakeTranscriptRepo();
    audioRepo.push({ startedAt: 0, samples: secondOfSpeech(), speechProbability: 0.9 });
    const { pipeline } = scriptedPipeline([
      { text: ' nope', chunks: [{ text: ' nope', timestamp: [0, 0.5] }] },
    ]);
    const loop = new NativeStreamingLoop({ pipeline, audioRepo, transcriptRepo, modelId: MOONSHINE });

    const outcome = await loop.tick();
    expect(outcome.kind).toBe('short-window');
    expect(outcome.rowsChanged).toBe(false);
    expect(loop.getCommittedAudioStartS()).toBe(0);
  });

  it('commits the pipeline output immediately and advances the anchor past the window', async () => {
    const audioRepo = new FakeAudioRepo();
    const transcriptRepo = new FakeTranscriptRepo();
    // Two 1 s chunks to clear the 2 s native-streaming minimum.
    audioRepo.push({ startedAt: 0, samples: secondOfSpeech(), speechProbability: 0.9 });
    audioRepo.push({ startedAt: 1, samples: secondOfSpeech(), speechProbability: 0.9 });
    const { pipeline } = scriptedPipeline([
      {
        text: ' hello world',
        chunks: [
          { text: ' hello', timestamp: [0, 0.4] },
          { text: ' world', timestamp: [0.4, 1.0] },
        ],
      },
    ]);
    const loop = new NativeStreamingLoop({ pipeline, audioRepo, transcriptRepo, modelId: MOONSHINE });

    const outcome = await loop.tick();
    expect(outcome.kind).toBe('inference');
    expect(outcome.rowsChanged).toBe(true);
    expect(outcome.rows.map((r) => r.text)).toEqual(['hello', 'world']);
    expect(outcome.rows.every((r) => r.isFinal === 1)).toBe(true);
    expect(loop.getCommittedWordCount()).toBe(2);
    expect(loop.getTentativeWordCount()).toBe(0);
    expect(loop.getCommittedAudioStartS()).toBe(2.0);
    expect(audioRepo.chunks).toHaveLength(0);
  });

  it('skips Whisper on a silent VAD verdict and still advances the anchor', async () => {
    const audioRepo = new FakeAudioRepo();
    const transcriptRepo = new FakeTranscriptRepo();
    audioRepo.push({ startedAt: 0, samples: SECOND_OF_SILENCE, speechProbability: 0.0 });
    audioRepo.push({ startedAt: 1, samples: SECOND_OF_SILENCE, speechProbability: 0.0 });
    const { pipeline } = scriptedPipeline([]);
    const loop = new NativeStreamingLoop({ pipeline, audioRepo, transcriptRepo, modelId: MOONSHINE });

    const outcome = await loop.tick();
    expect(outcome.kind).toBe('silence');
    expect(loop.getCommittedAudioStartS()).toBe(2.0);
    expect(transcriptRepo.rows).toEqual([]);
  });

  it('accumulates output across multiple ticks without overlap', async () => {
    const audioRepo = new FakeAudioRepo();
    const transcriptRepo = new FakeTranscriptRepo();
    audioRepo.push({ startedAt: 0, samples: secondOfSpeech(), speechProbability: 0.9 });
    audioRepo.push({ startedAt: 1, samples: secondOfSpeech(), speechProbability: 0.9 });
    const { pipeline } = scriptedPipeline([
      { text: ' alpha', chunks: [{ text: ' alpha', timestamp: [0, 0.5] }] },
      { text: ' beta', chunks: [{ text: ' beta', timestamp: [0, 0.5] }] },
    ]);
    const loop = new NativeStreamingLoop({ pipeline, audioRepo, transcriptRepo, modelId: MOONSHINE });

    // First tick consumes both chunks (2 s window meets the minimum).
    await loop.tick();
    expect(loop.getCommittedWordCount()).toBe(1);
    expect(loop.getCommittedAudioStartS()).toBe(2.0);

    // Need 2 more 1 s chunks to clear the minimum on the second window.
    audioRepo.push({ startedAt: 2, samples: secondOfSpeech(), speechProbability: 0.9 });
    audioRepo.push({ startedAt: 3, samples: secondOfSpeech(), speechProbability: 0.9 });
    await loop.tick();
    expect(loop.getCommittedWordCount()).toBe(2);
    expect(transcriptRepo.rows.map((r) => r.text)).toEqual(['alpha', 'beta']);
  });

  it('drops a hallucinated line and advances the anchor', async () => {
    const audioRepo = new FakeAudioRepo();
    const transcriptRepo = new FakeTranscriptRepo();
    audioRepo.push({ startedAt: 0, samples: secondOfSpeech(), speechProbability: 0.9 });
    audioRepo.push({ startedAt: 1, samples: secondOfSpeech(), speechProbability: 0.9 });
    const { pipeline } = scriptedPipeline([
      {
        text: ' Thanks for watching.',
        chunks: [{ text: ' Thanks for watching.', timestamp: [0, 1.0] }],
      },
    ]);
    const loop = new NativeStreamingLoop({ pipeline, audioRepo, transcriptRepo, modelId: MOONSHINE });

    const outcome = await loop.tick();
    expect(outcome.kind).toBe('hallucination');
    expect(loop.getCommittedWordCount()).toBe(0);
    expect(loop.getCommittedAudioStartS()).toBe(2.0);
  });

  it('drainAndFinalise consumes remaining audio (including sub-min-window) and clears the repo', async () => {
    const audioRepo = new FakeAudioRepo();
    const transcriptRepo = new FakeTranscriptRepo();
    // Only 1 s of audio: below the per-tick minimum, but the drain
    // overrides minWindowS=0 so this still gets transcribed on Stop.
    audioRepo.push({ startedAt: 0, samples: secondOfSpeech(), speechProbability: 0.9 });
    const { pipeline } = scriptedPipeline([
      { text: ' done', chunks: [{ text: ' done', timestamp: [0, 0.5] }] },
    ]);
    const loop = new NativeStreamingLoop({ pipeline, audioRepo, transcriptRepo, modelId: MOONSHINE });

    const rows = await loop.drainAndFinalise();
    expect(rows.map((r) => r.text)).toEqual(['done']);
    expect(audioRepo.chunks).toEqual([]);
    expect(loop.getCommittedAudioStartS()).toBe(0);
  });

  it('passes the committed tail as initialPrompt on subsequent ticks', async () => {
    const audioRepo = new FakeAudioRepo();
    const transcriptRepo = new FakeTranscriptRepo();
    audioRepo.push({ startedAt: 0, samples: secondOfSpeech(), speechProbability: 0.9 });
    audioRepo.push({ startedAt: 1, samples: secondOfSpeech(), speechProbability: 0.9 });
    const { pipeline, calls } = scriptedPipeline([
      {
        text: ' alpha beta',
        chunks: [
          { text: ' alpha', timestamp: [0, 0.4] },
          { text: ' beta', timestamp: [0.4, 1.0] },
        ],
      },
      { text: ' gamma', chunks: [{ text: ' gamma', timestamp: [0, 0.5] }] },
    ]);
    const loop = new NativeStreamingLoop({ pipeline, audioRepo, transcriptRepo, modelId: MOONSHINE });

    // First tick: no committed history, no prompt.
    await loop.tick();
    expect(calls[0].opts.prompt_ids).toBeUndefined();
    expect(loop.getCommittedWordCount()).toBe(2);

    // Second tick: committed tail "alpha beta" should be passed as prompt.
    audioRepo.push({ startedAt: 2, samples: secondOfSpeech(), speechProbability: 0.9 });
    audioRepo.push({ startedAt: 3, samples: secondOfSpeech(), speechProbability: 0.9 });
    await loop.tick();
    expect(calls[1].opts.prompt_ids).toBeDefined();
    expect(Array.isArray(calls[1].opts.prompt_ids)).toBe(true);
    expect((calls[1].opts.prompt_ids as number[]).length).toBeGreaterThan(0);
  });

  it('reset clears committed words and anchor', async () => {
    const audioRepo = new FakeAudioRepo();
    const transcriptRepo = new FakeTranscriptRepo();
    audioRepo.push({ startedAt: 0, samples: secondOfSpeech(), speechProbability: 0.9 });
    audioRepo.push({ startedAt: 1, samples: secondOfSpeech(), speechProbability: 0.9 });
    const { pipeline } = scriptedPipeline([
      { text: ' x', chunks: [{ text: ' x', timestamp: [0, 0.2] }] },
    ]);
    const loop = new NativeStreamingLoop({ pipeline, audioRepo, transcriptRepo, modelId: MOONSHINE });
    await loop.tick();
    expect(loop.getCommittedWordCount()).toBe(1);

    loop.reset();
    expect(loop.getCommittedWordCount()).toBe(0);
    expect(loop.getCommittedAudioStartS()).toBe(0);
  });
});
