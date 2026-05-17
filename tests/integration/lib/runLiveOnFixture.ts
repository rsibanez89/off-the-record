// Drive `LiveTranscriptionLoop` on a pre-chunked audio fixture as fast as
// the Whisper pipeline can infer. Mirrors production cadence: one chunk
// written by the producer triggers (at most) one tick on the consumer.

import { HypothesisBuffer } from '../../../src/lib/transcription/hypothesisBuffer';
import {
  LiveTranscriptionLoop,
  type TickKind,
} from '../../../src/lib/transcription/liveLoop';
import type { WhisperPipeline } from '../../../src/lib/transcription/whisperAdapter';
import type { ModelId } from '../../../src/lib/audio';
import {
  InMemoryAudioChunkRepository,
  InMemoryTranscriptRepository,
} from './inMemoryRepos';

export interface TickRecord {
  /** Tick index, 0-based. */
  index: number;
  /** Producer-clock time when this chunk became available (seconds since fixture start). */
  chunkAvailableAtS: number;
  kind: TickKind;
  windowDurationS: number;
  inferenceMs: number;
  /** Committed word count AFTER the tick. */
  committedCount: number;
  /** Tentative word count AFTER the tick. */
  tentativeCount: number;
}

export interface RunResult {
  liveTranscript: string;
  ticks: TickRecord[];
  /** Final committed words with their absolute-timeline start/end. */
  committedWords: { text: string; start: number; end: number }[];
  totalChunks: number;
  totalInferenceMs: number;
}

/**
 * Feed `chunks` into a fresh `LiveTranscriptionLoop` one by one. After every
 * push we run a single tick. At the end we drain.
 *
 * The `speechProbability` for each chunk is set to 0.9 by default so the
 * VAD silence gate does not short-circuit on real speech. Tests that want
 * to exercise the silence gate or the RMS fallback can pass their own
 * `probabilityFor` function.
 */
export async function runLiveOnFixture(opts: {
  pipeline: WhisperPipeline;
  modelId: ModelId;
  chunks: Float32Array[];
  /**
   * Returns the VAD probability to attach to chunk `i`. Default: 0.9 so the
   * VAD silence gate never fires.
   */
  probabilityFor?: (i: number, chunk: Float32Array) => number | undefined;
}): Promise<RunResult> {
  const probabilityFor = opts.probabilityFor ?? (() => 0.9);
  const audioRepo = new InMemoryAudioChunkRepository();
  const transcriptRepo = new InMemoryTranscriptRepository();
  const buffer = new HypothesisBuffer();
  const loop = new LiveTranscriptionLoop({
    pipeline: opts.pipeline,
    buffer,
    audioRepo,
    transcriptRepo,
    modelId: opts.modelId,
  });

  const ticks: TickRecord[] = [];
  let totalInferenceMs = 0;

  for (let i = 0; i < opts.chunks.length; i++) {
    const startedAt = i; // chunks are 1 s each, so startedAt in seconds = i.
    audioRepo.push({
      startedAt,
      samples: opts.chunks[i],
      speechProbability: probabilityFor(i, opts.chunks[i]),
    });
    const t0 = performance.now();
    const outcome = await loop.tick();
    const inferenceMs = performance.now() - t0;
    totalInferenceMs += inferenceMs;
    ticks.push({
      index: i,
      chunkAvailableAtS: startedAt + 1, // chunk available AFTER its full second.
      kind: outcome.kind,
      windowDurationS: outcome.windowDurationS,
      inferenceMs,
      committedCount: loop.getCommittedWordCount(),
      tentativeCount: loop.getTentativeWordCount(),
    });
  }

  // End-of-clip drain: force-commits any remaining tentative.
  const t0 = performance.now();
  await loop.drainAndFinalise();
  totalInferenceMs += performance.now() - t0;

  const committedWords = buffer.getCommitted().map((w) => ({
    text: w.text,
    start: w.start,
    end: w.end,
  }));
  const liveTranscript = transcriptRepo.rows
    .map((r) => r.text)
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim();

  return {
    liveTranscript,
    ticks,
    committedWords,
    totalChunks: opts.chunks.length,
    totalInferenceMs,
  };
}
