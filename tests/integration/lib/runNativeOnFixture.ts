// Drive `NativeStreamingLoop` on a pre-chunked audio fixture. Mirrors
// `runLiveOnFixture` (LA-2) but uses the native streaming policy:
// fixed 2 s windows, no overlap, no tentative state, output directly
// final. Used by the model matrix to compare end-to-end native
// streaming quality vs the one-shot `runWhisper` baseline.

import { NativeStreamingLoop } from '../../../src/lib/transcription/nativeStreamingLoop';
import type { TickKind } from '../../../src/lib/transcription/liveLoop';
import type { WhisperPipeline } from '../../../src/lib/transcription/whisperAdapter';
import type { ModelId } from '../../../src/lib/audio';
import {
  InMemoryAudioChunkRepository,
  InMemoryTranscriptRepository,
} from './inMemoryRepos';

export interface NativeTickRecord {
  index: number;
  kind: TickKind;
  windowDurationS: number;
  inferenceMs: number;
  committedCount: number;
}

export interface NativeRunResult {
  liveTranscript: string;
  ticks: NativeTickRecord[];
  totalChunks: number;
  totalInferenceMs: number;
}

/**
 * Feed `chunks` into a fresh `NativeStreamingLoop` one by one. After
 * every push we run a single tick (which may return `short-window` if
 * fewer than `NATIVE_STREAMING_WINDOW_S` seconds have accumulated). At
 * the end we drain so trailing sub-window audio still gets transcribed.
 *
 * `speechProbability` defaults to 0.9 so the VAD silence gate does not
 * short-circuit on real speech in fixture tests.
 */
export async function runNativeOnFixture(opts: {
  pipeline: WhisperPipeline;
  modelId: ModelId;
  chunks: Float32Array[];
  probabilityFor?: (i: number, chunk: Float32Array) => number | undefined;
}): Promise<NativeRunResult> {
  const probabilityFor = opts.probabilityFor ?? (() => 0.9);
  const audioRepo = new InMemoryAudioChunkRepository();
  const transcriptRepo = new InMemoryTranscriptRepository();
  const loop = new NativeStreamingLoop({
    pipeline: opts.pipeline,
    audioRepo,
    transcriptRepo,
    modelId: opts.modelId,
  });

  const ticks: NativeTickRecord[] = [];
  let totalInferenceMs = 0;

  for (let i = 0; i < opts.chunks.length; i++) {
    audioRepo.push({
      startedAt: i,
      samples: opts.chunks[i],
      speechProbability: probabilityFor(i, opts.chunks[i]),
    });
    const t0 = performance.now();
    const outcome = await loop.tick();
    const inferenceMs = performance.now() - t0;
    totalInferenceMs += inferenceMs;
    ticks.push({
      index: i,
      kind: outcome.kind,
      windowDurationS: outcome.windowDurationS,
      inferenceMs,
      committedCount: loop.getCommittedWordCount(),
    });
  }

  const t0 = performance.now();
  await loop.drainAndFinalise();
  totalInferenceMs += performance.now() - t0;

  const liveTranscript = transcriptRepo.rows
    .map((r) => r.text)
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim();

  return {
    liveTranscript,
    ticks,
    totalChunks: opts.chunks.length,
    totalInferenceMs,
  };
}
