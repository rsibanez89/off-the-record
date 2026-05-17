// One-shot Whisper pass on the full audio, no chunking, no LocalAgreement-2.
// Used as the "model-only" reference transcript when computing the
// streaming penalty (WER between offline and live transcripts).
//
// Mirrors what the batch worker would produce in production on Stop.

import { runWhisper, type WhisperPipeline } from '../../../src/lib/transcription/whisperAdapter';
import { isMultilingual, type ModelId } from '../../../src/lib/audio';

export interface OfflineResult {
  transcript: string;
  durationMs: number;
  wordCount: number;
}

export async function runOfflineOnFixture(opts: {
  pipeline: WhisperPipeline;
  modelId: ModelId;
  audio: Float32Array;
}): Promise<OfflineResult> {
  const t0 = performance.now();
  const result = await runWhisper(opts.pipeline, opts.audio, {
    language: isMultilingual(opts.modelId) ? 'en' : undefined,
    offsetSeconds: 0,
  });
  const durationMs = performance.now() - t0;
  // Concatenate the word texts with single spaces. The metrics normalizer
  // strips punctuation and whitespace anyway, so we do not need to
  // reconstruct the original punctuation.
  const transcript = result.words
    .map((w) => w.text)
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim();
  return { transcript, durationMs, wordCount: result.words.length };
}
