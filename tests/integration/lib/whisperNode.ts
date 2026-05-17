// Node-side WhisperPipeline factory. Mirrors the browser
// `whisperAdapter::createPipeline` but uses the CPU backend that
// `onnxruntime-node` provides, since browser WebGPU/WASM cannot run here.
//
// Returns a callable shaped exactly like the production `WhisperPipeline`
// type, so `LiveTranscriptionLoop` and `runWhisper` consume it unchanged.

import { pipeline, env } from '@huggingface/transformers';
import type { WhisperPipeline } from '../../../src/lib/transcription/whisperAdapter';
import { isDistil, isMoonshine, type ModelId } from '../../../src/lib/audio';

// Cache model weights under the repo. CI keys this directory in actions/cache
// so the first run pays the download, every subsequent run is free.
env.cacheDir = './.cache/transformers';
env.allowLocalModels = false;

export async function createNodeWhisperPipeline(modelId: ModelId): Promise<WhisperPipeline> {
  // Dtype mirrors the production `whisperAdapter::createPipeline`:
  //   - Moonshine: q8/q8 (only file layout the repo publishes).
  //   - Distil-large-v3.5: fp16 encoder, q4 decoder. fp32 encoder is
  //     ~500 MB which the browser cannot allocate; fp16 halves it and
  //     stays in widely-supported ORT operator coverage. Matching the
  //     dtype here means the matrix measures what the user actually runs.
  //   - Everyone else: fp32 encoder, q4 decoder (no turbo-fp16 branch
  //     because Node bench cannot use WebGPU).
  let dtype: Record<string, string>;
  if (isMoonshine(modelId)) {
    dtype = { encoder_model: 'q8', decoder_model_merged: 'q8' };
  } else if (isDistil(modelId)) {
    dtype = { encoder_model: 'fp16', decoder_model_merged: 'q4' };
  } else {
    dtype = { encoder_model: 'fp32', decoder_model_merged: 'q4' };
  }
  const transcriber = await pipeline('automatic-speech-recognition', modelId, {
    device: 'cpu',
    dtype: dtype as never,
  });
  return transcriber as unknown as WhisperPipeline;
}
