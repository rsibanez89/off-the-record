// Node-side WhisperPipeline factory. Mirrors the browser
// `whisperAdapter::createPipeline` but uses the CPU backend that
// `onnxruntime-node` provides, since browser WebGPU/WASM cannot run here.
//
// Returns a callable shaped exactly like the production `WhisperPipeline`
// type, so `LiveTranscriptionLoop` and `runWhisper` consume it unchanged.

import { pipeline, env } from '@huggingface/transformers';
import type { WhisperPipeline } from '../../../src/lib/transcription/whisperAdapter';
import type { ModelId } from '../../../src/lib/audio';

// Cache model weights under the repo. CI keys this directory in actions/cache
// so the first run pays the download, every subsequent run is free.
env.cacheDir = './.cache/transformers';
env.allowLocalModels = false;

export async function createNodeWhisperPipeline(modelId: ModelId): Promise<WhisperPipeline> {
  // Encoder fp32, decoder q4: matches the browser-WASM default. We do NOT
  // use the turbo fp16 encoder branch here because the integration tests
  // run with tiny.en or base.en for speed.
  const transcriber = await pipeline('automatic-speech-recognition', modelId, {
    device: 'cpu',
    dtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' } as never,
  });
  return transcriber as unknown as WhisperPipeline;
}
