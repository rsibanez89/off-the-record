// Node-side WhisperPipeline factory. Mirrors the browser
// `whisperAdapter::createPipeline` but uses the CPU backend that
// `onnxruntime-node` provides, since browser WebGPU/WASM cannot run here.
//
// Returns a callable shaped exactly like the production `WhisperPipeline`
// type, so `LiveTranscriptionLoop` and `runWhisper` consume it unchanged.

import { pipeline, env } from '@huggingface/transformers';
import type { WhisperPipeline } from '../../../src/lib/transcription/whisperAdapter';
import { isMoonshine, type ModelId } from '../../../src/lib/audio';

// Cache model weights under the repo. CI keys this directory in actions/cache
// so the first run pays the download, every subsequent run is free.
env.cacheDir = './.cache/transformers';
env.allowLocalModels = false;

export async function createNodeWhisperPipeline(modelId: ModelId): Promise<WhisperPipeline> {
  // Dtype mirrors the production `whisperAdapter::createPipeline` for the
  // CPU/WASM path: Whisper-family checkpoints get fp32 encoder + q4 merged
  // decoder. Moonshine ships pre-quantized files so it needs q8 across the
  // board (fp32 would 404 because the matching files are not published).
  // No turbo-fp16 branch here because the Node bench cannot use WebGPU.
  const dtype: Record<string, string> = isMoonshine(modelId)
    ? { encoder_model: 'q8', decoder_model_merged: 'q8' }
    : { encoder_model: 'fp32', decoder_model_merged: 'q4' };
  const transcriber = await pipeline('automatic-speech-recognition', modelId, {
    device: 'cpu',
    dtype: dtype as never,
  });
  return transcriber as unknown as WhisperPipeline;
}
