# Model matrix vs synth fixture (5 s, clean English TTS)

| Model | WER | Words | Inference (ms) | Status |
|---|---|---|---|---|
| `onnx-community/whisper-tiny.en_timestamped` | 0.0% | 18 | 572 | ✅ |
| `distil-whisper/distil-large-v3.5-ONNX` | 0.0% | 18 | 9467 | ✅ |
| `onnx-community/moonshine-base-ONNX` | 11.1% | 20 | 723 | ✅ |
| `onnx-community/whisper-large-v3-turbo_timestamped` | 0.0% | 18 | 8123 | ✅ |

## Per-model transcripts

### `onnx-community/whisper-tiny.en_timestamped`

**Transcript:** the weather is nice today. I am going to the park. We will have a great time together."

### `distil-whisper/distil-large-v3.5-ONNX`

**Transcript:** The weather is nice today. I am going to the park. We will have a great time together.

### `onnx-community/moonshine-base-ONNX`

**Transcript:** The weather is nice today, I am going to the park, we will have a great time together. Thank you.

### `onnx-community/whisper-large-v3-turbo_timestamped`

**Transcript:** The weather is nice today. I am going to the park. We will have a great time together.
