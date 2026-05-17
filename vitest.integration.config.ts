import { defineConfig } from 'vitest/config';

// Integration tests: real Whisper inference via onnxruntime-node, real
// audio fixtures. Slow (first run downloads model weights, ~10 s to load,
// ~5 to 60 s to infer). Kept out of `npm test` on purpose; run with
// `npm run test:integration`.
export default defineConfig({
  test: {
    include: ['tests/integration/**/*.integration.test.ts'],
    environment: 'node',
    // Long-running model load + inference. Tests inside set their own
    // beforeAll budgets too.
    testTimeout: 120_000,
    hookTimeout: 120_000,
    // Run files serially so two tests do not load the same model twice
    // into memory and contend for CPU.
    fileParallelism: false,
  },
});
