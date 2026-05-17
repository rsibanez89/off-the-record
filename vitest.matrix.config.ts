import { defineConfig } from 'vitest/config';

// Model verification matrix. Runs all picker models against the smallest
// integration fixture so you can confirm each one actually transcribes
// audio when importing a new entry into `MODELS`. This is NOT part of the
// regression-gated loop (`npm run bench`). Run it manually with
// `npm run bench:matrix` after touching the model list or dtype config.
export default defineConfig({
  test: {
    include: ['tests/matrix/**/*.matrix.test.ts'],
    environment: 'node',
    // Each model loads its own weights (potentially hundreds of MB on a
    // cold cache). First run is long; subsequent runs use the cached
    // weights in `.cache/transformers/`.
    testTimeout: 600_000,
    hookTimeout: 600_000,
    // Run files serially so two tests do not load the same weights twice.
    fileParallelism: false,
  },
});
