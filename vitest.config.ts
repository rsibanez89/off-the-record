import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
    // Test files are pure modules (no Vite-resolved virtual imports, no
    // workers). Node env is enough and avoids the cost of jsdom.
  },
});
