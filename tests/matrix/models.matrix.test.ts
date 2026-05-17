// Model verification matrix. Runs every model in `MODELS` against the
// `synth` fixture (5 s clean English TTS) via the production
// `runWhisper` adapter, and prints a per-model report covering WER vs
// ground truth, raw transcript, and inference time. This is used when
// importing a new model to confirm it actually transcribes audio
// before exposing it in the picker; it is NOT in the autonomous loop's
// regression bench because adding models should not gate the loop and
// pulling 4 model weights every loop iteration is wasteful.
//
// Run with: `npm run bench:matrix`.
//
// Failure semantics: each model gets a soft assertion that it produced a
// non-empty transcript. Hard quality assertions live in the integration
// suite (tiny.en only); this file is a diagnostic, not a regression gate.

import { beforeAll, describe, expect, it } from 'vitest';
import { join } from 'node:path';
import { MODELS, isMultilingual, supportsWordTimestamps, type ModelId } from '../../src/lib/audio';
import { runWhisper } from '../../src/lib/transcription/whisperAdapter';
import { mkdirSync, writeFileSync } from 'node:fs';
import { createNodeWhisperPipeline } from '../integration/lib/whisperNode';
import { loadGroundtruth, loadWav16kMono } from '../integration/lib/fixture';
import { wer, normalizeForScoring } from '../integration/lib/metrics';

const FIXTURE_DIR = join(__dirname, '..', 'fixtures');

interface MatrixRow {
  modelId: ModelId;
  durationMs: number;
  wordCount: number;
  werRate: number;
  transcript: string;
  error?: string;
}

const rows: MatrixRow[] = [];

describe('Model matrix vs synth fixture (5 s, clean English TTS)', () => {
  let fixture: ReturnType<typeof loadWav16kMono>;
  let groundtruth: ReturnType<typeof loadGroundtruth>;

  beforeAll(() => {
    fixture = loadWav16kMono(join(FIXTURE_DIR, 'synth.16k.wav'), 'synth');
    groundtruth = loadGroundtruth(join(FIXTURE_DIR, 'synth.json'));
  });

  for (const m of MODELS) {
    it(`${m.id} transcribes synth`, async () => {
      const t0 = performance.now();
      try {
        const pipeline = await createNodeWhisperPipeline(m.id);
        const result = await runWhisper(pipeline, fixture.samples, {
          language: isMultilingual(m.id) ? 'en' : undefined,
          offsetSeconds: 0,
          requestWordTimestamps: supportsWordTimestamps(m.id),
        });
        const durationMs = performance.now() - t0;
        const transcript = result.words
          .map((w) => w.text)
          .join(' ')
          .replace(/\s+/g, ' ')
          .trim();
        const werResult = wer(groundtruth.transcript, transcript);
        rows.push({
          modelId: m.id,
          durationMs,
          wordCount: result.words.length,
          werRate: werResult.rate,
          transcript,
        });
        /* eslint-disable no-console */
        console.log(`\n  ===== ${m.id} =====`);
        console.log(`  ground truth : ${normalizeForScoring(groundtruth.transcript)}`);
        console.log(`  transcript   : ${normalizeForScoring(transcript)}`);
        console.log(`  WER          : ${(werResult.rate * 100).toFixed(2)}%  (${werResult.edits}/${werResult.referenceLength})`);
        console.log(`  words        : ${result.words.length}`);
        console.log(`  inference    : ${durationMs.toFixed(0)} ms`);
        /* eslint-enable no-console */
        // Soft sanity: produced SOMETHING. A model that returns zero words
        // on clean English TTS is broken. Quality thresholds live in the
        // integration suite, not here.
        expect(transcript.length).toBeGreaterThan(0);
      } catch (err) {
        rows.push({
          modelId: m.id,
          durationMs: performance.now() - t0,
          wordCount: 0,
          werRate: 1,
          transcript: '',
          error: (err as Error).message,
        });
        throw err;
      }
    }, 300_000);
  }

  it('writes aggregated matrix report', () => {
    // Vitest's default reporter swallows console.log on passing tests, so
    // we persist the matrix report to disk where the user (and a future
    // diff) can read it. The file is `.gitignore`d by the bench's parent
    // `.bench/` convention; we put it alongside the matrix test.
    const outDir = join(__dirname, '.output');
    mkdirSync(outDir, { recursive: true });
    const lines: string[] = [];
    lines.push('# Model matrix vs synth fixture (5 s, clean English TTS)');
    lines.push('');
    lines.push('| Model | WER | Words | Inference (ms) | Status |');
    lines.push('|---|---|---|---|---|');
    for (const r of rows) {
      const status = r.error
        ? `❌ ${r.error.slice(0, 80)}`
        : r.werRate < 0.2
          ? '✅'
          : r.werRate < 0.5
            ? '⚠️ borderline'
            : '❌ garbage';
      lines.push(
        `| \`${r.modelId}\` | ${(r.werRate * 100).toFixed(1)}% | ${r.wordCount} | ${r.durationMs.toFixed(0)} | ${status} |`,
      );
    }
    lines.push('');
    lines.push('## Per-model transcripts');
    lines.push('');
    for (const r of rows) {
      lines.push(`### \`${r.modelId}\``);
      lines.push('');
      if (r.error) lines.push(`**Error:** ${r.error}`);
      else lines.push(`**Transcript:** ${r.transcript}`);
      lines.push('');
    }
    writeFileSync(join(outDir, 'report.md'), lines.join('\n'));
    writeFileSync(join(outDir, 'report.json'), JSON.stringify(rows, null, 2));
    expect(rows.length).toBeGreaterThan(0);
  });
});
