// Model verification matrix. Runs every model in `MODELS` across a small
// set of audio fixtures (clean short TTS, real short speech, real medium
// speech with background noise) via the production `runWhisper` adapter,
// and writes a per-(model, fixture) report covering WER vs ground truth,
// raw transcript, and inference time. This is used when importing a new
// model to confirm it actually transcribes audio across realistic
// conditions before exposing it in the picker. It is NOT in the
// autonomous loop's regression bench because adding models should not
// gate the loop and pulling 4 model weights every loop iteration is
// wasteful.
//
// Run with: `npm run bench:matrix`. Output is written to
// `tests/matrix/.output/report.{md,json}` because vitest's default
// reporter swallows console.log on passing tests.
//
// Failure semantics: each (model, fixture) pair gets a soft assertion
// that it produced a non-empty transcript. Hard quality assertions live
// in the integration suite (tiny.en only); this file is a diagnostic.

import { beforeAll, describe, expect, it } from 'vitest';
import { join } from 'node:path';
import { MODELS, isMultilingual, supportsWordTimestamps, type ModelId } from '../../src/lib/audio';
import { runWhisper } from '../../src/lib/transcription/whisperAdapter';
import { mkdirSync, writeFileSync } from 'node:fs';
import { createNodeWhisperPipeline } from '../integration/lib/whisperNode';
import { loadGroundtruth, loadWav16kMono } from '../integration/lib/fixture';
import { wer, normalizeForScoring } from '../integration/lib/metrics';

const FIXTURE_DIR = join(__dirname, '..', 'fixtures');

// Three fixtures span the regimes the user actually hits: short clean
// TTS, short real speech, medium real-plus-noise. The 335 s `long` and
// 720 s `jfk-inaugural` fixtures are intentionally excluded; per-model
// cold inference on those is minutes per model on Node CPU and offers
// little extra signal beyond apollo11 for "does the model handle
// realistic audio at all".
const FIXTURES: Array<{ id: string; wav: string; json: string }> = [
  { id: 'synth', wav: 'synth.16k.wav', json: 'synth.json' },
  { id: 'jfk', wav: 'jfk.16k.wav', json: 'jfk.json' },
  { id: 'apollo11', wav: 'apollo11-liftoff.wav', json: 'apollo11-liftoff.json' },
];

interface MatrixRow {
  modelId: ModelId;
  fixtureId: string;
  durationS: number;
  inferenceMs: number;
  wordCount: number;
  werRate: number;
  transcript: string;
  error?: string;
}

const rows: MatrixRow[] = [];

describe('Model matrix vs multiple fixtures', () => {
  const fixtures: Record<
    string,
    { audio: ReturnType<typeof loadWav16kMono>; groundtruth: ReturnType<typeof loadGroundtruth> }
  > = {};

  beforeAll(() => {
    for (const f of FIXTURES) {
      fixtures[f.id] = {
        audio: loadWav16kMono(join(FIXTURE_DIR, f.wav), f.id),
        groundtruth: loadGroundtruth(join(FIXTURE_DIR, f.json)),
      };
    }
  });

  for (const m of MODELS) {
    for (const f of FIXTURES) {
      it(`${m.id} on ${f.id}`, async () => {
        const t0 = performance.now();
        try {
          const pipeline = await createNodeWhisperPipeline(m.id);
          const { audio, groundtruth } = fixtures[f.id];
          const result = await runWhisper(pipeline, audio.samples, {
            language: isMultilingual(m.id) ? 'en' : undefined,
            offsetSeconds: 0,
            requestWordTimestamps: supportsWordTimestamps(m.id),
          });
          const inferenceMs = performance.now() - t0;
          const transcript = result.words
            .map((w) => w.text)
            .join(' ')
            .replace(/\s+/g, ' ')
            .trim();
          const werResult = wer(groundtruth.transcript, transcript);
          rows.push({
            modelId: m.id,
            fixtureId: f.id,
            durationS: audio.durationS,
            inferenceMs,
            wordCount: result.words.length,
            werRate: werResult.rate,
            transcript,
          });
          /* eslint-disable no-console */
          console.log(`\n  ===== ${m.id} / ${f.id} (${audio.durationS.toFixed(1)} s) =====`);
          console.log(`  WER       : ${(werResult.rate * 100).toFixed(2)}%`);
          console.log(`  words     : ${result.words.length}`);
          console.log(`  inference : ${inferenceMs.toFixed(0)} ms`);
          console.log(`  transcript: ${normalizeForScoring(transcript).slice(0, 200)}`);
          /* eslint-enable no-console */
          expect(transcript.length).toBeGreaterThan(0);
        } catch (err) {
          rows.push({
            modelId: m.id,
            fixtureId: f.id,
            durationS: fixtures[f.id]?.audio.durationS ?? 0,
            inferenceMs: performance.now() - t0,
            wordCount: 0,
            werRate: 1,
            transcript: '',
            error: (err as Error).message,
          });
          throw err;
        }
      }, 600_000);
    }
  }

  it('writes aggregated matrix report', () => {
    const outDir = join(__dirname, '.output');
    mkdirSync(outDir, { recursive: true });
    const lines: string[] = [];
    lines.push('# Model matrix');
    lines.push('');
    lines.push('| Model | Fixture | Duration (s) | WER | Words | Inference (ms) | Status |');
    lines.push('|---|---|---|---|---|---|---|');
    for (const r of rows) {
      const status = r.error
        ? `❌ ${r.error.slice(0, 80)}`
        : r.werRate < 0.2
          ? '✅'
          : r.werRate < 0.5
            ? '⚠️ borderline'
            : '❌ garbage';
      lines.push(
        `| \`${r.modelId}\` | ${r.fixtureId} | ${r.durationS.toFixed(1)} | ${(r.werRate * 100).toFixed(1)}% | ${r.wordCount} | ${r.inferenceMs.toFixed(0)} | ${status} |`,
      );
    }
    lines.push('');
    lines.push('## Per-(model, fixture) transcripts');
    lines.push('');
    for (const r of rows) {
      lines.push(`### \`${r.modelId}\` / \`${r.fixtureId}\``);
      lines.push('');
      if (r.error) lines.push(`**Error:** ${r.error}`);
      else lines.push(`**Transcript:** ${r.transcript.slice(0, 400)}`);
      lines.push('');
    }
    writeFileSync(join(outDir, 'report.md'), lines.join('\n'));
    writeFileSync(join(outDir, 'report.json'), JSON.stringify(rows, null, 2));
    expect(rows.length).toBeGreaterThan(0);
  });
});
