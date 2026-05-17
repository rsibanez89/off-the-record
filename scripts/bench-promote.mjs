// Promote the most recent integration-test run from `tests/integration/.bench/`
// into the committed baseline at `tests/integration/baselines.json`.
//
// Run after the change you are evaluating has produced an `improvement`
// or `neutral` verdict from `npm run bench:compare`. Promoting on a
// regression is allowed but discouraged: it locks in the regression as
// the new normal.
//
// Also regenerates BENCHMARKS.md so the human-readable summary stays in
// sync with the committed numbers.

import { readFileSync, readdirSync, writeFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const BENCH_DIR = 'tests/integration/.bench';
const BASELINE_PATH = 'tests/integration/baselines.json';
const BENCHMARKS_MD = 'BENCHMARKS.md';

if (!existsSync(BENCH_DIR)) {
  console.error(`[bench-promote] no candidate run at ${BENCH_DIR}. Run \`npm run test:integration\` first.`);
  process.exit(1);
}

const fixtures = {};
let model = null;
for (const f of readdirSync(BENCH_DIR)) {
  if (!f.endsWith('.json') || f === 'comparison.json') continue;
  const m = JSON.parse(readFileSync(join(BENCH_DIR, f), 'utf8'));
  fixtures[m.fixtureId] = m;
  model = m.model;
}

const fixtureIds = Object.keys(fixtures);
if (fixtureIds.length === 0) {
  console.error('[bench-promote] no fixtures found in candidate run.');
  process.exit(1);
}

const capturedAt = new Date().toISOString().slice(0, 10);
const baseline = { capturedAt, model, fixtures };
writeFileSync(BASELINE_PATH, JSON.stringify(baseline, null, 2) + '\n');
console.log(`[bench-promote] wrote ${BASELINE_PATH} (${fixtureIds.length} fixtures, captured ${capturedAt}).`);

// Regenerate BENCHMARKS.md.
const orderedIds = ['jfk', 'synth', 'apollo11', 'long', 'jfk-inaugural'].filter((id) => fixtures[id]);
const otherIds = fixtureIds.filter((id) => !orderedIds.includes(id));
const allIds = [...orderedIds, ...otherIds];

const lines = [];
lines.push('# Benchmarks');
lines.push('');
lines.push(`Captured ${capturedAt} with \`${model}\` on Node + onnxruntime-node CPU.`);
lines.push('');
lines.push('See `docs/AUTONOMOUS-LOOP.md` for the regression-driven improvement protocol.');
lines.push('');
lines.push('Run `npm run bench` to refresh the candidate snapshot, `npm run bench:compare` to diff against this baseline, `npm run bench:promote` to update it.');
lines.push('');
lines.push('## Per-fixture metrics');
lines.push('');
lines.push('| Fixture | Duration | Ticks | WER vs gt | CER vs gt | WER offline | Streaming penalty | Hallucinations | Realtime ×');
lines.push('|---|---|---|---|---|---|---|---|---|');

for (const id of allIds) {
  const m = fixtures[id];
  const wer = (m.werVsGt * 100).toFixed(2);
  const cer = (m.cerVsGt * 100).toFixed(2);
  const offWer = (m.werOfflineVsGt * 100).toFixed(2);
  const penalty = m.streamingPenaltyPp.toFixed(2);
  const rt = m.realtimeFactor.toFixed(2);
  lines.push(`| ${id} | ${m.durationS} s | ${m.ticks} | ${wer}% | ${cer}% | ${offWer}% | ${penalty}pp | ${m.hallucinationStrings} | ${rt}× |`);
}

lines.push('');
lines.push('## Long-session stability (jfk-inaugural fixture)');
lines.push('');
const ji = fixtures['jfk-inaugural'];
if (ji) {
  const drift = ji.latencyDriftQ4Q1 != null ? ji.latencyDriftQ4Q1.toFixed(2) : 'n/a';
  lines.push(`- Tick count: ${ji.ticks} (${ji.inferenceTicks} inference, ${ji.forceSlideTicks} force-slide, ${ji.hallucinationDeferTicks} hallucination-defer)`);
  lines.push(`- Latency drift Q4/Q1: ${drift}× (above 1.2× would suggest per-tick state growth or a memory leak)`);
  lines.push(`- Realtime factor: ${ji.realtimeFactor.toFixed(2)}×`);
}
lines.push('');
lines.push('## Decision rule');
lines.push('');
lines.push('A candidate run is a **regression** if ANY fixture shows:');
lines.push('- `werVsGt` rose by more than 0.5pp');
lines.push('- `hallucinationStrings` increased');
lines.push('- `streamingPenaltyPp` rose by more than 2.0pp');
lines.push('- `latencyDriftQ4Q1` rose by more than 0.2× (long-session only)');
lines.push('- `realtimeFactor` dropped below 70% of baseline');
lines.push('');
lines.push('A candidate is an **improvement** if at least one fixture\'s `werVsGt` dropped by ≥ 0.5pp AND no fixture regressed by the rule above.');
lines.push('');
lines.push('Otherwise: **neutral**.');
lines.push('');

writeFileSync(BENCHMARKS_MD, lines.join('\n'));
console.log(`[bench-promote] wrote ${BENCHMARKS_MD}.`);
