// Compare the most recent integration-test run against the committed
// baseline. Decides if the candidate is an `improvement`, a `regression`,
// or `neutral`, and prints a per-fixture diff table.
//
// Inputs:
//   - `tests/integration/.bench/*.json`      candidate (produced by tests)
//   - `tests/integration/baselines.json`     committed baseline
//
// Output:
//   - Human-readable Markdown table on stdout
//   - `tests/integration/.bench/comparison.json` with the structured verdict
//
// Exit codes:
//   0  improvement OR neutral (safe to keep the change)
//   1  regression (revert the change)
//   2  missing baseline / no candidate run / other setup error
//
// Decision rule (regression if ANY of):
//   - werVsGt rose by > 0.5pp on any fixture
//   - hallucinationStrings rose by > 0 on any fixture
//   - streamingPenaltyPp rose by > 2.0pp on any fixture
//   - latencyDriftQ4Q1 rose by > 0.2x on any fixture that reports it
//   - realtimeFactor dropped by > 30% on any fixture
// Improvement if BOTH:
//   - At least one fixture's werVsGt dropped by >= 0.5pp
//   - No fixture regressed by the rule above
// Otherwise: neutral.

import { readFileSync, readdirSync, writeFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const BENCH_DIR = 'tests/integration/.bench';
const BASELINE_PATH = 'tests/integration/baselines.json';

if (!existsSync(BENCH_DIR)) {
  console.error(`[bench-compare] no candidate run found at ${BENCH_DIR}.`);
  console.error('Run `npm run test:integration` first.');
  process.exit(2);
}
if (!existsSync(BASELINE_PATH)) {
  console.error(`[bench-compare] no committed baseline at ${BASELINE_PATH}.`);
  console.error('Bootstrap by running `npm run bench:promote` once.');
  process.exit(2);
}

const baseline = JSON.parse(readFileSync(BASELINE_PATH, 'utf8'));
const candidate = { fixtures: {} };
for (const f of readdirSync(BENCH_DIR)) {
  if (!f.endsWith('.json') || f === 'comparison.json') continue;
  const m = JSON.parse(readFileSync(join(BENCH_DIR, f), 'utf8'));
  candidate.fixtures[m.fixtureId] = m;
}

const fixtureIds = Object.keys(candidate.fixtures).sort();
const perFixture = [];
const regressions = [];
const improvements = [];

const REGRESS_WER_PP = 0.5;
const IMPROVE_WER_PP = 0.5;
const REGRESS_PENALTY_PP = 2.0;
const REGRESS_DRIFT_DELTA = 0.2;
const REGRESS_REALTIME_RATIO = 0.7;

function pp(n) {
  return (n * 100).toFixed(2);
}
function arrow(delta, regressIfPositive = true) {
  if (Math.abs(delta) < 1e-9) return '·';
  if (regressIfPositive) return delta > 0 ? '↑' : '↓';
  return delta > 0 ? '↑' : '↓';
}

for (const id of fixtureIds) {
  const c = candidate.fixtures[id];
  const b = baseline.fixtures?.[id];
  if (!b) {
    perFixture.push({ id, status: 'new', cand: c });
    continue;
  }

  const werDelta = (c.werVsGt - b.werVsGt) * 100;
  const cerDelta = (c.cerVsGt - b.cerVsGt) * 100;
  const penaltyDelta = c.streamingPenaltyPp - b.streamingPenaltyPp;
  const hallucDelta = c.hallucinationStrings - b.hallucinationStrings;
  const realtimeRatio = c.realtimeFactor / (b.realtimeFactor || 1);
  const driftDelta =
    c.latencyDriftQ4Q1 != null && b.latencyDriftQ4Q1 != null
      ? c.latencyDriftQ4Q1 - b.latencyDriftQ4Q1
      : null;

  const regressed =
    werDelta > REGRESS_WER_PP ||
    hallucDelta > 0 ||
    penaltyDelta > REGRESS_PENALTY_PP ||
    (driftDelta != null && driftDelta > REGRESS_DRIFT_DELTA) ||
    realtimeRatio < REGRESS_REALTIME_RATIO;

  const improvedHere = werDelta <= -IMPROVE_WER_PP;

  if (regressed) regressions.push({ id, werDelta, hallucDelta, penaltyDelta, driftDelta });
  if (improvedHere && !regressed) improvements.push({ id, werDelta });

  perFixture.push({
    id,
    status: regressed ? 'regression' : improvedHere ? 'improvement' : 'neutral',
    werDelta,
    cerDelta,
    penaltyDelta,
    hallucDelta,
    realtimeRatio,
    driftDelta,
    cand: c,
    base: b,
  });
}

let verdict;
if (regressions.length > 0) verdict = 'regression';
else if (improvements.length > 0) verdict = 'improvement';
else verdict = 'neutral';

const comparison = {
  verdict,
  rules: {
    regressWerPp: REGRESS_WER_PP,
    improveWerPp: IMPROVE_WER_PP,
    regressPenaltyPp: REGRESS_PENALTY_PP,
    regressDriftDelta: REGRESS_DRIFT_DELTA,
    regressRealtimeRatio: REGRESS_REALTIME_RATIO,
  },
  baselineCapturedAt: baseline.capturedAt,
  baselineModel: baseline.model,
  candidateModel: candidate.fixtures[fixtureIds[0]]?.model,
  perFixture: perFixture.map((p) => ({
    id: p.id,
    status: p.status,
    werDelta: p.werDelta,
    cerDelta: p.cerDelta,
    penaltyDelta: p.penaltyDelta,
    hallucDelta: p.hallucDelta,
    realtimeRatio: p.realtimeRatio,
    driftDelta: p.driftDelta,
  })),
  improvements,
  regressions,
};
writeFileSync(join(BENCH_DIR, 'comparison.json'), JSON.stringify(comparison, null, 2) + '\n');

// Human-readable output.
const verdictLabel = {
  improvement: '✅ improvement',
  neutral: '➖ neutral',
  regression: '❌ regression',
}[verdict];

console.log('## Benchmark comparison');
console.log('');
console.log(`Verdict: ${verdictLabel}`);
console.log(`Baseline captured: ${baseline.capturedAt}`);
console.log(`Model: ${comparison.candidateModel}`);
console.log('');
console.log('| Fixture | WER live (Δpp) | CER (Δpp) | Streaming penalty (Δpp) | Hallucinations (Δ) | Status |');
console.log('|---|---|---|---|---|---|');
for (const p of perFixture) {
  if (p.status === 'new') {
    console.log(`| ${p.id} | ${pp(p.cand.werVsGt)}% (new) | ${pp(p.cand.cerVsGt)}% (new) | ${p.cand.streamingPenaltyPp.toFixed(2)}pp (new) | ${p.cand.hallucinationStrings} (new) | new |`);
    continue;
  }
  const werCell = `${pp(p.cand.werVsGt)}% (${arrow(p.werDelta)} ${p.werDelta >= 0 ? '+' : ''}${p.werDelta.toFixed(2)}pp)`;
  const cerCell = `${pp(p.cand.cerVsGt)}% (${arrow(p.cerDelta)} ${p.cerDelta >= 0 ? '+' : ''}${p.cerDelta.toFixed(2)}pp)`;
  const penaltyCell = `${p.cand.streamingPenaltyPp.toFixed(2)}pp (${arrow(p.penaltyDelta)} ${p.penaltyDelta >= 0 ? '+' : ''}${p.penaltyDelta.toFixed(2)}pp)`;
  const hallucCell = `${p.cand.hallucinationStrings} (${p.hallucDelta >= 0 ? '+' : ''}${p.hallucDelta})`;
  const statusCell = p.status === 'improvement' ? '✅' : p.status === 'regression' ? '❌' : '·';
  console.log(`| ${p.id} | ${werCell} | ${cerCell} | ${penaltyCell} | ${hallucCell} | ${statusCell} |`);
}

if (regressions.length > 0) {
  console.log('');
  console.log('### Regressions');
  for (const r of regressions) {
    const reasons = [];
    if (r.werDelta > REGRESS_WER_PP) reasons.push(`WER +${r.werDelta.toFixed(2)}pp (> ${REGRESS_WER_PP}pp threshold)`);
    if (r.hallucDelta > 0) reasons.push(`+${r.hallucDelta} hallucination string(s)`);
    if (r.penaltyDelta > REGRESS_PENALTY_PP) reasons.push(`streaming penalty +${r.penaltyDelta.toFixed(2)}pp`);
    if (r.driftDelta != null && r.driftDelta > REGRESS_DRIFT_DELTA) reasons.push(`latency drift +${r.driftDelta.toFixed(2)}x`);
    console.log(`- **${r.id}**: ${reasons.join(', ')}`);
  }
}
if (improvements.length > 0) {
  console.log('');
  console.log('### Improvements');
  for (const i of improvements) {
    console.log(`- **${i.id}**: WER ${i.werDelta.toFixed(2)}pp`);
  }
}

console.log('');
console.log(`Detail JSON written to ${join(BENCH_DIR, 'comparison.json')}`);
if (verdict === 'regression') process.exit(1);
process.exit(0);
