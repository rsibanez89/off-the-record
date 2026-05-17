// Metrics for ASR integration tests: WER, CER, hallucination-substring count.
//
// Per the research, there is no maintained JS jiwer port; we implement a
// ~30-line edit distance on tokens / characters here. The normalizer is the
// load-bearing piece: Whisper's `EnglishTextNormalizer` (whisper repo) does
// far more than this, but the rules below cover the common cases that
// otherwise dominate the WER number.

// Long-dash codepoints referenced numerically to satisfy the project lint
// rule against literal U+2013 / U+2014 in source.
const EN_DASH = String.fromCharCode(0x2013);
const EM_DASH = String.fromCharCode(0x2014);
const DASH_REGEX = new RegExp('[-' + EN_DASH + EM_DASH + ']', 'g');

/**
 * Lowercase, strip punctuation, collapse whitespace, normalise apostrophes.
 * Mirrors Whisper's `EnglishTextNormalizer` at a 90/10 level. Not a 1:1 port
 * since we are scoring a tiny test set, not publishing a benchmark.
 *
 * Apostrophes are dropped (so "don't" and "dont" match) but spell-outs
 * ("do not") remain distinct, which is correct: spelling out a contraction
 * IS a transcription difference.
 */
export function normalizeForScoring(text: string): string {
  let t = text.toLowerCase();
  // Apostrophe variants: ASCII, curly left/right.
  t = t.replace(/['‘’]/g, '');
  // Dash variants (ASCII hyphen, en-dash, em-dash) become a space.
  t = t.replace(DASH_REGEX, ' ');
  // Strip remaining punctuation.
  t = t.replace(/[.,!?;:"()\[\]{}<>/\\]/g, ' ');
  // Collapse whitespace.
  t = t.replace(/\s+/g, ' ').trim();
  return t;
}

export function tokenize(text: string): string[] {
  const n = normalizeForScoring(text);
  if (!n) return [];
  return n.split(' ');
}

/**
 * Levenshtein distance with a rolling-row DP. O(N*M) time,
 * O(min(N, M)) memory. Sufficient for the tens-to-low-hundreds-of-tokens
 * transcripts we score.
 */
function editDistance<T>(a: readonly T[], b: readonly T[]): number {
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;
  // Keep the row small by ensuring a is the shorter sequence.
  if (a.length > b.length) [a, b] = [b, a];
  let prev = new Array(a.length + 1);
  let curr = new Array(a.length + 1);
  for (let i = 0; i <= a.length; i++) prev[i] = i;
  for (let j = 1; j <= b.length; j++) {
    curr[0] = j;
    for (let i = 1; i <= a.length; i++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[i] = Math.min(
        prev[i] + 1, // deletion
        curr[i - 1] + 1, // insertion
        prev[i - 1] + cost, // substitution
      );
    }
    [prev, curr] = [curr, prev];
  }
  return prev[a.length];
}

export interface ScoreResult {
  /** edits / reference length, clamped to non-negative. */
  rate: number;
  /** Raw counts for debugging and weighted aggregation across fixtures. */
  edits: number;
  referenceLength: number;
  hypothesisLength: number;
}

export function wer(reference: string, hypothesis: string): ScoreResult {
  const r = tokenize(reference);
  const h = tokenize(hypothesis);
  const edits = editDistance(r, h);
  return {
    rate: r.length === 0 ? (h.length === 0 ? 0 : 1) : edits / r.length,
    edits,
    referenceLength: r.length,
    hypothesisLength: h.length,
  };
}

export function cer(reference: string, hypothesis: string): ScoreResult {
  const r = normalizeForScoring(reference).split('');
  const h = normalizeForScoring(hypothesis).split('');
  const edits = editDistance(r, h);
  return {
    rate: r.length === 0 ? (h.length === 0 ? 0 : 1) : edits / r.length,
    edits,
    referenceLength: r.length,
    hypothesisLength: h.length,
  };
}

/**
 * Common Whisper hallucination triggers (cargo-culted from issue threads
 * and the Bag-of-Hallucinations literature). Real-speech fixtures should
 * assert this count is zero in their committed transcript.
 */
export const HALLUCINATION_DENYLIST: readonly string[] = [
  'thanks for watching',
  'thank you for watching',
  'subscribe',
  '[blank_audio]',
  '[music]',
  '(music)',
  '(phone beeping)',
];

export function countHallucinationMatches(text: string): number {
  const lower = text.toLowerCase();
  let count = 0;
  for (const phrase of HALLUCINATION_DENYLIST) {
    if (lower.includes(phrase)) count++;
  }
  return count;
}
