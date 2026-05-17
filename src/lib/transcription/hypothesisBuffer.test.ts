import { describe, expect, it } from 'vitest';
import { HypothesisBuffer, type TimedWord } from './hypothesisBuffer';

const word = (text: string, start: number, end: number): TimedWord => ({ text, start, end });

describe('HypothesisBuffer', () => {
  it('first ingest with empty tentative commits nothing and stores everything as tentative', () => {
    const buf = new HypothesisBuffer();
    const committed = buf.ingest([word('hello', 0, 0.5), word('world', 0.5, 1)]);
    expect(committed).toEqual([]);
    expect(buf.getCommitted()).toEqual([]);
    expect(buf.getTentative().map((w) => w.text)).toEqual(['hello', 'world']);
  });

  it('full agreement on two consecutive ingests commits every matched word', () => {
    const buf = new HypothesisBuffer();
    buf.ingest([word('hello', 0, 0.5), word('world', 0.5, 1)]);
    const committed = buf.ingest([word('hello', 0, 0.5), word('world', 0.5, 1)]);
    expect(committed.map((w) => w.text)).toEqual(['hello', 'world']);
    expect(buf.getTentative()).toEqual([]);
  });

  it('partial agreement commits the prefix and keeps the new tail as tentative', () => {
    const buf = new HypothesisBuffer();
    buf.ingest([word('hello', 0, 0.5), word('world', 0.5, 1)]);
    // The next hypothesis matches "hello" and "world" but extends with "today".
    const committed = buf.ingest([
      word('hello', 0, 0.5),
      word('world', 0.5, 1),
      word('today', 1, 1.5),
    ]);
    expect(committed.map((w) => w.text)).toEqual(['hello', 'world']);
    expect(buf.getCommitted().map((w) => w.text)).toEqual(['hello', 'world']);
    expect(buf.getTentative().map((w) => w.text)).toEqual(['today']);
  });

  it('stops committing at the first mismatch and adopts the new tail', () => {
    const buf = new HypothesisBuffer();
    buf.ingest([word('hello', 0, 0.5), word('there', 0.5, 1)]);
    // Whisper revised "there" to "world"; "hello" still agrees.
    const committed = buf.ingest([word('hello', 0, 0.5), word('world', 0.5, 1)]);
    expect(committed.map((w) => w.text)).toEqual(['hello']);
    expect(buf.getTentative().map((w) => w.text)).toEqual(['world']);
  });

  it('regression: does NOT double the last committed word on repeated drain ingests (1s-word boundary)', () => {
    // This is the live-streaming doubling the user reported. The producer
    // writes 1 s chunks. During drainAndFinalise the consumer runs Whisper
    // on the same audio repeatedly until the buffer is "settled". With the
    // previous `Math.abs(words[0].start - lastCommittedTime) >= 1` gate,
    // any word that was roughly 1 s long ended up with gap == 1.0 exactly
    // and skipped dedup, so the drain loop re-committed it every two iters.
    const buf = new HypothesisBuffer();

    // Walk the buffer up to "committed=[Hello, world], tentative=[goodbye]"
    // the way a live session would: each ingest extends the previous one.
    buf.ingest([word('Hello', 0, 0.5), word('world', 0.5, 1.5)]);
    buf.ingest([word('Hello', 0, 0.5), word('world', 0.5, 1.5), word('goodbye', 2, 3)]);
    expect(buf.getCommitted().map((w) => w.text)).toEqual(['Hello', 'world']);
    expect(buf.getTentative().map((w) => w.text)).toEqual(['goodbye']);
    expect(buf.getLastCommittedTime()).toBe(1.5);

    // Drain iter 1 (after Stop): same hypothesis. "goodbye" commits.
    buf.ingest([word('Hello', 0, 0.5), word('world', 0.5, 1.5), word('goodbye', 2, 3)]);
    expect(buf.getCommitted().map((w) => w.text)).toEqual(['Hello', 'world', 'goodbye']);
    expect(buf.getTentative()).toEqual([]);
    expect(buf.getLastCommittedTime()).toBe(3);

    // Drain iters 2..N on identical audio: must NOT double "goodbye".
    for (let i = 0; i < 10; i++) {
      buf.ingest([word('Hello', 0, 0.5), word('world', 0.5, 1.5), word('goodbye', 2, 3)]);
    }
    expect(buf.getCommitted().map((w) => w.text)).toEqual(['Hello', 'world', 'goodbye']);
    expect(buf.getTentative()).toEqual([]);
  });

  it('regression: does NOT double the initial committed word on repeated ingests', () => {
    // Variant of the above at the start of a session: the very first
    // committed word is also exactly 1 s long.
    const buf = new HypothesisBuffer();
    buf.ingest([word('hello', 0, 1)]);
    buf.ingest([word('hello', 0, 1)]);
    expect(buf.getCommitted().map((w) => w.text)).toEqual(['hello']);
    expect(buf.getTentative()).toEqual([]);

    for (let i = 0; i < 10; i++) {
      buf.ingest([word('hello', 0, 1)]);
    }
    expect(buf.getCommitted().map((w) => w.text)).toEqual(['hello']);
    expect(buf.getTentative()).toEqual([]);
  });

  it('dedups re-emitted committed tail at the chunk seam', () => {
    // Whisper, given an overlapping audio window, often re-emits the last
    // committed words at the head of its new hypothesis. stripCommittedTailOverlap
    // must remove them before LA-2 sees the candidate list.
    const buf = new HypothesisBuffer();
    buf.ingest([word('hello', 0, 0.5), word('world', 0.5, 1)]);
    buf.ingest([word('hello', 0, 0.5), word('world', 0.5, 1)]);
    expect(buf.getCommitted().map((w) => w.text)).toEqual(['hello', 'world']);

    // New audio added; Whisper re-emits the committed tail at the head.
    buf.ingest([
      word('hello', 0, 0.5),
      word('world', 0.5, 1),
      word('today', 1, 1.5),
    ]);
    // "today" should NOT commit yet (single ingest at the new tail).
    expect(buf.getTentative().map((w) => w.text)).toEqual(['today']);

    // Confirm: another ingest with same content commits "today" exactly once.
    buf.ingest([
      word('hello', 0, 0.5),
      word('world', 0.5, 1),
      word('today', 1, 1.5),
    ]);
    expect(buf.getCommitted().map((w) => w.text)).toEqual(['hello', 'world', 'today']);
  });

  it('does not eat a genuine repetition that the user actually said', () => {
    // The dedup must be conservative: when the user really repeats a word,
    // both occurrences are kept.
    const buf = new HypothesisBuffer();
    buf.ingest([word('hello', 0, 0.5), word('hello', 0.5, 1)]);
    buf.ingest([word('hello', 0, 0.5), word('hello', 0.5, 1), word('world', 1, 1.5)]);
    expect(buf.getCommitted().map((w) => w.text)).toEqual(['hello', 'hello']);
    expect(buf.getTentative().map((w) => w.text)).toEqual(['world']);
  });

  it('does not dedup when the new hypothesis starts clearly past the committed boundary', () => {
    // After a long pause (force-commit + large anchor advance), a new
    // hypothesis comes in well past lastCommittedTime. Dedup must be
    // skipped: the words there are NEW occurrences, not re-emissions.
    const buf = new HypothesisBuffer();
    buf.ingest([word('hello', 0, 0.5)]);
    buf.ingest([word('hello', 0, 0.5)]);
    expect(buf.getCommitted().map((w) => w.text)).toEqual(['hello']);
    expect(buf.getLastCommittedTime()).toBe(0.5);

    // Now a fresh "hello again" arriving 3 s later (start > last + 1).
    buf.ingest([word('hello', 3, 3.5), word('again', 3.5, 4)]);
    buf.ingest([word('hello', 3, 3.5), word('again', 3.5, 4)]);
    expect(buf.getCommitted().map((w) => w.text)).toEqual(['hello', 'hello', 'again']);
  });

  it('dropAlreadyCovered drops words that end well before the committed boundary', () => {
    const buf = new HypothesisBuffer();
    // Commit "alpha beta gamma".
    buf.ingest([word('alpha', 0, 0.5), word('beta', 0.5, 1), word('gamma', 1, 1.5)]);
    buf.ingest([word('alpha', 0, 0.5), word('beta', 0.5, 1), word('gamma', 1, 1.5)]);
    expect(buf.getCommitted().map((w) => w.text)).toEqual(['alpha', 'beta', 'gamma']);
    expect(buf.getLastCommittedTime()).toBe(1.5);

    // Whisper re-emits everything plus extends with "delta epsilon".
    // alpha, beta should be dropped by dropAlreadyCovered (end < last - 0.1).
    // gamma straddles the boundary; stripCommittedTailOverlap removes it.
    buf.ingest([
      word('alpha', 0, 0.5),
      word('beta', 0.5, 1),
      word('gamma', 1, 1.5),
      word('delta', 1.5, 2),
      word('epsilon', 2, 2.5),
    ]);
    expect(buf.getTentative().map((w) => w.text)).toEqual(['delta', 'epsilon']);
    expect(buf.getCommitted().map((w) => w.text)).toEqual(['alpha', 'beta', 'gamma']);
  });

  it('forceCommit moves tentative into committed and updates lastCommittedTime', () => {
    const buf = new HypothesisBuffer();
    buf.ingest([word('hello', 0, 0.5), word('world', 0.5, 1)]);
    const pinned = buf.forceCommit();
    expect(pinned.map((w) => w.text)).toEqual(['hello', 'world']);
    expect(buf.getCommitted().map((w) => w.text)).toEqual(['hello', 'world']);
    expect(buf.getTentative()).toEqual([]);
    expect(buf.getLastCommittedTime()).toBe(1);
  });

  it('forceCommit on empty tentative is a no-op', () => {
    const buf = new HypothesisBuffer();
    expect(buf.forceCommit()).toEqual([]);
    expect(buf.getCommitted()).toEqual([]);
    expect(buf.getLastCommittedTime()).toBe(0);
  });

  it('reset clears committed, tentative, and lastCommittedTime', () => {
    const buf = new HypothesisBuffer();
    buf.ingest([word('hello', 0, 0.5)]);
    buf.ingest([word('hello', 0, 0.5)]);
    expect(buf.getCommitted().length).toBe(1);
    buf.reset();
    expect(buf.getCommitted()).toEqual([]);
    expect(buf.getTentative()).toEqual([]);
    expect(buf.getLastCommittedTime()).toBe(0);
  });

  it('normalises case and punctuation when comparing words', () => {
    const buf = new HypothesisBuffer();
    // First hypothesis: leading capital, trailing comma.
    buf.ingest([word('Hello,', 0, 0.5), word('world', 0.5, 1)]);
    // Second hypothesis: different casing on both words; period vs comma.
    // LA-2 normalises before comparison, so both should still agree and
    // commit; the CURRENT hypothesis's text is what lands in committed.
    buf.ingest([word('hello', 0, 0.5), word('World.', 0.5, 1)]);
    expect(buf.getCommitted().map((w) => w.text)).toEqual(['hello', 'World.']);
  });

  it('preserves tentative when the hypothesis collapses to empty after upstream filters', () => {
    // Models the regurgitation-strip case: a tentative word is waiting for a
    // confirming ingest, but the next Whisper call returns text that is
    // entirely prompt regurgitation. The adapter strips it to `[]`. LA-2
    // must NOT wipe the pending tentative just because this tick was a
    // no-op upstream.
    const buf = new HypothesisBuffer();
    buf.ingest([word('hello', 0, 0.5), word('world', 0.5, 1)]);
    expect(buf.getTentative().map((w) => w.text)).toEqual(['hello', 'world']);

    // Empty hypothesis: no commits, tentative preserved.
    const committed = buf.ingest([]);
    expect(committed).toEqual([]);
    expect(buf.getTentative().map((w) => w.text)).toEqual(['hello', 'world']);

    // Next real hypothesis matches: tentative graduates as expected.
    buf.ingest([word('hello', 0, 0.5), word('world', 0.5, 1)]);
    expect(buf.getCommitted().map((w) => w.text)).toEqual(['hello', 'world']);
    expect(buf.getTentative()).toEqual([]);
  });
});
