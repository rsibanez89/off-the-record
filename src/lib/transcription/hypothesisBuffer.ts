// LocalAgreement-2 hypothesis buffer.
//
// Reference: Machacek, Dabre, Bojar (2023), "Turning Whisper into Real-Time
// Transcription System". Implementation faithful to ufal/whisper_streaming's
// `HypothesisBuffer` class in `whisper_online.py`.
//
// Invariant: a word is committed only when it appears in the same position in
// two consecutive Whisper hypotheses (after 5-gram tail dedup). Once
// committed, a word is monotonically stable. It never reverts to tentative.

export interface TimedWord {
  text: string;
  start: number; // absolute time in seconds, in chunk-startedAt timeline
  end: number;
}

// Punctuation to strip when comparing words. Includes en-dash (U+2013) and
// em-dash (U+2014) by codepoint so this file contains no literal long-dash
// characters (project lint rule).
const STRIP_PUNCT_ASCII = new Set('.,!?;:"\'()[]-'.split(''));
const STRIP_PUNCT_CODEPOINTS = new Set([0x2013, 0x2014]);

function norm(s: string): string {
  let out = '';
  for (const ch of s.trim().toLowerCase()) {
    if (STRIP_PUNCT_ASCII.has(ch)) continue;
    if (STRIP_PUNCT_CODEPOINTS.has(ch.charCodeAt(0))) continue;
    out += ch;
  }
  return out;
}

export class HypothesisBuffer {
  private committed: TimedWord[] = [];
  // `tentative` holds the surviving tail of the previous iteration's `new`.
  // The next iteration's `new` is compared against it position-wise to decide
  // what graduates to `committed`.
  private tentative: TimedWord[] = [];
  private lastCommittedTime = 0;

  /**
   * Feed a fresh hypothesis from Whisper and run one round of LocalAgreement-2.
   * Returns the words that were newly committed by this call (for logging and
   * stats; the buffer also stores them internally).
   */
  ingest(hypothesis: TimedWord[]): TimedWord[] {
    const next = this.dropAlreadyCovered(hypothesis);
    const deduped = this.stripCommittedTailOverlap(next);
    return this.runLocalAgreement(deduped);
  }

  /**
   * Drop words whose end is at or before the last committed boundary, with a
   * 0.1 s tolerance for timestamp jitter from Whisper. Without this, Whisper
   * sometimes re-emits old material with slightly different timestamps and
   * blocks fresh commits.
   */
  private dropAlreadyCovered(words: TimedWord[]): TimedWord[] {
    const cutoff = this.lastCommittedTime - 0.1;
    return words.filter((w) => w.end > cutoff);
  }

  /**
   * 5-gram tail overlap dedup. If the head of `words` repeats the tail of the
   * committed transcript (Whisper does this when fed overlapping audio), strip
   * the overlap from `words`. Greedy: longest overlap up to 5 wins.
   *
   * Skip dedup only when the new hypothesis is clearly past the committed
   * boundary by enough that overlap is impossible. We use the candidate's
   * END time, not START, because a word that overlaps committed has end at
   * or after lastCommittedTime even if its start is far before. The previous
   * check `Math.abs(words[0].start - lastCommittedTime) >= 1` would fire
   * exactly when a roughly-1-second word fully overlapped committed (gap was
   * exactly its duration), letting the re-emission slip through as a
   * duplicate. Boundary words and the drain loop both made this visible.
   */
  private stripCommittedTailOverlap(words: TimedWord[]): TimedWord[] {
    if (words.length === 0 || this.committed.length === 0) return words;
    // Words that clearly start past the committed boundary cannot be
    // re-emissions of committed material; skip the comparison.
    if (words[0].start > this.lastCommittedTime + 1) return words;

    const maxK = Math.min(this.committed.length, words.length, 5);
    for (let k = maxK; k >= 1; k--) {
      const committedTail = this.committed
        .slice(this.committed.length - k)
        .map((w) => norm(w.text))
        .join(' ');
      const newHead = words
        .slice(0, k)
        .map((w) => norm(w.text))
        .join(' ');
      if (committedTail === newHead) {
        return words.slice(k);
      }
    }
    return words;
  }

  /**
   * Core LocalAgreement-2 step. Walk `current` and `tentative` in parallel,
   * committing each word that matches. Stop at the first mismatch. The
   * surviving tail of `current` becomes the new `tentative` for next round.
   */
  private runLocalAgreement(current: TimedWord[]): TimedWord[] {
    const justCommitted: TimedWord[] = [];
    let i = 0;
    while (i < current.length && i < this.tentative.length) {
      if (norm(current[i].text) !== norm(this.tentative[i].text)) break;
      justCommitted.push(current[i]);
      this.lastCommittedTime = current[i].end;
      i++;
    }
    this.committed.push(...justCommitted);
    this.tentative = current.slice(i);
    return justCommitted;
  }

  /**
   * Force-commit the tentative buffer. Used by callers that have external
   * knowledge that no more agreement will come for the current audio window
   * (silence, hallucination, MAX_WINDOW_S force-slide).
   */
  forceCommit(): TimedWord[] {
    if (this.tentative.length === 0) return [];
    const committed = this.tentative;
    this.lastCommittedTime = committed[committed.length - 1].end;
    this.committed.push(...committed);
    this.tentative = [];
    return committed;
  }

  getCommitted(): readonly TimedWord[] {
    return this.committed;
  }

  getTentative(): readonly TimedWord[] {
    return this.tentative;
  }

  getLastCommittedTime(): number {
    return this.lastCommittedTime;
  }

  reset(): void {
    this.committed = [];
    this.tentative = [];
    this.lastCommittedTime = 0;
  }
}
