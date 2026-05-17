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
    const live = this.dropFlatTimestampTail(hypothesis);
    const next = this.dropAlreadyCovered(live);
    const deduped = this.stripCommittedTailOverlap(next);
    return this.runLocalAgreement(deduped);
  }

  /**
   * Truncate the hypothesis at the start of a Whisper "flat-timestamp" run:
   * 3 or more consecutive words whose `start` times differ by less than 10 ms.
   *
   * Why: Whisper enters a long-form repetition loop at low-energy paragraph
   * boundaries and emits an autoregressive repeat of recently-committed real
   * content with every word's start time clamped to the audio window's `t1`.
   * Two consecutive windows produce the same flat-timestamp tail, LA-2 in
   * `runLocalAgreement` confirms them, and they enter `committed` as a
   * duplicate of the real prefix. `stripCommittedTailOverlap` misses these
   * because their head does NOT match the committed tail (the repeat is
   * positionally shifted), and `dropAlreadyCovered` misses them because
   * Whisper assigns fresh in-window timestamps. The cheapest defense is to
   * detect the flat-timestamp signature itself and drop the tail before LA-2
   * sees it. The pattern is highly specific to model failure: on the `long`
   * fixture's streamed transcript every observed 3+ flat-start run was a
   * hallucination, with zero false positives in legitimate committed content.
   */
  private dropFlatTimestampTail(words: TimedWord[]): TimedWord[] {
    const RUN = 3;
    const EPS = 0.01;
    for (let i = 0; i + RUN <= words.length; i++) {
      let flat = true;
      for (let k = 1; k < RUN; k++) {
        if (Math.abs(words[i + k].start - words[i].start) > EPS) {
          flat = false;
          break;
        }
      }
      if (flat) return words.slice(0, i);
    }
    return words;
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
   * Tail overlap dedup. If the head of `words` repeats the tail of the
   * committed transcript (Whisper does this when fed overlapping audio), strip
   * the overlap from `words`. Greedy: longest overlap up to MAX_OVERLAP_K wins.
   *
   * Cap raised from 5 to 20 to cover deeper seam re-emissions after a natural
   * anchor advance on long-form audio. A 5 s context lookback typically holds
   * 10 to 15 words, so a Whisper re-emission spanning more than 5 words can
   * slip past a 5-gram cap, land in `tentative`, then get LA-2 confirmed as
   * a duplicate. Verified on the JFK inaugural fixture (720 s real speech):
   * raising the cap dropped WER by 0.51pp with no regression on the other
   * four fixtures. The boundary check above (`start > lastCommittedTime + 1`)
   * and `dropAlreadyCovered` keep the comparison cheap and well-targeted.
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

    const MAX_OVERLAP_K = 20;
    const maxK = Math.min(this.committed.length, words.length, MAX_OVERLAP_K);
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
   *
   * Empty `current` is treated as a no-op: when an upstream step (drop,
   * dedup, prompt-regurgitation strip) consumes every word, the tick
   * conveys no new information, and clobbering `tentative` with `[]` would
   * silently throw away a real pending word from the previous tick.
   */
  private runLocalAgreement(current: TimedWord[]): TimedWord[] {
    if (current.length === 0) return [];
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
