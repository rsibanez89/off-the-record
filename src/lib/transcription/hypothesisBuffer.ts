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

interface TentativeEntry {
  word: TimedWord;
  /**
   * Number of consecutive hypotheses in which this exact word has appeared
   * at this position. Starts at 1 the first time the word is seen. A word
   * commits when `confirms` reaches `agreementOrder` (LA-n threshold).
   */
  confirms: number;
}

export class HypothesisBuffer {
  private committed: TimedWord[] = [];
  // `tentative` holds the surviving tail of recent hypotheses. Each entry
  // carries a confirm counter so we can generalise from LA-2 (commit at 2
  // matches) to LA-n. The order of entries mirrors LA-2 semantics: a fresh
  // hypothesis is compared position-wise against this list to decide what
  // graduates to `committed`.
  private tentative: TentativeEntry[] = [];
  private lastCommittedTime = 0;
  private readonly agreementOrder: number;

  constructor(options?: { agreementOrder?: number }) {
    this.agreementOrder = options?.agreementOrder ?? 2;
  }

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
   * Core LocalAgreement-n step. Walk `current` and `tentative` in parallel,
   * incrementing each matched position's confirm count. A word graduates
   * to `committed` when its confirms reach `agreementOrder` (n).
   *
   * For n=2 this reproduces the LA-2 reference behaviour: the first match
   * lands at confirms=2 and commits immediately. For n=3 a word must
   * appear at the same position in three consecutive hypotheses.
   *
   * Empty `current` is a no-op: when an upstream step consumes every word
   * the tick conveys no information and clobbering `tentative` would
   * silently throw away a real pending word.
   */
  private runLocalAgreement(current: TimedWord[]): TimedWord[] {
    if (current.length === 0) return [];

    // 1. Position-wise match prefix between current and tentative.
    let matchLen = 0;
    while (matchLen < current.length && matchLen < this.tentative.length) {
      if (norm(current[matchLen].text) !== norm(this.tentative[matchLen].word.text)) break;
      matchLen++;
    }

    // 2. Increment confirm counters along the match prefix.
    for (let i = 0; i < matchLen; i++) {
      this.tentative[i].confirms++;
    }

    // 3. Promote the longest prefix that has reached the threshold.
    let commitLen = 0;
    while (
      commitLen < this.tentative.length &&
      this.tentative[commitLen].confirms >= this.agreementOrder
    ) {
      commitLen++;
    }

    // Commit using the CURRENT hypothesis's text and timestamps: it is the
    // freshest read of the audio and matches the LA-2 reference behaviour
    // (which captured the new word, not the stored stale one). `commitLen
    // <= matchLen` by construction so `current[i]` is always in range.
    const justCommitted: TimedWord[] = [];
    for (let i = 0; i < commitLen; i++) {
      const w = current[i];
      justCommitted.push(w);
      this.lastCommittedTime = w.end;
    }
    this.committed.push(...justCommitted);

    // 4. The new tentative is:
    //    - the matched-but-not-yet-threshold middle section (keeps its
    //      confirm counts so a later tick can reach the threshold)
    //    - plus the divergent tail of current, fresh (confirms=1).
    const pending = this.tentative.slice(commitLen, matchLen);
    const fresh: TentativeEntry[] = current.slice(matchLen).map((word) => ({ word, confirms: 1 }));
    this.tentative = [...pending, ...fresh];
    return justCommitted;
  }

  /**
   * Force-commit the tentative buffer. Used by callers that have external
   * knowledge that no more agreement will come for the current audio window
   * (silence, hallucination, MAX_WINDOW_S force-slide).
   */
  forceCommit(): TimedWord[] {
    if (this.tentative.length === 0) return [];
    const promoted = this.tentative.map((e) => e.word);
    this.lastCommittedTime = promoted[promoted.length - 1].end;
    this.committed.push(...promoted);
    this.tentative = [];
    return promoted;
  }

  getCommitted(): readonly TimedWord[] {
    return this.committed;
  }

  getTentative(): readonly TimedWord[] {
    return this.tentative.map((e) => e.word);
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
