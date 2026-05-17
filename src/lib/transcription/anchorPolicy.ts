// AnchorPolicy decides when and where to advance the audio anchor that
// marks the left edge of the audio window still being transcribed by the
// live loop. Pulled out of `LiveTranscriptionLoop.tick` so that anchor
// strategies are swappable: today's conservative sentence-boundary policy
// lives here; future AlignAtt or aggressive-per-word policies can plug in
// without touching the loop.
//
// SOLID:
//   Single Responsibility: anchor advancement only. Whisper inference, LA-2
//     buffer mutation, and silence gating stay in the loop.
//   Open/Closed: alternate policies implement the same `decide` shape.
//   No I/O: pure function over the immediate tick state. The loop is
//     responsible for applying the decision (advanceAnchor, forceCommit,
//     writeTranscript).

import type { TimedWord } from './hypothesisBuffer';
import { isSentenceEnd } from './heuristics';
import {
  CONTEXT_LOOKBACK_S,
  FAST_TRIM_THRESHOLD_S,
  MAX_WINDOW_S,
} from '../config';

export interface AnchorDecisionState {
  /** Words committed so far across the session. */
  committed: readonly TimedWord[];
  /** Words still pending LA-2 confirmation. */
  tentative: readonly TimedWord[];
  /** Duration in seconds of the current inference window. */
  windowDurationS: number;
  /** Absolute end time of the current inference window. */
  windowEndS: number;
  /** Current anchor: left edge of the audio still being processed. */
  currentAnchorS: number;
  /** End time of the last committed word (0 if no commits yet). */
  lastCommittedEndS: number;
}

export interface AnchorDecision {
  /**
   * Where the anchor should be after this tick. `null` means "no change";
   * the loop must NOT call `advanceAnchor`.
   */
  newAnchorS: number | null;
  /**
   * When true, the loop must force-commit any tentative buffer before
   * advancing the anchor and tag this tick as a force-slide outcome.
   */
  forceCommit: boolean;
}

export interface AnchorPolicy {
  decide(state: AnchorDecisionState): AnchorDecision;
}

/**
 * The default policy: trim only on a committed sentence boundary, with two
 * safety nets. Matches the original inline behaviour in
 * `LiveTranscriptionLoop.tick` so this extraction is bench-neutral.
 *
 *   - Natural trim: if the latest committed word ends a sentence, slide the
 *     anchor to `sentenceEnd - CONTEXT_LOOKBACK_S`. Keep `CONTEXT_LOOKBACK_S`
 *     of context for Whisper.
 *   - FAST_TRIM_THRESHOLD_S: if the window has grown past this without a
 *     sentence end, trim to `lastCommittedEnd - CONTEXT_LOOKBACK_S` so
 *     inference does not fall behind real-time on long unpunctuated speech.
 *   - MAX_WINDOW_S: if the window has grown past this without a natural
 *     commit, force-commit tentative and trim to `windowEnd - LOOKBACK`.
 */
export class SentenceBoundaryAnchorPolicy implements AnchorPolicy {
  decide(state: AnchorDecisionState): AnchorDecision {
    // Force-slide safety net wins if it fires; the loop has to do the
    // force-commit before applying the new anchor.
    if (state.windowDurationS > MAX_WINDOW_S) {
      return {
        newAnchorS: state.windowEndS - CONTEXT_LOOKBACK_S,
        forceCommit: true,
      };
    }

    const sentenceEnd = findLatestSentenceEnd(state.committed);
    if (sentenceEnd > 0) {
      const target = sentenceEnd - CONTEXT_LOOKBACK_S;
      if (target > state.currentAnchorS) {
        return { newAnchorS: target, forceCommit: false };
      }
    } else if (state.windowDurationS > FAST_TRIM_THRESHOLD_S) {
      const target = state.lastCommittedEndS - CONTEXT_LOOKBACK_S;
      if (target > state.currentAnchorS) {
        return { newAnchorS: target, forceCommit: false };
      }
    }

    return { newAnchorS: null, forceCommit: false };
  }
}

function findLatestSentenceEnd(committed: readonly TimedWord[]): number {
  for (let i = committed.length - 1; i >= 0; i--) {
    if (isSentenceEnd(committed[i].text)) return committed[i].end;
  }
  return 0;
}
