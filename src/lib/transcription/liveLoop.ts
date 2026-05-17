// LiveTranscriptionLoop is the pure, testable core of the live consumer
// worker. It owns the LocalAgreement-2 streaming algorithm and the audio
// anchor / silence-gate / hallucination policy, but takes its Whisper pipeline
// and storage as injected dependencies.
//
// The worker (`src/workers/consumer.worker.ts`) is now a thin shell around
// this class: it owns pipeline creation, message dispatch, timer pacing, and
// the `BroadcastChannel` wake-up. Anything algorithmic lives here, so we can
// drive it from Vitest with an in-memory pipeline and in-memory repositories
// for integration tests (improvement 2.3 in the workspace improvements backlog).

import { isMultilingual, TARGET_SAMPLE_RATE, type ModelId } from '../audio';
import type { TranscriptToken } from '../db';
import { HypothesisBuffer, type TimedWord } from './hypothesisBuffer';
import { isHallucinationLine, isSilent, rms } from './heuristics';
import { runWhisper, type WhisperPipeline } from './whisperAdapter';
import {
  AnchorPolicy,
  SentenceBoundaryAnchorPolicy,
} from './anchorPolicy';
import {
  DRAIN_MAX_ITERATIONS,
  MAX_PROMPT_CHARS,
  MIN_DRAIN_WINDOW_S,
  MIN_WINDOW_S,
  VAD_SILENCE_THRESHOLD,
} from '../config';

export interface CollectedAudio {
  samples: Float32Array;
  t0: number;
  t1: number;
  /**
   * Maximum `speechProbability` across all chunks in this window. `undefined`
   * when we cannot trust a VAD verdict for this window (any chunk lacked a
   * verdict). The caller falls back to RMS in that case. See the silence
   * gate path in `tick`.
   */
  speechProbability: number | undefined;
}

export interface AudioChunkRepository {
  collectFrom(startS: number): Promise<CollectedAudio | null>;
  deleteBefore(startS: number): Promise<void>;
  clear(): Promise<void>;
}

export interface TranscriptRepository {
  /** Full rewrite: replace stored rows with `rows`. Used at session reset. */
  write(rows: TranscriptToken[]): Promise<void>;
  /**
   * Incremental update: bulkPut `diff` (rows whose `tokenId` already
   * indicates the target slot) and trim anything at or beyond `totalCount`.
   * Used per inference tick so the repo does not rewrite 5,000+
   * already-stable committed rows on every tick of a long session.
   * Implementations must remain compatible with `write` semantics: after
   * `writeIncremental(diff, totalCount)` the underlying store contains
   * exactly the rows whose `tokenId < totalCount`, with `diff` taking
   * precedence at its positions.
   */
  writeIncremental(diff: TranscriptToken[], totalCount: number): Promise<void>;
  clear(): Promise<void>;
}

export type TickKind =
  | 'idle'
  | 'short-window'
  | 'silence'
  | 'hallucination'
  | 'inference'
  | 'force-slide'
  | 'error';

export interface TickOutcome {
  kind: TickKind;
  windowDurationS: number;
  /** When true, the transcript was rewritten this tick and `rows` is the new state. */
  rowsChanged: boolean;
  /** Latest transcript rows; empty when `rowsChanged` is false. */
  rows: TranscriptToken[];
  errorMessage?: string;
}

export interface LiveLoopDeps {
  pipeline: WhisperPipeline;
  buffer: HypothesisBuffer;
  audioRepo: AudioChunkRepository;
  transcriptRepo: TranscriptRepository;
  modelId: ModelId;
  /**
   * Optional anchor advancement policy. Defaults to the conservative
   * sentence-boundary policy that matches the original inline behaviour.
   * Injectable so tests and future variants (e.g. AlignAtt) can swap it
   * without touching the loop.
   */
  anchorPolicy?: AnchorPolicy;
}

export class LiveTranscriptionLoop {
  private committedAudioStartS = 0;
  private readonly anchorPolicy: AnchorPolicy;
  // Diff-only writeTranscript state (3.5): rows at indices [0, prevCommittedCount)
  // are already stored and stable across ticks (LA-2 invariant: committed rows
  // never revert). Each tick only emits rows from prevCommittedCount onward
  // (the newly-promoted committed words plus the entire current tentative
  // tail), and `writeIncremental` trims anything past the new totalCount.
  private prevCommittedCount = 0;

  constructor(private readonly deps: LiveLoopDeps) {
    this.anchorPolicy = deps.anchorPolicy ?? new SentenceBoundaryAnchorPolicy();
  }

  getCommittedAudioStartS(): number {
    return this.committedAudioStartS;
  }

  getCommittedWordCount(): number {
    return this.deps.buffer.getCommitted().length;
  }

  getTentativeWordCount(): number {
    return this.deps.buffer.getTentative().length;
  }

  /** Reset in-memory state (anchor + buffer). Does NOT clear repositories. */
  reset(): void {
    this.committedAudioStartS = 0;
    this.prevCommittedCount = 0;
    this.deps.buffer.reset();
  }

  /**
   * Run one inference tick. Mirrors the production cadence: one chunk written
   * by the producer triggers one wake-up which triggers (at most) one `tick`.
   * `minDur` defaults to `MIN_WINDOW_S`; drain passes `MIN_DRAIN_WINDOW_S`.
   */
  async tick(minDur: number = MIN_WINDOW_S): Promise<TickOutcome> {
    const { audioRepo, buffer, pipeline, modelId } = this.deps;

    const audio = await audioRepo.collectFrom(this.committedAudioStartS);
    if (!audio) {
      return { kind: 'idle', windowDurationS: 0, rowsChanged: false, rows: [] };
    }
    const dur = audio.samples.length / TARGET_SAMPLE_RATE;
    if (dur < minDur) {
      return { kind: 'short-window', windowDurationS: dur, rowsChanged: false, rows: [] };
    }

    // Silence gate: VAD first, RMS as fallback. See PLAN.md "Live
    // transcription design invariants". One undefined-VAD chunk forces RMS
    // for the whole window so unverdicted speech cannot be silently dropped.
    const vadVerdict = audio.speechProbability;
    const isVadSilent =
      vadVerdict !== undefined ? vadVerdict < VAD_SILENCE_THRESHOLD : isSilent(audio.samples);
    if (isVadSilent) {
      const forced = buffer.forceCommit();
      const rows = await this.writeTranscript();
      await this.advanceAnchor(audio.t1);
      const reason =
        vadVerdict !== undefined
          ? `vad=${vadVerdict.toFixed(3)}`
          : `rms=${rms(audio.samples).toFixed(4)}`;
      console.log(
        `[live-loop] silence dur=${dur.toFixed(2)}s ${reason} force-committed=${forced.length}`,
      );
      return { kind: 'silence', windowDurationS: dur, rowsChanged: true, rows };
    }

    let result;
    try {
      result = await runWhisper(pipeline, audio.samples, {
        language: isMultilingual(modelId) ? 'en' : undefined,
        offsetSeconds: audio.t0,
        initialPrompt: buildInitialPrompt(buffer.getCommitted()),
      });
    } catch (err) {
      return {
        kind: 'error',
        windowDurationS: dur,
        rowsChanged: false,
        rows: [],
        errorMessage: `inference failed: ${(err as Error).message}`,
      };
    }

    console.log(
      `[live-loop] dur=${dur.toFixed(2)}s rms=${rms(audio.samples).toFixed(3)} text="${result.text.slice(0, 120)}"`,
    );

    if (isHallucinationLine(result.text)) {
      // Already past the silence gate so this audio has real energy. A
      // hallucinated line on real audio is a model failure, not empty input.
      // Don't advance the anchor and don't force-commit; let the window grow
      // so the next inference has more context. MAX_WINDOW_S is the safety
      // net if the pattern persists.
      console.log(
        `[live-loop] hallucination on non-silent audio, deferring text="${result.text.slice(0, 80)}"`,
      );
      return { kind: 'hallucination', windowDurationS: dur, rowsChanged: false, rows: [] };
    }

    const justCommitted = buffer.ingest(result.words);
    const rows = await this.writeTranscript();
    console.log(
      `[live-loop] LA2 just-committed=${justCommitted.length} ` +
        `committed=${buffer.getCommitted().length} tentative=${buffer.getTentative().length}`,
    );

    const decision = this.anchorPolicy.decide({
      committed: buffer.getCommitted(),
      tentative: buffer.getTentative(),
      windowDurationS: dur,
      windowEndS: audio.t1,
      currentAnchorS: this.committedAudioStartS,
      lastCommittedEndS: buffer.getLastCommittedTime(),
    });

    if (decision.forceCommit) {
      const forced = buffer.forceCommit();
      const finalRows = await this.writeTranscript();
      if (decision.newAnchorS != null) {
        await this.advanceAnchor(decision.newAnchorS);
      }
      console.log(
        `[live-loop] force-slide dur=${dur.toFixed(2)}s force-committed=${forced.length}`,
      );
      return { kind: 'force-slide', windowDurationS: dur, rowsChanged: true, rows: finalRows };
    }

    if (decision.newAnchorS != null) {
      await this.advanceAnchor(decision.newAnchorS);
    }

    return { kind: 'inference', windowDurationS: dur, rowsChanged: true, rows };
  }

  /**
   * End-of-session drain: run ticks back-to-back at the drain window size
   * until the loop settles, then force-commit any remaining tentative,
   * clear chunks, and reset the anchor. Used by the worker on `flush` and
   * by integration tests at the end of a clip.
   */
  async drainAndFinalise(): Promise<TranscriptToken[]> {
    for (let i = 0; i < DRAIN_MAX_ITERATIONS; i++) {
      const beforeAnchor = this.committedAudioStartS;
      const beforeCommittedLen = this.deps.buffer.getCommitted().length;
      const beforeTentativeLen = this.deps.buffer.getTentative().length;
      const outcome = await this.tick(MIN_DRAIN_WINDOW_S);
      if (outcome.kind === 'idle' || outcome.kind === 'error') break;
      const settled =
        this.committedAudioStartS === beforeAnchor &&
        this.deps.buffer.getCommitted().length === beforeCommittedLen &&
        this.deps.buffer.getTentative().length === beforeTentativeLen;
      if (settled) break;
    }
    this.deps.buffer.forceCommit();
    const rows = await this.writeTranscript();
    await this.deps.audioRepo.clear();
    this.committedAudioStartS = 0;
    return rows;
  }

  private async writeTranscript(): Promise<TranscriptToken[]> {
    const committed = this.deps.buffer.getCommitted();
    const tentative = this.deps.buffer.getTentative();
    const rows: TranscriptToken[] = [];
    for (const w of committed) {
      rows.push({ tokenId: rows.length, text: w.text, t: w.start, isFinal: 1 });
    }
    for (const w of tentative) {
      rows.push({ tokenId: rows.length, text: w.text, t: w.start, isFinal: 0 });
    }

    // Diff write: rows at [0, prevCommittedCount) are stable LA-2 commits
    // already stored. Each tick only ships:
    //   - newly-committed rows: [prevCommittedCount, committed.length)
    //   - the entire current tentative tail (it changes every tick)
    //   - a trim to rows.length so any older tentative beyond the boundary
    //     is removed.
    const diff = rows.slice(this.prevCommittedCount);
    await this.deps.transcriptRepo.writeIncremental(diff, rows.length);
    this.prevCommittedCount = committed.length;
    return rows;
  }

  private async advanceAnchor(toS: number): Promise<void> {
    if (toS <= this.committedAudioStartS) return;
    this.committedAudioStartS = toS;
    await this.deps.audioRepo.deleteBefore(toS);
  }
}

function buildInitialPrompt(committed: readonly TimedWord[]): string | undefined {
  if (committed.length === 0) return undefined;
  let text = '';
  for (let i = committed.length - 1; i >= 0; i--) {
    const sep = text ? ' ' : '';
    const next = committed[i].text + sep + text;
    if (next.length > MAX_PROMPT_CHARS) break;
    text = next;
  }
  const trimmed = text.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}
