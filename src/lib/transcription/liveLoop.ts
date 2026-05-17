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
import { isHallucinationLine, isSentenceEnd, isSilent, rms } from './heuristics';
import { runWhisper, type WhisperPipeline } from './whisperAdapter';
import { VAD_SILENCE_THRESHOLD } from '../config';

// Window-size policy (see PLAN.md "Live transcription design invariants").
// `MIN_WINDOW_S = 0` means "run on the first chunk available" so the live
// panel gives feedback fast; LA-2 revises junk on subsequent ticks.
const MIN_WINDOW_S = 0;
const MIN_DRAIN_WINDOW_S = 0;
const MAX_WINDOW_S = 24.0;
// Audio anchor advancement: preserve intra-sentence context for accuracy.
// Trim only on sentence ends, except force-trim past FAST_TRIM_THRESHOLD_S
// and force-slide past MAX_WINDOW_S as safety nets.
const CONTEXT_LOOKBACK_S = 5.0;
const FAST_TRIM_THRESHOLD_S = 10.0;
// Drain loop safety: cap ticks at Stop so a misbehaving model cannot stall
// the flush forever.
const DRAIN_MAX_ITERATIONS = 20;
// Prompt length cap (characters as a cheap proxy for Whisper's ~224-token
// prompt window; the adapter further caps by token count).
const MAX_PROMPT_CHARS = 800;

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
  write(rows: TranscriptToken[]): Promise<void>;
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
}

export class LiveTranscriptionLoop {
  private committedAudioStartS = 0;

  constructor(private readonly deps: LiveLoopDeps) {}

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

    // Anchor advancement (conservative).
    const sentenceEnd = findLatestSentenceEnd(buffer.getCommitted());
    if (sentenceEnd > 0) {
      const target = sentenceEnd - CONTEXT_LOOKBACK_S;
      if (target > this.committedAudioStartS) await this.advanceAnchor(target);
    } else if (dur > FAST_TRIM_THRESHOLD_S) {
      const target = buffer.getLastCommittedTime() - CONTEXT_LOOKBACK_S;
      if (target > this.committedAudioStartS) await this.advanceAnchor(target);
    }

    // Force-slide safety net.
    if (dur > MAX_WINDOW_S) {
      const forced = buffer.forceCommit();
      const finalRows = await this.writeTranscript();
      await this.advanceAnchor(audio.t1 - CONTEXT_LOOKBACK_S);
      console.log(
        `[live-loop] force-slide dur=${dur.toFixed(2)}s force-committed=${forced.length}`,
      );
      return { kind: 'force-slide', windowDurationS: dur, rowsChanged: true, rows: finalRows };
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
    const rows: TranscriptToken[] = [];
    for (const w of this.deps.buffer.getCommitted()) {
      rows.push({ tokenId: rows.length, text: w.text, t: w.start, isFinal: 1 });
    }
    for (const w of this.deps.buffer.getTentative()) {
      rows.push({ tokenId: rows.length, text: w.text, t: w.start, isFinal: 0 });
    }
    await this.deps.transcriptRepo.write(rows);
    return rows;
  }

  private async advanceAnchor(toS: number): Promise<void> {
    if (toS <= this.committedAudioStartS) return;
    this.committedAudioStartS = toS;
    await this.deps.audioRepo.deleteBefore(toS);
  }
}

function findLatestSentenceEnd(committed: readonly TimedWord[]): number {
  for (let i = committed.length - 1; i >= 0; i--) {
    if (isSentenceEnd(committed[i].text)) return committed[i].end;
  }
  return 0;
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
