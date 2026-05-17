// NativeStreamingLoop drives a model that was designed for streaming
// (Moonshine v2, etc.). Architecture intentionally diverges from
// `LiveTranscriptionLoop`'s LocalAgreement-2 path:
//
//   - No overlapping windows. The anchor advances past each processed
//     window so no audio is fed to the model twice.
//   - No `HypothesisBuffer`, no tentative/committed split, no agreement
//     gate. The model's output for a window is immediately final.
//   - VAD silence gate is preserved (cheap, prevents Whisper-class
//     "thank you for watching" hallucinations on silent windows).
//   - Hallucination-line filter is preserved as a defensive measure
//     (LA-2 defers, native commits; we drop the line entirely if it
//     matches the known patterns).
//
// SOLID:
//   Single Responsibility: drive a native-streaming model. No model
//     selection, no Dexie I/O details, no UI concerns.
//   Liskov: same public surface as `LiveTranscriptionLoop` so the
//     consumer worker can hold `LiveTranscriptionLoop | NativeStreamingLoop`
//     and dispatch by interface.
//
// Edge case: words straddling chunk seams can be cut or duplicated by
// the model. With 1-second windows on a model designed for short
// context this is rare; the simpler architecture wins. If quality on
// real recordings shows seam loss, add a small overlap + text-dedup of
// the prior tail.

import { isMultilingual, streamingPolicyFor, supportsWordTimestamps, TARGET_SAMPLE_RATE, type ModelId } from '../audio';
import type { TranscriptToken } from '../db';
import type { TimedWord } from './hypothesisBuffer';
import { isHallucinationLine, isSilent, rms } from './heuristics';
import { runWhisper, type WhisperPipeline } from './whisperAdapter';
import {
  MAX_PROMPT_CHARS,
  NATIVE_STREAMING_OVERLAP_S,
  NATIVE_STREAMING_WINDOW_S,
  VAD_SILENCE_THRESHOLD,
} from '../config';
import type {
  AudioChunkRepository,
  TickKind,
  TickOutcome,
  TranscriptRepository,
} from './liveLoop';

export interface NativeStreamingLoopDeps {
  pipeline: WhisperPipeline;
  audioRepo: AudioChunkRepository;
  transcriptRepo: TranscriptRepository;
  modelId: ModelId;
}

/**
 * Word normalisation for seam-dedup comparisons. Mirrors the
 * `HypothesisBuffer.norm` private helper: lowercase, strip ASCII and
 * dash punctuation. Inlined here (rather than imported) because
 * HypothesisBuffer keeps it private; if a third call site shows up,
 * promote to a shared module.
 */
const STRIP_PUNCT_ASCII = new Set('.,!?;:"\'()[]-'.split(''));
const STRIP_PUNCT_CODEPOINTS = new Set([0x2013, 0x2014]);
function normWord(s: string): string {
  let out = '';
  for (const ch of s.trim().toLowerCase()) {
    if (STRIP_PUNCT_ASCII.has(ch)) continue;
    if (STRIP_PUNCT_CODEPOINTS.has(ch.charCodeAt(0))) continue;
    out += ch;
  }
  return out;
}

export class NativeStreamingLoop {
  private committedAudioStartS = 0;
  private committed: TimedWord[] = [];
  // Tail of the previous tick's input audio, kept in memory so the next
  // tick can prepend it as decoder context. The audio repo only stores
  // chunks at 1 s boundaries, so we cannot achieve sub-chunk overlap via
  // the anchor alone. Holding the overlap samples here gives the model
  // the bridging audio it needs without re-running entire chunks.
  private overlapSamples: Float32Array = new Float32Array(0);
  private overlapStartS = 0;

  constructor(private readonly deps: NativeStreamingLoopDeps) {
    if (streamingPolicyFor(deps.modelId) !== 'native') {
      // Soft guard: surfaces a misconfiguration at construction rather than
      // at runtime. The worker selects the loop class from the policy so
      // this should not normally fire.
      console.warn(
        `[native-loop] model ${deps.modelId} has streamingPolicy != 'native'; ` +
          `LA-2 (LiveTranscriptionLoop) is the right choice for that model.`,
      );
    }
  }

  getCommittedAudioStartS(): number {
    return this.committedAudioStartS;
  }

  getCommittedWordCount(): number {
    return this.committed.length;
  }

  /** Native streaming has no tentative state; everything is final. */
  getTentativeWordCount(): number {
    return 0;
  }

  reset(): void {
    this.committedAudioStartS = 0;
    this.committed = [];
    this.overlapSamples = new Float32Array(0);
    this.overlapStartS = 0;
  }

  async tick(opts?: { minWindowS?: number }): Promise<TickOutcome> {
    const { audioRepo, pipeline, modelId } = this.deps;
    const minWindowS = opts?.minWindowS ?? NATIVE_STREAMING_WINDOW_S;

    const audio = await audioRepo.collectFrom(this.committedAudioStartS);
    if (!audio) {
      return { kind: 'idle', windowDurationS: 0, rowsChanged: false, rows: [] };
    }
    const dur = audio.samples.length / TARGET_SAMPLE_RATE;

    // Wait until we have at least `minWindowS` seconds of audio before
    // invoking the model. Production cadence uses
    // `NATIVE_STREAMING_WINDOW_S` (2 s) because small models like
    // Moonshine 61M need enough acoustic context per call to maintain
    // coherence; firing on every 1 s producer chunk produces fragmented
    // hallucinated output. `drainAndFinalise` overrides to 0 so the
    // trailing sub-window on Stop still gets transcribed.
    if (dur < minWindowS) {
      return { kind: 'short-window', windowDurationS: dur, rowsChanged: false, rows: [] };
    }

    // Silence gate: same VAD-first, RMS-fallback policy as the LA-2 path.
    // On silence we still advance the anchor so the next tick sees fresh
    // audio; we just skip the model invocation.
    const vadVerdict = audio.speechProbability;
    const isVadSilent =
      vadVerdict !== undefined ? vadVerdict < VAD_SILENCE_THRESHOLD : isSilent(audio.samples);
    if (isVadSilent) {
      await this.advanceAnchor(audio.t1);
      const reason =
        vadVerdict !== undefined
          ? `vad=${vadVerdict.toFixed(3)}`
          : `rms=${rms(audio.samples).toFixed(4)}`;
      console.log(`[native-loop] silence dur=${dur.toFixed(2)}s ${reason}`);
      return { kind: 'silence', windowDurationS: dur, rowsChanged: false, rows: [] };
    }

    // Prepend the prior tick's tail audio so the model sees continuous
    // input across the seam. After inference we strip the overlap
    // re-emission from the head of the new output via text dedup.
    let inferenceSamples = audio.samples;
    let inferenceStartS = audio.t0;
    if (this.overlapSamples.length > 0) {
      inferenceSamples = new Float32Array(this.overlapSamples.length + audio.samples.length);
      inferenceSamples.set(this.overlapSamples, 0);
      inferenceSamples.set(audio.samples, this.overlapSamples.length);
      inferenceStartS = this.overlapStartS;
    }

    let result;
    try {
      result = await runWhisper(pipeline, inferenceSamples, {
        language: isMultilingual(modelId) ? 'en' : undefined,
        offsetSeconds: inferenceStartS,
        requestWordTimestamps: supportsWordTimestamps(modelId),
        // Feed the committed tail as decoder context. Without this each
        // 2 s window is decoded from scratch with no preceding context,
        // which on small models (Moonshine 61M) produces fragmented
        // hallucinated output. The same trick LA-2 uses; the prompt-
        // regurgitation strip in `runWhisper` cleans up the case where
        // the decoder echoes the prompt at the head of its output.
        // Soft-failure: if the pipeline's tokenizer is missing or its
        // encode shape differs, `encodePromptIds` returns [] and the
        // prompt is silently skipped.
        initialPrompt: this.buildInitialPrompt(),
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

    if (isHallucinationLine(result.text)) {
      // Drop this window entirely. Without LA-2 we cannot "defer until
      // more context arrives" the way LiveTranscriptionLoop does; the
      // safer policy is to skip the window and advance the anchor.
      await this.advanceAnchor(audio.t1);
      console.log(
        `[native-loop] hallucination on non-silent audio, dropping text="${result.text.slice(0, 80)}"`,
      );
      return { kind: 'hallucination', windowDurationS: dur, rowsChanged: false, rows: [] };
    }

    // Strip overlap re-emission: when the model re-decodes the prior
    // tail-audio prefix, it tends to emit the same words again. Dedup
    // head of new output against tail of committed text (normalised),
    // up to MAX_OVERLAP_K words. Reuses the same approach as
    // HypothesisBuffer.stripCommittedTailOverlap.
    const newWords = this.stripCommittedHeadOverlap(result.words);
    this.saveOverlapTail(audio);

    if (newWords.length === 0) {
      // All words were duplicates of the committed tail (or the model
      // returned nothing past the overlap). Advance the anchor and exit.
      await this.advanceAnchor(audio.t1);
      return { kind: 'inference', windowDurationS: dur, rowsChanged: false, rows: [] };
    }

    const firstNewIndex = this.committed.length;
    this.committed.push(...newWords);
    const rows = this.committed.map<TranscriptToken>((w, i) => ({
      tokenId: i,
      text: w.text,
      t: w.start,
      isFinal: 1,
    }));
    await this.deps.transcriptRepo.writeIncremental(rows.slice(firstNewIndex), rows.length);
    await this.advanceAnchor(audio.t1);
    console.log(
      `[native-loop] dur=${dur.toFixed(2)}s rms=${rms(audio.samples).toFixed(3)} ` +
        `+${newWords.length}/${result.words.length} words (deduped ${result.words.length - newWords.length}) ` +
        `text="${result.text.slice(0, 120)}"`,
    );
    const kind: TickKind = 'inference';
    return { kind, windowDurationS: dur, rowsChanged: true, rows };
  }

  /**
   * Strip the head of `words` that overlaps the committed tail. Greedy:
   * tries the longest k-gram first and returns words.slice(k) if it
   * matches the last k committed words after text normalisation.
   * Mirrors HypothesisBuffer.stripCommittedTailOverlap; could be hoisted
   * once a third caller appears.
   */
  private stripCommittedHeadOverlap(words: TimedWord[]): TimedWord[] {
    if (words.length === 0 || this.committed.length === 0) return words;
    const MAX_OVERLAP_K = 20;
    const maxK = Math.min(this.committed.length, words.length, MAX_OVERLAP_K);
    for (let k = maxK; k >= 1; k--) {
      const committedTail = this.committed
        .slice(this.committed.length - k)
        .map((w) => normWord(w.text))
        .join(' ');
      const newHead = words
        .slice(0, k)
        .map((w) => normWord(w.text))
        .join(' ');
      if (committedTail === newHead) return words.slice(k);
    }
    return words;
  }

  /**
   * Save the trailing `NATIVE_STREAMING_OVERLAP_S` of THIS tick's audio
   * (just the new audio, not the prepended overlap) so the next tick can
   * prepend it as decoder context. Caps at the window size in case the
   * caller fed a sub-overlap window.
   */
  private saveOverlapTail(audio: { samples: Float32Array; t0: number; t1: number }): void {
    const wantSamples = Math.floor(NATIVE_STREAMING_OVERLAP_S * TARGET_SAMPLE_RATE);
    const overlapCount = Math.min(audio.samples.length, wantSamples);
    if (overlapCount <= 0) {
      this.overlapSamples = new Float32Array(0);
      this.overlapStartS = audio.t1;
      return;
    }
    this.overlapSamples = audio.samples.slice(audio.samples.length - overlapCount);
    this.overlapStartS = audio.t1 - overlapCount / TARGET_SAMPLE_RATE;
  }

  async drainAndFinalise(): Promise<TranscriptToken[]> {
    // Process any remaining audio with one final tick; native streaming
    // does not need the multi-iteration drain that LA-2 uses to confirm
    // tentative words (there is no tentative state to confirm). Override
    // the min-window guard so the trailing sub-window (audio shorter
    // than NATIVE_STREAMING_WINDOW_S) still gets transcribed.
    while (true) {
      const outcome = await this.tick({ minWindowS: 0 });
      if (outcome.kind === 'idle' || outcome.kind === 'error') break;
      if (outcome.kind === 'short-window') break;
    }
    const rows = this.committed.map<TranscriptToken>((w, i) => ({
      tokenId: i,
      text: w.text,
      t: w.start,
      isFinal: 1,
    }));
    await this.deps.audioRepo.clear();
    this.committedAudioStartS = 0;
    return rows;
  }

  private async advanceAnchor(toS: number): Promise<void> {
    if (toS <= this.committedAudioStartS) return;
    this.committedAudioStartS = toS;
    await this.deps.audioRepo.deleteBefore(toS);
  }

  /**
   * Build a Whisper-style initial prompt from the committed tail. Walks
   * backwards from the most-recent word until either the buffer is empty
   * or the next addition would exceed `MAX_PROMPT_CHARS` (the adapter's
   * tokenizer further caps by token count). Same shape as
   * `LiveTranscriptionLoop.buildInitialPrompt`; could be hoisted into a
   * shared helper once a third caller appears.
   */
  private buildInitialPrompt(): string | undefined {
    if (this.committed.length === 0) return undefined;
    let text = '';
    for (let i = this.committed.length - 1; i >= 0; i--) {
      const sep = text ? ' ' : '';
      const next = this.committed[i].text + sep + text;
      if (next.length > MAX_PROMPT_CHARS) break;
      text = next;
    }
    const trimmed = text.trim();
    return trimmed.length > 0 ? trimmed : undefined;
  }
}
