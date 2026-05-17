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
import { NATIVE_STREAMING_WINDOW_S, VAD_SILENCE_THRESHOLD } from '../config';
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

export class NativeStreamingLoop {
  private committedAudioStartS = 0;
  private committed: TimedWord[] = [];

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

    let result;
    try {
      result = await runWhisper(pipeline, audio.samples, {
        language: isMultilingual(modelId) ? 'en' : undefined,
        offsetSeconds: audio.t0,
        requestWordTimestamps: supportsWordTimestamps(modelId),
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

    if (result.words.length === 0) {
      // Model returned no words for a non-silent window (rare). Treat as
      // idle progress: advance the anchor so we do not re-process the
      // same window forever.
      await this.advanceAnchor(audio.t1);
      return { kind: 'inference', windowDurationS: dur, rowsChanged: false, rows: [] };
    }

    this.committed.push(...result.words);
    const rows = this.committed.map<TranscriptToken>((w, i) => ({
      tokenId: i,
      text: w.text,
      t: w.start,
      isFinal: 1,
    }));
    await this.deps.transcriptRepo.writeIncremental(
      rows.slice(rows.length - result.words.length),
      rows.length,
    );
    await this.advanceAnchor(audio.t1);
    console.log(
      `[native-loop] dur=${dur.toFixed(2)}s rms=${rms(audio.samples).toFixed(3)} +${result.words.length} words text="${result.text.slice(0, 120)}"`,
    );
    const kind: TickKind = 'inference';
    return { kind, windowDurationS: dur, rowsChanged: true, rows };
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
}
