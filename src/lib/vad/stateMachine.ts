// Frame-to-segment hysteresis state machine for VAD output. Pure, sync, no
// I/O. Consumes a stream of per-frame speech probabilities and emits
// `onSpeechStart`, `onSpeechEnd`, `onMisfire` callbacks.
//
// Standard Silero VAD hysteresis: enter SPEAKING the moment a frame crosses
// `positiveSpeechThreshold`; track redemption frames below
// `negativeSpeechThreshold`; close the segment after `redemptionMs` of
// continuous redemption; drop segments shorter than `minSpeechMs` as
// misfires. A small pre-speech ring buffer holds the last `preSpeechPadMs`
// of frame-end times so the emitted segment includes pre-speech padding.

import {
  DEFAULT_STATE_MACHINE_CONFIG,
  type FrameProbability,
  type SpeechSegment,
  type VadStateMachineConfig,
} from './types';

export interface VadStateMachineHandlers {
  onSpeechStart?: (s: SpeechSegment) => void;
  onSpeechEnd?: (s: SpeechSegment) => void;
  onMisfire?: (s: SpeechSegment) => void;
}

type State = 'silence' | 'speaking';

interface PreSpeechEntry {
  startS: number;
}

export class VadStateMachine {
  private readonly cfg: VadStateMachineConfig;
  private state: State = 'silence';
  /** Frames spent below `negativeSpeechThreshold` while in SPEAKING. */
  private redemptionFrames = 0;
  /** Maximum redemption-frame count before closing a segment. */
  private readonly redemptionLimit: number;
  /** Minimum number of frames required for a segment to graduate to "real". */
  private readonly minSpeechFrames: number;
  /** Ring buffer of recent frame start times for pre-speech padding. */
  private preSpeech: PreSpeechEntry[] = [];
  private readonly preSpeechCapacity: number;
  /** When was the current SPEAKING segment opened (padded). */
  private currentSegmentStartS = 0;
  /** End time of the last real-speech frame in the current segment. */
  private currentSegmentLastRealEndS = 0;
  /** How many frames in this segment have crossed the positive threshold. */
  private currentSegmentRealFrames = 0;

  constructor(
    cfg: Partial<VadStateMachineConfig> = {},
    private readonly handlers: VadStateMachineHandlers = {},
  ) {
    this.cfg = { ...DEFAULT_STATE_MACHINE_CONFIG, ...cfg };
    this.redemptionLimit = Math.max(1, Math.round(this.cfg.redemptionMs / this.cfg.frameMs));
    this.minSpeechFrames = Math.max(1, Math.round(this.cfg.minSpeechMs / this.cfg.frameMs));
    this.preSpeechCapacity = Math.max(0, Math.round(this.cfg.preSpeechPadMs / this.cfg.frameMs));
  }

  /**
   * Feed one frame's probability and timing. Fires zero or one handler.
   */
  ingest(frame: FrameProbability): void {
    if (this.state === 'silence') {
      if (frame.probability >= this.cfg.positiveSpeechThreshold) {
        // Open a segment. The segment start is the earliest pre-speech entry
        // (the padding before the crossing frame). If the ring is empty
        // (preSpeechPadMs === 0 or no prior frames), the segment starts at
        // this frame.
        const paddedStart =
          this.preSpeech.length > 0 ? this.preSpeech[0].startS : frame.startS;
        this.state = 'speaking';
        this.currentSegmentStartS = paddedStart;
        this.currentSegmentLastRealEndS = frame.endS;
        this.currentSegmentRealFrames = 1;
        this.redemptionFrames = 0;
        // Clear the ring so it does not bleed into the next segment.
        this.preSpeech = [];
        this.handlers.onSpeechStart?.({
          startS: this.currentSegmentStartS,
          endS: frame.endS,
        });
        return;
      }
      // Pre-speech ring: record only frames that do NOT cross threshold, so
      // padding is purely "the silence right before the crossing".
      this.preSpeech.push({ startS: frame.startS });
      if (this.preSpeech.length > this.preSpeechCapacity) {
        this.preSpeech.splice(0, this.preSpeech.length - this.preSpeechCapacity);
      }
      return;
    }

    // SPEAKING.
    if (frame.probability >= this.cfg.positiveSpeechThreshold) {
      this.redemptionFrames = 0;
      this.currentSegmentRealFrames++;
      this.currentSegmentLastRealEndS = frame.endS;
      return;
    }

    if (frame.probability < this.cfg.negativeSpeechThreshold) {
      this.redemptionFrames++;
      if (this.redemptionFrames >= this.redemptionLimit) {
        this.closeSegment(this.currentSegmentLastRealEndS);
      }
      return;
    }
    // Between negative and positive threshold: hold steady. Do not advance
    // redemption (the frame is ambiguous), do not extend "real" end.
  }

  /**
   * Force-close any open segment. Called when the engine stops or resets.
   */
  flush(): void {
    if (this.state === 'speaking') {
      this.closeSegment(this.currentSegmentLastRealEndS);
    }
  }

  reset(): void {
    this.state = 'silence';
    this.redemptionFrames = 0;
    this.preSpeech = [];
    this.currentSegmentStartS = 0;
    this.currentSegmentLastRealEndS = 0;
    this.currentSegmentRealFrames = 0;
  }

  /** Exposed for tests / instrumentation. */
  getState(): State {
    return this.state;
  }

  /** Exposed for tests: number of pre-speech entries currently buffered. */
  getPreSpeechBuffered(): number {
    return this.preSpeech.length;
  }

  private closeSegment(endS: number): void {
    const seg: SpeechSegment = {
      startS: this.currentSegmentStartS,
      endS,
    };
    const wasMisfire = this.currentSegmentRealFrames < this.minSpeechFrames;
    this.state = 'silence';
    this.redemptionFrames = 0;
    this.currentSegmentRealFrames = 0;
    this.preSpeech = [];
    if (wasMisfire) {
      this.handlers.onMisfire?.(seg);
    } else {
      this.handlers.onSpeechEnd?.(seg);
    }
  }
}
