// In-memory implementations of the repositories `LiveTranscriptionLoop`
// consumes. Used by integration tests so we never touch Dexie / IndexedDB
// and the test runs at Whisper-inference speed rather than wall-clock speed.

import type { TranscriptToken } from '../../../src/lib/db';
import type {
  AudioChunkRepository,
  CollectedAudio,
  TranscriptRepository,
} from '../../../src/lib/transcription/liveLoop';

export interface InMemoryAudioChunk {
  startedAt: number;
  samples: Float32Array;
  speechProbability: number | undefined;
}

export class InMemoryAudioChunkRepository implements AudioChunkRepository {
  private chunks: InMemoryAudioChunk[] = [];

  push(chunk: InMemoryAudioChunk): void {
    this.chunks.push(chunk);
    this.chunks.sort((a, b) => a.startedAt - b.startedAt);
  }

  size(): number {
    return this.chunks.length;
  }

  async collectFrom(startS: number): Promise<CollectedAudio | null> {
    const inWindow = this.chunks.filter((c) => c.startedAt >= startS);
    if (inWindow.length === 0) return null;
    const total = inWindow.reduce((s, c) => s + c.samples.length, 0);
    const out = new Float32Array(total);
    let offset = 0;
    let maxProb = -1;
    let allVerdicted = true;
    for (const c of inWindow) {
      out.set(c.samples, offset);
      offset += c.samples.length;
      if (c.speechProbability === undefined) allVerdicted = false;
      else if (c.speechProbability > maxProb) maxProb = c.speechProbability;
    }
    const t0 = inWindow[0].startedAt;
    const t1 = t0 + total / 16_000;
    return {
      samples: out,
      t0,
      t1,
      speechProbability: allVerdicted ? maxProb : undefined,
    };
  }

  async deleteBefore(startS: number): Promise<void> {
    this.chunks = this.chunks.filter((c) => c.startedAt >= startS);
  }

  async clear(): Promise<void> {
    this.chunks = [];
  }
}

export class InMemoryTranscriptRepository implements TranscriptRepository {
  rows: TranscriptToken[] = [];
  writeCount = 0;

  async write(rows: TranscriptToken[]): Promise<void> {
    this.writeCount++;
    this.rows = [...rows];
  }

  async clear(): Promise<void> {
    this.rows = [];
  }
}
