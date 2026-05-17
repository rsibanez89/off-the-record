import Dexie, { type Table } from 'dexie';

// IndexedDB cannot index boolean values, so flags are stored as 0 | 1.
export type Flag = 0 | 1;

export interface AudioChunk {
  id?: number;
  startedAt: number;
  samples: Float32Array;
  /**
   * Maximum Silero v5 speech probability across the ~31 VAD frames in this
   * 1-second chunk, in `[0, 1]`. `undefined` when VAD was not yet loaded at
   * write time (allowed for the first half-second of a session). The
   * consumer uses this to decide whether to skip Whisper.
   *
   * Added in Dexie schema v4. Not indexed.
   */
  speechProbability?: number;
}

export interface TranscriptToken {
  tokenId: number;
  text: string;
  t: number;
  isFinal: Flag;
}

class LiveDB extends Dexie {
  chunks!: Table<AudioChunk, number>;
  transcript!: Table<TranscriptToken, number>;
  // `audioArchive` is the full recording. The producer writes every chunk to
  // both `chunks` (live, evicted as the live consumer's anchor advances) and
  // `audioArchive` (kept until Clear or next Record). The batch worker reads
  // `audioArchive` on Stop to run a one-shot Whisper pass for comparison.
  audioArchive!: Table<AudioChunk, number>;

  constructor() {
    super('off-the-record');
    this.version(2).stores({
      chunks: '++id, startedAt',
      transcript: 'tokenId, t, isFinal',
    });
    this.version(3).stores({
      chunks: '++id, startedAt',
      transcript: 'tokenId, t, isFinal',
      audioArchive: '++id, startedAt',
    });
    // v4 adds the optional `speechProbability` field on `AudioChunk` rows
    // (both `chunks` and `audioArchive` share the shape). The field is NOT
    // indexed; only `startedAt` is. The store schema strings stay the same,
    // so this is a no-op migration as far as Dexie is concerned. We still
    // bump the version so existing browsers re-open the DB cleanly and the
    // intent is recorded in source.
    this.version(4).stores({
      chunks: '++id, startedAt',
      transcript: 'tokenId, t, isFinal',
      audioArchive: '++id, startedAt',
    });
  }
}

export const db = new LiveDB();

export async function clearAll(): Promise<void> {
  await db.transaction('rw', db.chunks, db.transcript, db.audioArchive, async () => {
    await db.chunks.clear();
    await db.transcript.clear();
    await db.audioArchive.clear();
  });
}

export async function clearLiveOnly(): Promise<void> {
  // Used when starting a new recording: wipe live state, keep audioArchive
  // intact only if you want to preserve it. We currently wipe both on new
  // Record (see main thread).
  await db.transaction('rw', db.chunks, db.transcript, async () => {
    await db.chunks.clear();
    await db.transcript.clear();
  });
}
