import Dexie, { type Table } from 'dexie';

// IndexedDB cannot index boolean values, so flags are stored as 0 | 1.
export type Flag = 0 | 1;

export interface AudioChunk {
  id?: number;
  startedAt: number;
  samples: Float32Array;
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
