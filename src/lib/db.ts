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

  constructor() {
    super('off-the-record');
    this.version(2).stores({
      chunks: '++id, startedAt',
      transcript: 'tokenId, t, isFinal',
    });
  }
}

export const db = new LiveDB();

export async function clearAll(): Promise<void> {
  await db.transaction('rw', db.chunks, db.transcript, async () => {
    await db.chunks.clear();
    await db.transcript.clear();
  });
}
