// Dexie-backed implementations of the repository interfaces consumed by
// `LiveTranscriptionLoop`, the producer worker, and the batch worker.
// Workers use these; integration tests inject in-memory implementations
// for the loop, and the producer's and batch's Dexie paths run in browser
// only.
//
// Kept thin on purpose: the Dexie schema and table choice (chunks vs
// audioArchive vs transcript) is defined in `src/lib/db.ts`; we just wrap
// the live tables behind the interfaces consumers need. SOLID DIP:
// producer/batch workers depend on the interface, not on Dexie.

import { db, type TranscriptToken } from '../db';
import { TARGET_SAMPLE_RATE } from '../audio';
import type {
  AudioChunkRepository,
  CollectedAudio,
  TranscriptRepository,
} from '../transcription/liveLoop';

/**
 * Atomic publisher used by the producer worker. Writes each 1-second
 * chunk to both `db.chunks` (live, evictable by the consumer's anchor)
 * and `db.audioArchive` (kept for the batch worker) in one transaction.
 * Keeping the two writes atomic prevents an orphan chunk in one table.
 */
export interface ChunkPublisher {
  publish(chunk: {
    startedAt: number;
    samples: Float32Array;
    speechProbability: number | undefined;
  }): Promise<void>;
}

/**
 * Reader used by the batch worker. Returns the concatenated `audioArchive`
 * samples in one Float32Array (or null if empty), and exposes `clear` so
 * the worker can drop the archive at session boundaries.
 */
export interface AudioArchiveRepository {
  toFloat32(): Promise<Float32Array | null>;
  clear(): Promise<void>;
}

export class DexieAudioChunkRepository implements AudioChunkRepository {
  async collectFrom(startS: number): Promise<CollectedAudio | null> {
    const chunks = await db.chunks.where('startedAt').aboveOrEqual(startS).sortBy('startedAt');
    if (chunks.length === 0) return null;
    const total = chunks.reduce((s, c) => s + c.samples.length, 0);
    const out = new Float32Array(total);
    let offset = 0;
    // `maxProb` is sentinel-initialised to -1 so the first verdicted chunk
    // always wins. When `allVerdicted` ends up true we are guaranteed to have
    // seen at least one chunk with a probability in [0, 1].
    let maxProb = -1;
    let allVerdicted = true;
    for (const c of chunks) {
      out.set(c.samples, offset);
      offset += c.samples.length;
      if (c.speechProbability === undefined) {
        // A single undefined chunk poisons the whole window: max() over the
        // verdicted subset would mask real speech in the undefined one. Force
        // the caller onto the RMS fallback instead of a confident wrong
        // answer.
        allVerdicted = false;
      } else if (c.speechProbability > maxProb) {
        maxProb = c.speechProbability;
      }
    }
    const t0 = chunks[0].startedAt;
    const t1 = t0 + total / TARGET_SAMPLE_RATE;
    return {
      samples: out,
      t0,
      t1,
      speechProbability: allVerdicted ? maxProb : undefined,
    };
  }

  async deleteBefore(startS: number): Promise<void> {
    await db.chunks.where('startedAt').below(startS).delete();
  }

  async clear(): Promise<void> {
    await db.chunks.clear();
  }
}

export class DexieChunkPublisher implements ChunkPublisher {
  async publish(chunk: {
    startedAt: number;
    samples: Float32Array;
    speechProbability: number | undefined;
  }): Promise<void> {
    await db.transaction('rw', db.chunks, db.audioArchive, async () => {
      await db.chunks.add(chunk);
      await db.audioArchive.add(chunk);
    });
  }
}

export class DexieAudioArchiveRepository implements AudioArchiveRepository {
  async toFloat32(): Promise<Float32Array | null> {
    const chunks = await db.audioArchive.orderBy('startedAt').toArray();
    if (chunks.length === 0) return null;
    const total = chunks.reduce((s, c) => s + c.samples.length, 0);
    const out = new Float32Array(total);
    let offset = 0;
    for (const c of chunks) {
      out.set(c.samples, offset);
      offset += c.samples.length;
    }
    return out;
  }

  async clear(): Promise<void> {
    await db.audioArchive.clear();
  }
}

export class DexieTranscriptRepository implements TranscriptRepository {
  async write(rows: TranscriptToken[]): Promise<void> {
    // Rewrite the transcript table from in-memory state. bulkPut new rows
    // first, then delete only the tail beyond the new length, so Dexie live
    // queries do not flash through an empty table on every inference tick.
    await db.transaction('rw', db.transcript, async () => {
      if (rows.length === 0) {
        await db.transcript.clear();
      } else {
        await db.transcript.bulkPut(rows);
        await db.transcript.where('tokenId').aboveOrEqual(rows.length).delete();
      }
    });
  }

  async writeIncremental(diff: TranscriptToken[], totalCount: number): Promise<void> {
    // bulkPut overwrites by `tokenId`, so rows in `diff` land exactly at
    // their target slots without touching the stable prefix. Trim anything
    // beyond `totalCount` so a shrinking tentative does not leave a stale
    // tail. Single transaction prevents UI live-queries from observing a
    // half-applied state.
    await db.transaction('rw', db.transcript, async () => {
      if (totalCount === 0) {
        await db.transcript.clear();
        return;
      }
      if (diff.length > 0) {
        await db.transcript.bulkPut(diff);
      }
      await db.transcript.where('tokenId').aboveOrEqual(totalCount).delete();
    });
  }

  async clear(): Promise<void> {
    await db.transcript.clear();
  }
}
