// Dexie-backed implementations of the repository interfaces consumed by
// `LiveTranscriptionLoop`. The worker uses these; integration tests inject
// in-memory implementations instead.
//
// Kept thin on purpose: the Dexie schema and table choice (chunks vs
// audioArchive vs transcript) is defined in `src/lib/db.ts`; we just wrap
// the live tables behind the interfaces the loop needs.

import { db, type TranscriptToken } from '../db';
import { TARGET_SAMPLE_RATE } from '../audio';
import type {
  AudioChunkRepository,
  CollectedAudio,
  TranscriptRepository,
} from '../transcription/liveLoop';

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

  async clear(): Promise<void> {
    await db.transcript.clear();
  }
}
