import { useLiveQuery } from 'dexie-react-hooks';
import { useEffect, useRef, useState } from 'react';
import { db } from '../lib/db';
import { TARGET_SAMPLE_RATE } from '../lib/audio';

export interface ProducerStats {
  totalChunksWritten: number;
  accumSamples: number;
  running: boolean;
  sourceSampleRate: number;
  lastChunkAt: number;
}

export type ConsumerTickKind =
  | 'idle'
  | 'short-window'
  | 'silence'
  | 'hallucination'
  | 'inference'
  | 'force-slide'
  | 'error';

export interface ConsumerStats {
  tickCount: number;
  lastTickMs: number;
  lastTickAt: number;
  lastTickKind: ConsumerTickKind;
  windowDurationS: number;
  committedWordsCount: number;
  prevHypothesisCount: number;
  stableFullTicks: number;
  committedAudioStartS: number;
  processing: boolean;
  draining: boolean;
}

export interface DevPanelProps {
  producer: ProducerStats | null;
  consumer: ConsumerStats | null;
}

const TICK_KIND_ICON: Record<ConsumerTickKind, string> = {
  idle: '💤',
  'short-window': '⏳',
  silence: '🤫',
  hallucination: '👻',
  inference: '🧠',
  'force-slide': '⏭️',
  error: '⚠️',
};

function useNow(intervalMs = 250): number {
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    const id = setInterval(() => setNow(performance.now()), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);
  return now;
}

function fmtAgo(eventAt: number, now: number): string {
  if (!eventAt) return 'never';
  const s = Math.max(0, (now - eventAt) / 1000);
  if (s < 1) return `${Math.round(s * 1000)}ms ago`;
  if (s < 60) return `${s.toFixed(1)}s ago`;
  return `${Math.round(s / 60)}m ago`;
}

function useRate(eventCount: number, windowMs = 3000): number {
  // Estimate events/s by remembering recent (count, time) snapshots and
  // computing delta / delta-time over the trailing window.
  const history = useRef<Array<{ count: number; at: number }>>([]);
  useEffect(() => {
    const now = performance.now();
    history.current.push({ count: eventCount, at: now });
    const cutoff = now - windowMs;
    while (history.current.length > 1 && history.current[0].at < cutoff) {
      history.current.shift();
    }
  }, [eventCount, windowMs]);
  const h = history.current;
  if (h.length < 2) return 0;
  const first = h[0];
  const last = h[h.length - 1];
  const dt = (last.at - first.at) / 1000;
  if (dt <= 0) return 0;
  return (last.count - first.count) / dt;
}

export function DevPanel({ producer, consumer }: DevPanelProps) {
  const chunkCount = useLiveQuery(() => db.chunks.count(), [], 0);
  const tokenCount = useLiveQuery(() => db.transcript.count(), [], 0);
  const finalCount = useLiveQuery(
    () => db.transcript.where('isFinal').equals(1).count(),
    [],
    0
  );
  const volatileCount = Math.max(0, (tokenCount ?? 0) - (finalCount ?? 0));
  const now = useNow();

  const pushRate = useRate(producer?.totalChunksWritten ?? 0);
  const tickRate = useRate(consumer?.tickCount ?? 0);

  const accumSeconds = producer ? producer.accumSamples / TARGET_SAMPLE_RATE : 0;

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs font-mono">
      <Card title="💾 IDB">
        <Row label="chunks" value={`${chunkCount ?? 0}`} hint={`~${chunkCount ?? 0}s audio`} />
        <Row label="tokens" value={`${tokenCount ?? 0}`} />
        <Row label="final" value={`${finalCount ?? 0}`} />
        <Row label="volatile" value={`${volatileCount}`} />
      </Card>

      <Card title="📤 Producer">
        {producer ? (
          <>
            <Row
              label="state"
              value={producer.running ? '🔴 recording' : '⏸ idle'}
            />
            <Row label="pushed" value={`${producer.totalChunksWritten}`} hint={`${pushRate.toFixed(1)}/s`} />
            <Row
              label="accum"
              value={`${producer.accumSamples} samples`}
              hint={`${accumSeconds.toFixed(2)}s`}
            />
            <Row
              label="last push"
              value={fmtAgo(producer.lastChunkAt, now)}
            />
            <Row label="src rate" value={producer.sourceSampleRate ? `${producer.sourceSampleRate} Hz` : '?'} />
          </>
        ) : (
          <Empty />
        )}
      </Card>

      <Card title="📥 Consumer">
        {consumer ? (
          <>
            <Row
              label="state"
              value={
                consumer.draining
                  ? '🚿 draining'
                  : consumer.processing
                  ? '⚙️ processing'
                  : '⏸ idle'
              }
            />
            <Row label="ticks" value={`${consumer.tickCount}`} hint={`${tickRate.toFixed(1)}/s`} />
            <Row
              label="last tick"
              value={`${consumer.lastTickMs.toFixed(0)}ms`}
              hint={fmtAgo(consumer.lastTickAt, now)}
            />
            <Row
              label="last kind"
              value={`${TICK_KIND_ICON[consumer.lastTickKind]} ${consumer.lastTickKind}`}
            />
            <Row label="window" value={`${consumer.windowDurationS.toFixed(2)}s`} />
            <Row label="committed" value={`${consumer.committedWordsCount} words`} />
            <Row label="tentative" value={`${consumer.prevHypothesisCount} words`} />
            <Row
              label="anchor"
              value={`${consumer.committedAudioStartS.toFixed(2)}s`}
            />
          </>
        ) : (
          <Empty />
        )}
      </Card>
    </div>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded border border-neutral-200 bg-neutral-50 p-3 space-y-1">
      <div className="font-medium text-neutral-700 text-[11px] uppercase tracking-wide mb-1">
        {title}
      </div>
      {children}
    </div>
  );
}

function Row({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-neutral-500">{label}</span>
      <span className="text-neutral-900 text-right">
        {value}
        {hint && <span className="ml-2 text-neutral-400">{hint}</span>}
      </span>
    </div>
  );
}

function Empty() {
  return <div className="text-neutral-400">no data yet</div>;
}
