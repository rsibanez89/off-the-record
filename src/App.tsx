import { useCallback, useEffect, useRef, useState } from 'react';
import { Mic, Square, Trash2 } from 'lucide-react';
import { Waveform } from './components/Waveform';
import { TranscriptPanel } from './components/TranscriptPanel';
import {
  DevPanel,
  type ProducerStats,
  type ConsumerStats,
} from './components/DevPanel';
import {
  DEFAULT_BATCH_MODEL,
  DEFAULT_LIVE_MODEL,
  isValidModel,
  type ModelId,
} from './lib/audio';
import { clearAll, type TranscriptToken } from './lib/db';

type RecordStatus = 'idle' | 'loading-model' | 'ready' | 'recording' | 'stopping' | 'error';
type BatchStatus = 'idle' | 'loading-model' | 'ready' | 'transcribing' | 'done' | 'error';
type Backend = 'webgpu' | 'wasm' | null;

interface ProgressEntry {
  file: string;
  loaded: number;
  total: number;
  status: string;
}

const LIVE_MODEL_KEY = 'off-the-record:model';
const BATCH_MODEL_KEY = 'off-the-record:batch-model';

function loadStoredModel(key: string, fallback: ModelId): ModelId {
  const stored = localStorage.getItem(key);
  return isValidModel(stored) ? stored : fallback;
}

function postAndWait(worker: Worker, message: unknown, doneType: string): Promise<void> {
  return new Promise((resolve) => {
    const listener = (e: MessageEvent) => {
      if (e.data?.type === doneType) {
        worker.removeEventListener('message', listener);
        resolve();
      }
    };
    worker.addEventListener('message', listener);
    worker.postMessage(message);
  });
}

export default function App() {
  const [liveModelId, setLiveModelId] = useState<ModelId>(() =>
    loadStoredModel(LIVE_MODEL_KEY, DEFAULT_LIVE_MODEL),
  );
  const [batchModelId, setBatchModelId] = useState<ModelId>(() =>
    loadStoredModel(BATCH_MODEL_KEY, DEFAULT_BATCH_MODEL),
  );
  const [status, setStatus] = useState<RecordStatus>('idle');
  const [batchStatus, setBatchStatus] = useState<BatchStatus>('idle');
  const [liveBackend, setLiveBackend] = useState<Backend>(null);
  const [batchBackend, setBatchBackend] = useState<Backend>(null);
  const [error, setError] = useState<string | null>(null);
  const [liveProgress, setLiveProgress] = useState<Record<string, ProgressEntry>>({});
  const [batchProgress, setBatchProgress] = useState<Record<string, ProgressEntry>>({});
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const [liveTokens, setLiveTokens] = useState<TranscriptToken[] | null>(null);
  const [batchTokens, setBatchTokens] = useState<TranscriptToken[] | null>(null);
  const [batchInferenceMs, setBatchInferenceMs] = useState<number | null>(null);
  const [batchAudioSeconds, setBatchAudioSeconds] = useState<number | null>(null);
  const [producerStats, setProducerStats] = useState<ProducerStats | null>(null);
  const [consumerStats, setConsumerStats] = useState<ConsumerStats | null>(null);

  const producerRef = useRef<Worker | null>(null);
  const consumerRef = useRef<Worker | null>(null);
  const batchRef = useRef<Worker | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);
  // Session counter for the batch worker. Each Stop bumps it. Late `done`
  // messages from a stale session (e.g. user clicked Record again before the
  // previous batch finished) are ignored.
  const batchSessionRef = useRef(0);

  const spawnConsumer = useCallback((id: ModelId) => {
    consumerRef.current?.terminate();
    const worker = new Worker(
      new URL('./workers/consumer.worker.ts', import.meta.url),
      { type: 'module' }
    );
    worker.onmessage = (e: MessageEvent) => {
      const msg = e.data;
      if (msg.type === 'ready') {
        setLiveBackend(msg.backend);
        setStatus((s) => (s === 'loading-model' ? 'ready' : s));
        setLiveProgress({});
      } else if (msg.type === 'progress') {
        setLiveProgress((prev) => ({ ...prev, [msg.file]: msg }));
      } else if (msg.type === 'error') {
        setError(msg.message);
      } else if (msg.type === 'display') {
        setLiveTokens(msg.tokens);
      } else if (msg.type === 'stats') {
        setConsumerStats(msg);
      }
    };
    consumerRef.current = worker;
    setStatus('loading-model');
    setLiveProgress({});
    setLiveTokens(null);
    worker.postMessage({ type: 'init', modelId: id });
  }, []);

  const spawnBatch = useCallback((id: ModelId) => {
    batchRef.current?.terminate();
    const worker = new Worker(
      new URL('./workers/batch.worker.ts', import.meta.url),
      { type: 'module' }
    );
    worker.onmessage = (e: MessageEvent) => {
      const msg = e.data;
      if (msg.type === 'ready') {
        setBatchBackend(msg.backend);
        setBatchStatus('ready');
        setBatchProgress({});
      } else if (msg.type === 'progress') {
        setBatchProgress((prev) => ({ ...prev, [msg.file]: msg }));
      } else if (msg.type === 'error') {
        setError(msg.message);
        setBatchStatus('error');
      } else if (msg.type === 'transcribe-start') {
        if (msg.sessionId !== batchSessionRef.current) return;
        setBatchStatus('transcribing');
      } else if (msg.type === 'transcribe-done') {
        if (msg.sessionId !== batchSessionRef.current) return;
        setBatchTokens(msg.tokens);
        setBatchInferenceMs(msg.inferenceMs);
        setBatchAudioSeconds(msg.durationS);
        setBatchStatus('done');
      }
    };
    batchRef.current = worker;
    setBatchStatus('loading-model');
    setBatchProgress({});
    setBatchTokens(null);
    setBatchInferenceMs(null);
    setBatchAudioSeconds(null);
    worker.postMessage({ type: 'init', modelId: id });
  }, []);

  const spawnProducer = useCallback(() => {
    producerRef.current?.terminate();
    const worker = new Worker(
      new URL('./workers/producer.worker.ts', import.meta.url),
      { type: 'module' }
    );
    worker.onmessage = (e: MessageEvent) => {
      const msg = e.data;
      if (msg.type === 'error') setError(msg.message);
      else if (msg.type === 'stats') setProducerStats(msg);
    };
    producerRef.current = worker;
  }, []);

  useEffect(() => {
    spawnProducer();
    spawnConsumer(liveModelId);
    spawnBatch(batchModelId);
    return () => {
      producerRef.current?.terminate();
      consumerRef.current?.terminate();
      batchRef.current?.terminate();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleLiveModelChange = useCallback(
    (id: ModelId) => {
      setLiveModelId(id);
      localStorage.setItem(LIVE_MODEL_KEY, id);
      spawnConsumer(id);
    },
    [spawnConsumer]
  );

  const handleBatchModelChange = useCallback(
    (id: ModelId) => {
      setBatchModelId(id);
      localStorage.setItem(BATCH_MODEL_KEY, id);
      spawnBatch(id);
    },
    [spawnBatch]
  );

  const resetEverything = useCallback(async () => {
    // Reset the live consumer's in-memory state (committed buffer, anchor,
    // stats) AND its DB tables. Then wipe audioArchive too via clearAll.
    const worker = consumerRef.current;
    if (worker) {
      await new Promise<void>((resolve) => {
        const listener = (e: MessageEvent) => {
          if (e.data?.type === 'reset-done') {
            worker.removeEventListener('message', listener);
            resolve();
          }
        };
        worker.addEventListener('message', listener);
        worker.postMessage({ type: 'reset' });
      });
    }
    await clearAll();
    setLiveTokens([]);
    setBatchTokens(null);
    setBatchInferenceMs(null);
    setBatchAudioSeconds(null);
    batchSessionRef.current += 1; // invalidate any in-flight session
  }, []);

  const startRecording = useCallback(async () => {
    setError(null);
    // Each Record session starts fresh: wipes the previous transcripts (live
    // and batch), the audioArchive, and the consumer's in-memory buffer.
    // This keeps the live-vs-batch comparison scoped to one session.
    await resetEverything();
    if (batchStatus !== 'loading-model') setBatchStatus('ready');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;

      const ctx = new AudioContext();
      audioCtxRef.current = ctx;
      await ctx.audioWorklet.addModule('/audio-worklet.js');

      const source = ctx.createMediaStreamSource(stream);
      const an = ctx.createAnalyser();
      an.fftSize = 1024;
      source.connect(an);
      setAnalyser(an);

      const node = new AudioWorkletNode(ctx, 'capture-processor');
      workletRef.current = node;
      source.connect(node);

      const producer = producerRef.current!;
      producer.postMessage({ type: 'start', sourceSampleRate: ctx.sampleRate });

      node.port.onmessage = (e) => {
        const { samples } = e.data;
        producer.postMessage({ type: 'frame', samples }, [samples.buffer]);
      };

      setStatus('recording');
    } catch (err) {
      setError((err as Error).message);
      setStatus('error');
    }
  }, [batchStatus, resetEverything]);

  const stopRecording = useCallback(async () => {
    setStatus('stopping');
    const worklet = workletRef.current;
    if (worklet) {
      worklet.port.onmessage = null;
      worklet.disconnect();
    }
    workletRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (audioCtxRef.current) {
      await audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    setAnalyser(null);
    const producer = producerRef.current;
    if (producer) {
      await postAndWait(producer, { type: 'stop' }, 'stopped');
    }
    const consumer = consumerRef.current;
    if (consumer) {
      await postAndWait(consumer, { type: 'flush' }, 'flush-done');
    }
    // Live flush is done. Trigger batch transcription. Don't await: the user
    // can record again or read the live transcript while batch crunches.
    const batch = batchRef.current;
    if (batch && batchStatus !== 'loading-model') {
      const sessionId = ++batchSessionRef.current;
      setBatchStatus('transcribing');
      batch.postMessage({ type: 'transcribe', sessionId });
    }
    setStatus('ready');
  }, [batchStatus]);

  const handleClear = useCallback(async () => {
    if (status === 'recording') await stopRecording();
    await resetEverything();
    if (batchStatus !== 'loading-model') setBatchStatus('ready');
  }, [status, stopRecording, resetEverything, batchStatus]);

  const isBusy = status === 'loading-model' || status === 'stopping';
  const isRecording = status === 'recording' || status === 'stopping';
  const isStopping = status === 'stopping';
  const modelPickersDisabled = isRecording;

  const allProgress = Object.values({ ...liveProgress, ...batchProgress }).filter(
    (p) => p.total > 0
  );

  const liveStatusBadge: { label: string; tone: 'idle' | 'busy' | 'done' | 'error' } =
    status === 'loading-model'
      ? { label: 'loading model', tone: 'busy' }
      : status === 'recording'
      ? { label: 'recording', tone: 'busy' }
      : status === 'stopping'
      ? { label: 'finalising', tone: 'busy' }
      : status === 'error'
      ? { label: 'error', tone: 'error' }
      : (liveTokens && liveTokens.length > 0)
      ? { label: 'done', tone: 'done' }
      : { label: 'ready', tone: 'idle' };

  const batchStatusBadge: { label: string; tone: 'idle' | 'busy' | 'done' | 'error' } =
    batchStatus === 'loading-model'
      ? { label: 'loading model', tone: 'busy' }
      : batchStatus === 'transcribing'
      ? { label: 'transcribing', tone: 'busy' }
      : batchStatus === 'error'
      ? { label: 'error', tone: 'error' }
      : batchStatus === 'done'
      ? {
          label:
            batchInferenceMs != null && batchAudioSeconds != null
              ? `done · ${batchInferenceMs.toFixed(0)}ms / ${batchAudioSeconds.toFixed(1)}s`
              : 'done',
          tone: 'done',
        }
      : { label: 'ready', tone: 'idle' };

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <header className="shrink-0 border-b border-neutral-200 px-6 py-4 flex items-center gap-4">
        <h1 className="text-lg font-medium">Off The Record</h1>
        <span className="text-xs text-neutral-500">
          🟢 live (LA-2) vs 🔵 batch (one-shot) comparison
        </span>
        <div className="flex-1" />
        {isRecording ? (
          <button
            onClick={stopRecording}
            disabled={isStopping}
            className="px-4 py-2 rounded bg-neutral-900 text-white text-sm flex items-center gap-2 hover:bg-neutral-700"
          >
            <Square className="w-4 h-4" /> {isStopping ? 'Stopping…' : 'Stop'}
          </button>
        ) : (
          <button
            onClick={startRecording}
            disabled={isBusy}
            className="px-4 py-2 rounded bg-neutral-900 text-white text-sm flex items-center gap-2 hover:bg-neutral-700 disabled:opacity-50"
          >
            <Mic className="w-4 h-4" /> Record
          </button>
        )}
        <button
          onClick={handleClear}
          className="px-3 py-2 rounded border border-neutral-200 text-neutral-600 hover:bg-neutral-50"
          title="Clear transcripts"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </header>

      <main className="flex-1 min-h-0 px-6 py-6 grid grid-cols-1 md:grid-cols-2 gap-6">
        <TranscriptPanel
          icon="🟢"
          title="Live · LocalAgreement-2"
          subtitle="streams while you speak"
          tokens={liveTokens}
          emptyHint="Press Record. Live transcript streams here."
          modelId={liveModelId}
          onModelChange={handleLiveModelChange}
          modelPickerDisabled={modelPickersDisabled}
          role="live"
          backend={liveBackend}
          statusBadge={liveStatusBadge}
        />
        <TranscriptPanel
          icon="🔵"
          title="Batch · one-shot on Stop"
          subtitle="full audio in a single Whisper pass"
          tokens={batchTokens}
          emptyHint="Stop a recording to run the batch transcription."
          modelId={batchModelId}
          onModelChange={handleBatchModelChange}
          modelPickerDisabled={modelPickersDisabled}
          role="batch"
          backend={batchBackend}
          statusBadge={batchStatusBadge}
        />
      </main>

      <footer className="shrink-0 border-t border-neutral-200 px-6 py-3 space-y-3">
        <DevPanel producer={producerStats} consumer={consumerStats} />
        <Waveform analyser={analyser} />
        {error && <div className="text-sm text-red-600">{error}</div>}
        {allProgress.length > 0 && (
          <div className="text-sm text-neutral-500 font-mono text-xs">
            {allProgress
              .map((p) => `${p.file} ${Math.round((p.loaded / p.total) * 100)}%`)
              .join('  ')}
          </div>
        )}
      </footer>
    </div>
  );
}
