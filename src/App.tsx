import { useCallback, useEffect, useRef, useState } from 'react';
import { Mic, Square, Trash2, Cpu } from 'lucide-react';
import { Transcript } from './components/Transcript';
import { Waveform } from './components/Waveform';
import { ModelPicker } from './components/ModelPicker';
import { DEFAULT_MODEL, isValidModel, type ModelId } from './lib/audio';
import { clearAll, type TranscriptToken } from './lib/db';

type Status = 'idle' | 'loading-model' | 'ready' | 'recording' | 'stopping' | 'error';

interface ProgressEntry {
  file: string;
  loaded: number;
  total: number;
  status: string;
}

const MODEL_KEY = 'off-the-record:model';

function loadStoredModel(): ModelId {
  const stored = localStorage.getItem(MODEL_KEY);
  return isValidModel(stored) ? stored : DEFAULT_MODEL;
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
  const [modelId, setModelId] = useState<ModelId>(loadStoredModel);
  const [status, setStatus] = useState<Status>('idle');
  const [backend, setBackend] = useState<'webgpu' | 'wasm' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<Record<string, ProgressEntry>>({});
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const [displayTokens, setDisplayTokens] = useState<TranscriptToken[] | null>(null);

  const producerRef = useRef<Worker | null>(null);
  const consumerRef = useRef<Worker | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);

  const spawnConsumer = useCallback((id: ModelId) => {
    consumerRef.current?.terminate();
    const worker = new Worker(
      new URL('./workers/consumer.worker.ts', import.meta.url),
      { type: 'module' }
    );
    worker.onmessage = (e: MessageEvent) => {
      const msg = e.data;
      if (msg.type === 'ready') {
        setBackend(msg.backend);
        setStatus((s) => (s === 'loading-model' ? 'ready' : s));
        setProgress({});
      } else if (msg.type === 'progress') {
        setProgress((prev) => ({ ...prev, [msg.file]: msg }));
      } else if (msg.type === 'error') {
        setError(msg.message);
      } else if (msg.type === 'log') {
        console.log(msg.message);
      } else if (msg.type === 'display') {
        setDisplayTokens(msg.tokens);
      }
    };
    consumerRef.current = worker;
    setStatus('loading-model');
    setProgress({});
    setDisplayTokens(null);
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
    };
    producerRef.current = worker;
  }, []);

  useEffect(() => {
    spawnProducer();
    spawnConsumer(modelId);
    return () => {
      producerRef.current?.terminate();
      consumerRef.current?.terminate();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleModelChange = useCallback(
    (id: ModelId) => {
      setModelId(id);
      localStorage.setItem(MODEL_KEY, id);
      spawnConsumer(id);
    },
    [spawnConsumer]
  );

  const startRecording = useCallback(async () => {
    setError(null);
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
      // Worklet has no output to speakers; just process.

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
  }, []);

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
    setStatus('ready');
  }, []);

  const handleClear = useCallback(async () => {
    if (status === 'recording') await stopRecording();
    // Tell the consumer to reset its in-memory committed/prev state and AWAIT
    // its ack. Otherwise the next Record may tick before reset is processed,
    // and the stale committedWords get written back into the DB ("flash" of
    // the old transcript when starting a new recording).
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
    setDisplayTokens([]);
  }, [status, stopRecording]);

  const isBusy = status === 'loading-model' || status === 'stopping';
  const isRecording = status === 'recording' || status === 'stopping';
  const isStopping = status === 'stopping';

  const progressEntries = Object.values(progress).filter((p) => p.total > 0);

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <header className="shrink-0 border-b border-neutral-200 px-6 py-4 flex items-center gap-4">
        <h1 className="text-lg font-medium">Off The Record</h1>
        <div className="flex-1" />
        <div className="flex items-center gap-2 text-xs text-neutral-500">
          <Cpu className="w-3.5 h-3.5" />
          {backend ?? '…'}
        </div>
        <ModelPicker value={modelId} onChange={handleModelChange} disabled={isRecording} />
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
          title="Clear transcript"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </header>

      <main className="flex-1 min-h-0 px-6 py-6">
        <Transcript tokensOverride={displayTokens} />
      </main>

      <footer className="shrink-0 border-t border-neutral-200 px-6 py-3 space-y-2">
        <Waveform analyser={analyser} />
        {error && <div className="text-sm text-red-600">{error}</div>}
        {isBusy && (
          <div className="text-sm text-neutral-500">
            Loading {modelId}…
            {progressEntries.length > 0 && (
              <span className="ml-2 font-mono text-xs">
                {progressEntries
                  .map((p) => `${p.file} ${Math.round((p.loaded / p.total) * 100)}%`)
                  .join('  ')}
              </span>
            )}
          </div>
        )}
      </footer>
    </div>
  );
}
