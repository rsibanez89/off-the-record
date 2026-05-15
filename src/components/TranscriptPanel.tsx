import { useEffect, useRef } from 'react';
import { Cpu } from 'lucide-react';
import { ModelPicker } from './ModelPicker';
import { type ModelId } from '../lib/audio';
import { type TranscriptToken } from '../lib/db';

export interface TranscriptPanelProps {
  // Display
  icon: string;
  title: string;
  subtitle?: string;
  tokens: TranscriptToken[] | null;
  emptyHint: string;

  // Model selection
  modelId: ModelId;
  onModelChange: (id: ModelId) => void;
  modelPickerDisabled?: boolean;

  // Backend and status
  backend: 'webgpu' | 'wasm' | null;
  statusBadge?: { label: string; tone: 'idle' | 'busy' | 'done' | 'error' };
}

const STATUS_TONE: Record<NonNullable<TranscriptPanelProps['statusBadge']>['tone'], string> = {
  idle: 'bg-neutral-100 text-neutral-500',
  busy: 'bg-amber-100 text-amber-800',
  done: 'bg-emerald-100 text-emerald-800',
  error: 'bg-red-100 text-red-800',
};

export function TranscriptPanel({
  icon,
  title,
  subtitle,
  tokens,
  emptyHint,
  modelId,
  onModelChange,
  modelPickerDisabled,
  backend,
  statusBadge,
}: TranscriptPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickyRef = useRef(true);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onScroll = () => {
      const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
      stickyRef.current = distance < 40;
    };
    el.addEventListener('scroll', onScroll);
    return () => el.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el || !stickyRef.current) return;
    el.scrollTop = el.scrollHeight;
  }, [tokens?.length]);

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="shrink-0 flex items-center gap-3 mb-3">
        <div className="flex items-center gap-2">
          <span className="text-lg">{icon}</span>
          <div>
            <div className="text-sm font-medium text-neutral-800">{title}</div>
            {subtitle && <div className="text-xs text-neutral-500">{subtitle}</div>}
          </div>
        </div>
        <div className="flex-1" />
        {statusBadge && (
          <span
            className={`text-[11px] uppercase tracking-wide font-medium px-2 py-1 rounded ${STATUS_TONE[statusBadge.tone]}`}
          >
            {statusBadge.label}
          </span>
        )}
        <div className="flex items-center gap-1 text-xs text-neutral-500">
          <Cpu className="w-3 h-3" />
          {backend ?? '…'}
        </div>
        <ModelPicker value={modelId} onChange={onModelChange} disabled={modelPickerDisabled} />
      </div>

      <div
        ref={scrollRef}
        className="flex-1 min-h-0 overflow-y-auto rounded border border-neutral-200 bg-neutral-50 p-4 leading-relaxed text-lg"
      >
        {!tokens || tokens.length === 0 ? (
          <div className="text-neutral-400 italic">{emptyHint}</div>
        ) : (
          tokens.map((tok) => (
            <span
              key={tok.tokenId}
              className={tok.isFinal ? 'text-neutral-900' : 'text-neutral-400'}
            >
              {tok.text}{' '}
            </span>
          ))
        )}
      </div>
    </div>
  );
}
