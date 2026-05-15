import { useLiveQuery } from 'dexie-react-hooks';
import { db, type TranscriptToken } from '../lib/db';
import { useEffect, useRef } from 'react';

interface TranscriptProps {
  tokensOverride?: TranscriptToken[] | null;
}

export const Transcript = ({ tokensOverride }: TranscriptProps) => {
  const storedTokens = useLiveQuery(() => db.transcript.orderBy('tokenId').toArray(), [], []);
  const tokens = tokensOverride ?? storedTokens;
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickyRef = useRef(true);

  // Track whether the user is parked at the bottom. If they scroll up to read,
  // we stop auto-scrolling until they return.
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
    <div
      ref={scrollRef}
      className="h-full overflow-y-auto rounded border border-neutral-200 bg-neutral-50 p-4 leading-relaxed text-xl"
    >
      {!tokens || tokens.length === 0 ? (
        <div className="text-neutral-400 italic">
          Press Record. The transcript will appear here.
        </div>
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
  );
};
