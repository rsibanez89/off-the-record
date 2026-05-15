// Pure heuristics that classify a Whisper output or an audio frame. No I/O,
// no DB, no Whisper dependency. Used by the consumer worker to decide when to
// force-commit or skip a window.

export const SILENCE_RMS_THRESHOLD = 0.005;

const HALLUCINATION_LINE_PATTERNS: RegExp[] = [
  /^>+$/,
  /^\.+$/,
  /^[♪♫]+$/,
  /^\[.*\]$/,
  /^\(.*\)$/,
  /^thank you\.?$/i,
  /^thanks for watching\.?$/i,
  /^you$/i,
  /^bye\.?$/i,
];

export function isHallucinationLine(text: string): boolean {
  const t = text.trim();
  if (!t) return true;
  if (HALLUCINATION_LINE_PATTERNS.some((p) => p.test(t))) return true;

  // Strip every bracketed/parenthesised filler and music notation. If nothing
  // meaningful remains, treat the whole line as a hallucination. Catches
  // combinations like "(water splashing) [BLANK_AUDIO]" that don't match a
  // single pattern.
  const stripped = t
    .replace(/\[[^\]]*\]/g, '')
    .replace(/\([^)]*\)/g, '')
    .replace(/[♪♫]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  return stripped.length === 0;
}

export function isHallucinationWord(word: string): boolean {
  const t = word.trim();
  if (!t) return true;
  if (/^>+$/.test(t)) return true;
  if (/^\.+$/.test(t)) return true;
  if (/^[♪♫]+$/.test(t)) return true;
  // Stand-alone dash artefact: any run of ASCII hyphen, en-dash (U+2013), or
  // em-dash (U+2014). Codepoints used so this file has no literal long-dashes.
  for (const ch of t) {
    const c = ch.charCodeAt(0);
    if (c !== 0x2d && c !== 0x2013 && c !== 0x2014) return false;
  }
  return true;
}

/**
 * True when `text` ends with sentence-final punctuation. Used by the consumer
 * worker to decide when the audio anchor may advance: we only trim audio at
 * sentence boundaries, so Whisper retains intra-sentence context across ticks.
 */
export function isSentenceEnd(text: string): boolean {
  const t = text.trim();
  if (!t) return false;
  const last = t[t.length - 1];
  return last === '.' || last === '?' || last === '!';
}

export function rms(samples: Float32Array): number {
  if (samples.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

export function isSilent(samples: Float32Array, threshold = SILENCE_RMS_THRESHOLD): boolean {
  return rms(samples) < threshold;
}
