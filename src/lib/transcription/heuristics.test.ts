import { describe, expect, it } from 'vitest';
import {
  SILENCE_RMS_THRESHOLD,
  isHallucinationLine,
  isHallucinationWord,
  isSentenceEnd,
  isSilent,
  rms,
} from './heuristics';

// Long-dash codepoints are referenced numerically here so this file has no
// literal U+2013 / U+2014 characters (project rule).
const EN_DASH = String.fromCharCode(0x2013);
const EM_DASH = String.fromCharCode(0x2014);

describe('isHallucinationLine', () => {
  it('returns true for empty and whitespace-only input', () => {
    expect(isHallucinationLine('')).toBe(true);
    expect(isHallucinationLine('   ')).toBe(true);
    expect(isHallucinationLine('\n\t ')).toBe(true);
  });

  it('matches the chevron / dot / music-note patterns', () => {
    expect(isHallucinationLine('>')).toBe(true);
    expect(isHallucinationLine('>>>')).toBe(true);
    expect(isHallucinationLine('....')).toBe(true);
    expect(isHallucinationLine('♪')).toBe(true);
    expect(isHallucinationLine('♪♫')).toBe(true);
  });

  it('matches whole-line bracket and parenthesis fillers', () => {
    expect(isHallucinationLine('[BLANK_AUDIO]')).toBe(true);
    expect(isHallucinationLine('(music playing)')).toBe(true);
  });

  it('matches the well-known Whisper filler phrases case-insensitively', () => {
    expect(isHallucinationLine('thank you')).toBe(true);
    expect(isHallucinationLine('Thank you.')).toBe(true);
    expect(isHallucinationLine('THANK YOU')).toBe(true);
    expect(isHallucinationLine('thanks for watching')).toBe(true);
    expect(isHallucinationLine('Thanks for watching.')).toBe(true);
    expect(isHallucinationLine('you')).toBe(true);
    expect(isHallucinationLine('bye')).toBe(true);
    expect(isHallucinationLine('Bye.')).toBe(true);
  });

  it('treats combined bracket / music fillers as a hallucination after stripping', () => {
    // Catches lines where no single pattern matches but every meaningful char
    // is inside brackets or is a music notation.
    expect(isHallucinationLine('(water splashing) [BLANK_AUDIO]')).toBe(true);
    expect(isHallucinationLine('[BLANK_AUDIO] ♪')).toBe(true);
    expect(isHallucinationLine('[a] (b) ♪♫')).toBe(true);
  });

  it('passes real speech through', () => {
    expect(isHallucinationLine('hello world')).toBe(false);
    expect(isHallucinationLine('thanks for the help')).toBe(false);
    // Bracketed filler embedded in real speech survives the strip.
    expect(isHallucinationLine('hello [pause] world')).toBe(false);
  });
});

describe('isHallucinationWord', () => {
  it('returns true for empty / whitespace', () => {
    expect(isHallucinationWord('')).toBe(true);
    expect(isHallucinationWord('   ')).toBe(true);
  });

  it('matches the chevron / dot / music-note runs', () => {
    expect(isHallucinationWord('>')).toBe(true);
    expect(isHallucinationWord('>>')).toBe(true);
    expect(isHallucinationWord('.')).toBe(true);
    expect(isHallucinationWord('....')).toBe(true);
    expect(isHallucinationWord('♪')).toBe(true);
    expect(isHallucinationWord('♫')).toBe(true);
    expect(isHallucinationWord('♪♫')).toBe(true);
  });

  it('matches stand-alone dash runs across hyphen, en-dash, and em-dash codepoints', () => {
    expect(isHallucinationWord('-')).toBe(true);
    expect(isHallucinationWord('--')).toBe(true);
    expect(isHallucinationWord(EN_DASH)).toBe(true);
    expect(isHallucinationWord(EM_DASH)).toBe(true);
    expect(isHallucinationWord(EN_DASH + EM_DASH + '-')).toBe(true);
  });

  it('passes real words through, even when they contain a chevron or punctuation', () => {
    expect(isHallucinationWord('hello')).toBe(false);
    expect(isHallucinationWord('>hello')).toBe(false);
    expect(isHallucinationWord('don’t')).toBe(false);
    // Mixed dash-and-letter run is real content (e.g. an em-dash followed by a
    // word fragment Whisper concatenated).
    expect(isHallucinationWord(EM_DASH + 'okay')).toBe(false);
  });
});

describe('isSentenceEnd', () => {
  it('returns true for the three sentence-final terminators', () => {
    expect(isSentenceEnd('hello.')).toBe(true);
    expect(isSentenceEnd('hello?')).toBe(true);
    expect(isSentenceEnd('hello!')).toBe(true);
  });

  it('tolerates trailing whitespace', () => {
    expect(isSentenceEnd('hello.   ')).toBe(true);
    expect(isSentenceEnd('  hello?\n')).toBe(true);
  });

  it('returns false for non-terminating punctuation', () => {
    expect(isSentenceEnd('hello')).toBe(false);
    expect(isSentenceEnd('hello,')).toBe(false);
    expect(isSentenceEnd('hello:')).toBe(false);
    expect(isSentenceEnd('hello;')).toBe(false);
  });

  it('returns false for empty / whitespace input', () => {
    expect(isSentenceEnd('')).toBe(false);
    expect(isSentenceEnd('   ')).toBe(false);
  });
});

describe('rms', () => {
  it('returns 0 for an empty buffer', () => {
    expect(rms(new Float32Array(0))).toBe(0);
  });

  it('returns 0 for all-zero samples', () => {
    expect(rms(new Float32Array(1024))).toBe(0);
  });

  it('returns the constant magnitude for a DC signal', () => {
    const samples = new Float32Array(1024).fill(0.5);
    expect(rms(samples)).toBeCloseTo(0.5, 6);
  });

  it('returns A / sqrt(2) for a full-cycle sine of amplitude A', () => {
    const A = 0.8;
    const N = 16_000;
    const samples = new Float32Array(N);
    for (let i = 0; i < N; i++) samples[i] = A * Math.sin((2 * Math.PI * i) / N);
    expect(rms(samples)).toBeCloseTo(A / Math.SQRT2, 3);
  });
});

describe('isSilent', () => {
  it('uses SILENCE_RMS_THRESHOLD by default', () => {
    // Slightly below the default threshold.
    const quiet = new Float32Array(1024).fill(SILENCE_RMS_THRESHOLD * 0.5);
    expect(isSilent(quiet)).toBe(true);

    // Above the default threshold.
    const loud = new Float32Array(1024).fill(SILENCE_RMS_THRESHOLD * 5);
    expect(isSilent(loud)).toBe(false);
  });

  it('respects a custom threshold override', () => {
    const samples = new Float32Array(1024).fill(0.1);
    expect(isSilent(samples, 0.05)).toBe(false);
    expect(isSilent(samples, 0.2)).toBe(true);
  });
});
