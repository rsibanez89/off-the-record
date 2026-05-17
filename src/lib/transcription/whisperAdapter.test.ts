import { describe, expect, it, vi } from 'vitest';
import { runWhisper, type WhisperPipeline } from './whisperAdapter';

/**
 * Build a mocked `WhisperPipeline` that returns a transformers.js-shaped
 * result. The pipeline is a callable function with extra properties; we
 * cast through `unknown` because the real pipeline object also has
 * processor/model fields we do not need here.
 */
function mockPipeline(result: unknown, opts?: { tokenizer?: WhisperPipeline['tokenizer'] }): {
  pipeline: WhisperPipeline;
  call: ReturnType<typeof vi.fn>;
} {
  const call = vi.fn(async () => result);
  const pipeline = call as unknown as WhisperPipeline;
  pipeline.tokenizer = opts?.tokenizer ?? {
    encode: () => [], // empty: prompt_ids never gets set in callOpts
  };
  return { pipeline, call };
}

const audio = new Float32Array(16_000); // 1 s of silence; never inspected

describe('runWhisper', () => {
  it('returns parsed words when no initial prompt is given', async () => {
    const { pipeline } = mockPipeline({
      text: ' hello world',
      chunks: [
        { text: ' hello', timestamp: [0, 0.5] },
        { text: ' world', timestamp: [0.5, 1] },
      ],
    });
    const result = await runWhisper(pipeline, audio, { offsetSeconds: 0 });
    expect(result.words.map((w) => w.text)).toEqual(['hello', 'world']);
  });

  it('strips a leading prompt-regurgitation from the parsed words', async () => {
    // Whisper produced the full prompt at the head of its output, then
    // appended one new word. The adapter must strip the prompt.
    const { pipeline } = mockPipeline({
      text: ' Okay, now I am trying to see if passing a prompt to the model',
      chunks: [
        { text: ' Okay,', timestamp: [0, 0.3] },
        { text: ' now', timestamp: [0.3, 0.5] },
        { text: ' I', timestamp: [0.5, 0.7] },
        { text: ' am', timestamp: [0.7, 0.9] },
        { text: ' trying', timestamp: [0.9, 1.2] },
        { text: ' to', timestamp: [1.2, 1.4] },
        { text: ' see', timestamp: [1.4, 1.6] },
        { text: ' if', timestamp: [1.6, 1.8] },
        { text: ' passing', timestamp: [1.8, 2.2] },
        { text: ' a', timestamp: [2.2, 2.3] },
        { text: ' prompt', timestamp: [2.3, 2.6] },
        { text: ' to', timestamp: [2.6, 2.8] },
        { text: ' the', timestamp: [2.8, 3.0] },
        { text: ' model', timestamp: [3.0, 3.4] }, // the one genuinely-new word
      ],
    });
    const result = await runWhisper(pipeline, audio, {
      offsetSeconds: 0,
      initialPrompt: 'Okay, now I am trying to see if passing a prompt to the',
    });
    expect(result.words.map((w) => w.text)).toEqual(['model']);
  });

  it('does not strip when the prompt does not match the head', async () => {
    // Whisper output starts with content that does NOT match the prompt;
    // nothing should be stripped even though a prompt was provided.
    const { pipeline } = mockPipeline({
      text: ' new content here',
      chunks: [
        { text: ' new', timestamp: [0, 0.3] },
        { text: ' content', timestamp: [0.3, 0.7] },
        { text: ' here', timestamp: [0.7, 1.0] },
      ],
    });
    const result = await runWhisper(pipeline, audio, {
      offsetSeconds: 0,
      initialPrompt: 'unrelated previous context',
    });
    expect(result.words.map((w) => w.text)).toEqual(['new', 'content', 'here']);
  });

  it('does not strip on a tiny coincidental match (under the 3-word threshold)', async () => {
    // Single-word match should NOT trigger stripping; "the" is a common
    // prompt/output word and stripping it would lose real content.
    const { pipeline } = mockPipeline({
      text: ' the model is way',
      chunks: [
        { text: ' the', timestamp: [0, 0.2] },
        { text: ' model', timestamp: [0.2, 0.5] },
        { text: ' is', timestamp: [0.5, 0.7] },
        { text: ' way', timestamp: [0.7, 1.0] },
      ],
    });
    const result = await runWhisper(pipeline, audio, {
      offsetSeconds: 0,
      initialPrompt: 'the quick brown fox',
    });
    expect(result.words.map((w) => w.text)).toEqual(['the', 'model', 'is', 'way']);
  });

  it('strip is case- and punctuation-insensitive', async () => {
    const { pipeline } = mockPipeline({
      text: ' Hello, world. new',
      chunks: [
        { text: ' Hello,', timestamp: [0, 0.5] },
        { text: ' world.', timestamp: [0.5, 1.0] },
        { text: ' here.', timestamp: [1.0, 1.4] },
        { text: ' new', timestamp: [1.4, 1.7] },
      ],
    });
    const result = await runWhisper(pipeline, audio, {
      offsetSeconds: 0,
      initialPrompt: 'hello WORLD here',
    });
    expect(result.words.map((w) => w.text)).toEqual(['new']);
  });

  it('encodes a prompt to prompt_ids and forwards them through callOpts', async () => {
    // Capture what callOpts the adapter actually sends to the pipeline.
    const tokenizer = {
      encode: vi.fn(() => [101, 102, 103]),
    };
    const { pipeline, call } = mockPipeline(
      { text: ' x', chunks: [{ text: ' x', timestamp: [0, 0.1] }] },
      { tokenizer },
    );
    await runWhisper(pipeline, audio, {
      offsetSeconds: 0,
      initialPrompt: 'context tail',
    });
    expect(tokenizer.encode).toHaveBeenCalledWith('context tail', { add_special_tokens: false });
    expect(call).toHaveBeenCalledTimes(1);
    const callOpts = (call.mock.calls[0] as unknown as unknown[])[1] as Record<string, unknown>;
    expect(callOpts.prompt_ids).toEqual([101, 102, 103]);
  });

  it('skips prompt_ids when the tokenizer is unavailable', async () => {
    // Pipeline without a tokenizer should not crash; just skip the prompt
    // path silently. We still might strip on the output, but without a
    // prompt the strip is a no-op.
    const call = vi.fn(async () => ({
      text: ' a b',
      chunks: [
        { text: ' a', timestamp: [0, 0.2] },
        { text: ' b', timestamp: [0.2, 0.4] },
      ],
    }));
    const pipeline = call as unknown as WhisperPipeline;
    pipeline.tokenizer = undefined;
    const result = await runWhisper(pipeline, audio, {
      offsetSeconds: 0,
      initialPrompt: 'x y z',
    });
    expect(result.words.map((w) => w.text)).toEqual(['a', 'b']);
    const callOpts = (call.mock.calls[0] as unknown as unknown[])[1] as Record<string, unknown>;
    expect(callOpts.prompt_ids).toBeUndefined();
  });
});

describe('parseResult (via runWhisper)', () => {
  it('unwraps an array-shaped pipeline result by taking the first element', async () => {
    // transformers.js sometimes returns the run result as a one-element array.
    const { pipeline } = mockPipeline([
      {
        text: ' a b',
        chunks: [
          { text: ' a', timestamp: [0, 0.2] },
          { text: ' b', timestamp: [0.2, 0.4] },
        ],
      },
    ]);
    const result = await runWhisper(pipeline, audio, { offsetSeconds: 0 });
    expect(result.text).toBe(' a b');
    expect(result.words.map((w) => w.text)).toEqual(['a', 'b']);
  });

  it('returns empty text and no words for a null pipeline result', async () => {
    const { pipeline } = mockPipeline(null);
    const result = await runWhisper(pipeline, audio, { offsetSeconds: 0 });
    expect(result.text).toBe('');
    expect(result.words).toEqual([]);
  });

  it('synthesises evenly-distributed words from text when chunks is missing', async () => {
    // Moonshine via transformers.js returns `{ text }` with no `chunks`
    // array. The fallback path tokenises on whitespace and distributes
    // timestamps across the audio duration so the batch panel can still
    // render a transcript. These synthesised entries are inaccurate
    // enough that the model is filtered from the live picker via
    // `supportsWordTimestamps: false`, but the batch path stays useful.
    const { pipeline } = mockPipeline({ text: 'hello world from moonshine' });
    const result = await runWhisper(pipeline, audio, { offsetSeconds: 0 });
    expect(result.text).toBe('hello world from moonshine');
    expect(result.words.map((w) => w.text)).toEqual(['hello', 'world', 'from', 'moonshine']);
    // Timestamps are monotonic and span the audio duration.
    const audioDurationS = audio.length / 16_000;
    expect(result.words[0].start).toBe(0);
    expect(result.words[result.words.length - 1].end).toBeCloseTo(audioDurationS, 5);
    for (let i = 1; i < result.words.length; i++) {
      expect(result.words[i].start).toBeGreaterThanOrEqual(result.words[i - 1].end - 1e-9);
    }
  });

  it('returns empty words when both chunks and text are empty', async () => {
    const { pipeline } = mockPipeline({ text: '', chunks: [] });
    expect((await runWhisper(pipeline, audio, { offsetSeconds: 0 })).words).toEqual([]);
  });

  it('drops chunks whose start OR end timestamp is null', async () => {
    // Whisper emits nulls for word timestamps near window edges. LA-2 cannot
    // safely commit those words, so the adapter drops them.
    const { pipeline } = mockPipeline({
      text: ' a b c d',
      chunks: [
        { text: ' a', timestamp: [0, 0.2] }, // kept
        { text: ' b', timestamp: [null, 0.4] }, // dropped (start null)
        { text: ' c', timestamp: [0.4, null] }, // dropped (end null)
        { text: ' d', timestamp: [null, null] }, // dropped (both null)
      ],
    });
    const result = await runWhisper(pipeline, audio, { offsetSeconds: 0 });
    expect(result.words.map((w) => w.text)).toEqual(['a']);
  });

  it('drops chunks with empty or whitespace-only text', async () => {
    const { pipeline } = mockPipeline({
      text: ' a b',
      chunks: [
        { text: ' a', timestamp: [0, 0.2] },
        { text: '   ', timestamp: [0.2, 0.4] },
        { text: '', timestamp: [0.4, 0.6] },
        { text: ' b', timestamp: [0.6, 0.8] },
      ],
    });
    const result = await runWhisper(pipeline, audio, { offsetSeconds: 0 });
    expect(result.words.map((w) => w.text)).toEqual(['a', 'b']);
  });

  it('applies isHallucinationWord to drop standalone artefacts', async () => {
    // ">>" is a common tiny.en hallucination word. Adapter must strip it
    // before LA-2 sees it (otherwise it pollutes the committed buffer).
    const { pipeline } = mockPipeline({
      text: ' >> hello',
      chunks: [
        { text: ' >>', timestamp: [0, 0.1] },
        { text: ' hello', timestamp: [0.1, 0.5] },
      ],
    });
    const result = await runWhisper(pipeline, audio, { offsetSeconds: 0 });
    expect(result.words.map((w) => w.text)).toEqual(['hello']);
  });

  it('adds offsetSeconds to every word start and end timestamp', async () => {
    const { pipeline } = mockPipeline({
      text: ' a b',
      chunks: [
        { text: ' a', timestamp: [0.1, 0.3] },
        { text: ' b', timestamp: [0.4, 0.6] },
      ],
    });
    const result = await runWhisper(pipeline, audio, { offsetSeconds: 10 });
    expect(result.words).toEqual([
      { text: 'a', start: 10.1, end: 10.3 },
      { text: 'b', start: 10.4, end: 10.6 },
    ]);
  });

  it('drops chunks whose timestamp field is missing entirely', async () => {
    const { pipeline } = mockPipeline({
      text: ' a b',
      chunks: [
        { text: ' a', timestamp: [0, 0.2] },
        { text: ' b' }, // no `timestamp` at all
      ],
    });
    const result = await runWhisper(pipeline, audio, { offsetSeconds: 0 });
    expect(result.words.map((w) => w.text)).toEqual(['a']);
  });
});
