// Model fetcher + integrity verification for the Silero VAD ONNX file.
//
// DIP: callers depend on `ModelFetcher`, not on `fetch`. Tests pass a fake
// fetcher that returns a synthetic buffer; production uses `httpModelFetcher`
// which fetches from `public/models/`.

import type { ModelFetcher } from '../types';

export class HttpModelFetcher implements ModelFetcher {
  async fetch(url: string): Promise<ArrayBuffer> {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`HttpModelFetcher: GET ${url} failed: ${res.status} ${res.statusText}`);
    }
    return res.arrayBuffer();
  }
}

/**
 * SubtleCrypto-based SHA-256 hex digest. Available in browsers, web workers,
 * and modern Node. Returns lowercase hex.
 */
export async function sha256Hex(buf: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', buf);
  const bytes = new Uint8Array(digest);
  let hex = '';
  for (let i = 0; i < bytes.length; i++) {
    const b = bytes[i].toString(16);
    hex += b.length === 1 ? '0' + b : b;
  }
  return hex;
}

/** Optional: verify a buffer matches an expected SHA-256. Throws on mismatch. */
export async function verifyHash(buf: ArrayBuffer, expectedHex: string): Promise<void> {
  const actual = await sha256Hex(buf);
  if (actual !== expectedHex) {
    throw new Error(
      `Silero model hash mismatch. expected=${expectedHex} actual=${actual}`,
    );
  }
}
