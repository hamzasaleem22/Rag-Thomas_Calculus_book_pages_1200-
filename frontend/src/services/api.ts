/* ──────────────────────────────────────────────
   API Client
   Thomas' Calculus RAG — Frontend
   Proxies /api/* → http://localhost:8000/*
   ────────────────────────────────────────────── */

import type { QueryRequest, QueryResponse, StreamCallbacks, Citation } from "../types";

const API_BASE = "/api";

/** Synchronous query (non-streaming fallback) */
export async function querySync(params: QueryRequest): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`Query failed: ${res.status}`);
  return res.json();
}

/** Streaming query via SSE. Returns an AbortController so caller can cancel. */
export function queryStream(
  params: QueryRequest,
  history: { role: string; content: string }[],
  callbacks: StreamCallbacks
): AbortController {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${API_BASE}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...params, history }),
        signal: controller.signal,
      });

      if (!res.ok) {
        callbacks.onError(`Stream failed: ${res.status}`);
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        callbacks.onError("No response body");
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data: ")) continue;

          const data = trimmed.slice(6);
          if (data === "[DONE]") {
            callbacks.onDone();
            return;
          }

          try {
            const parsed = JSON.parse(data);
            if (parsed.type === "token") {
              callbacks.onToken(parsed.content);
            } else if (parsed.type === "citations") {
              callbacks.onCitations(parsed.citations as Citation[]);
            } else if (parsed.type === "error") {
              callbacks.onError(parsed.content || "Unknown error");
              return;
            }
          } catch {
            // Skip malformed JSON lines
          }
        }
      }

      callbacks.onDone();
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      callbacks.onError(err instanceof Error ? err.message : "Connection failed");
    }
  })();

  return controller;
}

/** Health check */
export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
