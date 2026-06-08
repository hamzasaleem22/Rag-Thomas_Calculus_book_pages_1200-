/* ──────────────────────────────────────────────
   useChat Hook
   Manages chat state, SSE streaming, and message operations.
   ────────────────────────────────────────────── */

import { useState, useRef, useCallback, useEffect } from "react";
import type { Message, UseChatReturn, QueryRequest, Citation } from "../types";
import { queryStream } from "../services/api";
import { generateId } from "../utils/storage";

export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  // Refs to avoid stale closures in callbacks
  const abortRef = useRef<AbortController | null>(null);
  const streamingRef = useRef(false);
  const messagesRef = useRef<Message[]>([]);

  // Keep messagesRef in sync with state
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  /* ── Internal streaming helper ───────────────── */

  function startStreaming(query: string, assistantId: string) {
    // Build conversation history from the last 10 pairs (20 messages)
    const history = messagesRef.current
      .slice(-20)
      .map((m) => ({ role: m.role, content: m.content }));

    // Ensure the current user query is the last history entry
    if (
      history.length === 0 ||
      history[history.length - 1].content !== query ||
      history[history.length - 1].role !== "user"
    ) {
      history.push({ role: "user", content: query });
    }

    setIsStreaming(true);
    streamingRef.current = true;

    const request: QueryRequest = {
      query,
      top_k: 5,
      rerank: true,
    };

    const controller = queryStream(request, history, {
      onToken: (token: string) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, content: m.content + token } : m
          )
        );
      },

      onCitations: (citations: Citation[]) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, citations } : m
          )
        );
      },

      onDone: () => {
        setIsStreaming(false);
        streamingRef.current = false;
        abortRef.current = null;
      },

      onError: (error: string) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: `Error: ${error}` }
              : m
          )
        );
        setIsStreaming(false);
        streamingRef.current = false;
        abortRef.current = null;
      },
    });

    abortRef.current = controller;
  }

  /* ── Public actions ──────────────────────────── */

  const sendMessage = useCallback((text: string) => {
    const trimmed = text.trim();
    if (!trimmed || streamingRef.current) return;

    const userMessage: Message = {
      id: generateId(),
      role: "user",
      content: trimmed,
      timestamp: Date.now(),
    };

    const assistantId = generateId();
    const assistantMessage: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      timestamp: Date.now(),
    };

    // Optimistically append both messages
    setMessages((prev) => [...prev, userMessage, assistantMessage]);

    // Manually update ref so startStreaming sees the new user message
    // (useEffect hasn't flushed yet)
    messagesRef.current = [...messagesRef.current, userMessage];

    startStreaming(trimmed, assistantId);
  }, []);

  const regenerateMessage = useCallback((messageId: string) => {
    if (streamingRef.current) return;

    const currentMessages = messagesRef.current;
    const idx = currentMessages.findIndex((m) => m.id === messageId);
    if (idx === -1) return;

    const targetMsg = currentMessages[idx];
    if (targetMsg.role !== "assistant") return;

    // The preceding message must be the user query
    const userMsg = idx > 0 ? currentMessages[idx - 1] : null;
    if (!userMsg || userMsg.role !== "user") return;

    // Remove the old assistant message
    const filtered = currentMessages.filter((m) => m.id !== messageId);

    // Create a fresh assistant placeholder
    const assistantId = generateId();
    const assistantMessage: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      timestamp: Date.now(),
    };

    setMessages([...filtered, assistantMessage]);

    // Ref gets the filtered list (history without old assistant)
    messagesRef.current = filtered;

    startStreaming(userMsg.content, assistantId);
  }, []);

  const deleteMessage = useCallback((messageId: string) => {
    if (streamingRef.current) return;

    setMessages((prev) => {
      const idx = prev.findIndex((m) => m.id === messageId);
      if (idx === -1) return prev;

      const msg = prev[idx];
      const idsToRemove = new Set<string>();
      idsToRemove.add(messageId);

      if (msg.role === "user") {
        // Also remove the paired assistant response that follows
        const next = prev[idx + 1];
        if (next && next.role === "assistant") {
          idsToRemove.add(next.id);
        }
      } else if (msg.role === "assistant") {
        // Also remove the paired user query that precedes it
        const prevMsg = prev[idx - 1];
        if (prevMsg && prevMsg.role === "user") {
          idsToRemove.add(prevMsg.id);
        }
      }

      return prev.filter((m) => !idsToRemove.has(m.id));
    });
  }, []);

  const clearChat = useCallback(() => {
    if (streamingRef.current) {
      abortRef.current?.abort();
      streamingRef.current = false;
      setIsStreaming(false);
      abortRef.current = null;
    }
    setMessages([]);
  }, []);

  const cancelStream = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    streamingRef.current = false;
    setIsStreaming(false);
  }, []);

  const loadMessages = useCallback((messages: Message[]) => {
    if (streamingRef.current) {
      abortRef.current?.abort();
      streamingRef.current = false;
      setIsStreaming(false);
      abortRef.current = null;
    }
    setMessages(messages);
    messagesRef.current = messages;
  }, []);

  return {
    messages,
    isStreaming,
    sendMessage,
    regenerateMessage,
    deleteMessage,
    clearChat,
    cancelStream,
    loadMessages,
  };
}
