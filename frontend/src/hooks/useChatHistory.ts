/* ──────────────────────────────────────────────
   useChatHistory Hook
   Manages chat session list and active session.
   Persists to localStorage via storage utils.
   ────────────────────────────────────────────── */

import { useState, useCallback, useEffect, useRef } from "react";
import type { ChatSession, Message, UseChatHistoryReturn } from "../types";
import {
  loadSessions,
  saveSessions,
  getActiveSessionId,
  setActiveSessionId,
  createSession,
  updateSession,
  deleteSessionById,
} from "../utils/storage";

export function useChatHistory(): UseChatHistoryReturn {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveId] = useState<string | null>(null);

  // Refs for current values to avoid stale closures in callbacks
  const sessionsRef = useRef<ChatSession[]>([]);
  const activeIdRef = useRef<string | null>(null);
  const loadedRef = useRef(false);

  useEffect(() => {
    sessionsRef.current = sessions;
  }, [sessions]);

  useEffect(() => {
    activeIdRef.current = activeSessionId;
  }, [activeSessionId]);

  /* ── Load sessions on mount ──────────────────── */

  useEffect(() => {
    const loaded = loadSessions();
    setSessions(loaded);

    const activeId = getActiveSessionId();
    if (activeId && loaded.find((s) => s.id === activeId)) {
      setActiveId(activeId);
    }

    // Mark as loaded so the persist effect doesn't overwrite
    // with the initial empty state before this runs
    loadedRef.current = true;
  }, []);

  /* ── Persist sessions whenever they change ────── */

  useEffect(() => {
    if (!loadedRef.current) return;
    saveSessions(sessions);
  }, [sessions]);

  /* ── Actions ─────────────────────────────────── */

  const newSession = useCallback(() => {
    const session = createSession();
    setSessions((prev) => [session, ...prev]);
    setActiveSessionId(session.id);
    setActiveId(session.id);
  }, []);

  const switchSession = useCallback((id: string) => {
    const exists = sessionsRef.current.find((s) => s.id === id);
    if (!exists) return;

    setActiveSessionId(id);
    setActiveId(id);
  }, []);

  const deleteSession = useCallback((id: string) => {
    const updated = deleteSessionById(sessionsRef.current, id);
    setSessions(updated);
    saveSessions(updated);

    // If the deleted session was active, switch to the next available
    if (activeIdRef.current === id) {
      const nextId = updated.length > 0 ? updated[0].id : null;
      setActiveSessionId(nextId);
      setActiveId(nextId);
    }
  }, []);

  const saveCurrentSession = useCallback((messages: Message[]) => {
    const currentId = activeIdRef.current;
    if (!currentId || messages.length === 0) return;

    setSessions((prev) => {
      // Only update if the session exists
      const exists = prev.find((s) => s.id === currentId);
      if (!exists) return prev;
      return updateSession(prev, currentId, messages);
    });
  }, []);

  return {
    sessions,
    activeSessionId,
    newSession,
    switchSession,
    deleteSession,
    saveCurrentSession,
  };
}
