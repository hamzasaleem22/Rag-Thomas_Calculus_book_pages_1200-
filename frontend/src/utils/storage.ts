/* ──────────────────────────────────────────────
   LocalStorage Utilities
   Handles chat session persistence with auto-pruning
   ────────────────────────────────────────────── */

import type { ChatSession, Message } from "../types";

const STORAGE_KEY = "ragbook_sessions";
const ACTIVE_SESSION_KEY = "ragbook_active_session";
const MAX_SESSIONS = 20;

/** Generate a UUID v4 */
export function generateId(): string {
  return crypto.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

/** Read all sessions from localStorage */
export function loadSessions(): ChatSession[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as ChatSession[];
    // Sort by updatedAt descending (most recent first)
    return parsed.sort((a, b) => b.updatedAt - a.updatedAt);
  } catch {
    console.warn("[storage] Failed to load sessions, resetting.");
    return [];
  }
}

/** Write all sessions to localStorage (with auto-prune) */
export function saveSessions(sessions: ChatSession[]): void {
  try {
    // Prune oldest sessions beyond MAX_SESSIONS
    const pruned = sessions.slice(0, MAX_SESSIONS);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(pruned));
  } catch (e) {
    console.warn("[storage] Failed to save sessions:", e);
  }
}

/** Get the active session ID */
export function getActiveSessionId(): string | null {
  return localStorage.getItem(ACTIVE_SESSION_KEY);
}

/** Set the active session ID */
export function setActiveSessionId(id: string | null): void {
  if (id) {
    localStorage.setItem(ACTIVE_SESSION_KEY, id);
  } else {
    localStorage.removeItem(ACTIVE_SESSION_KEY);
  }
}

/** Create a new session */
export function createSession(messages: Message[] = []): ChatSession {
  const now = Date.now();
  const firstUserMsg = messages.find((m) => m.role === "user");
  const title = firstUserMsg
    ? firstUserMsg.content.slice(0, 50) + (firstUserMsg.content.length > 50 ? "..." : "")
    : "New Chat";

  return {
    id: generateId(),
    title,
    messages,
    createdAt: now,
    updatedAt: now,
  };
}

/** Update a session's messages and title */
export function updateSession(
  sessions: ChatSession[],
  sessionId: string,
  messages: Message[]
): ChatSession[] {
  return sessions.map((s) => {
    if (s.id !== sessionId) return s;

    const firstUserMsg = messages.find((m) => m.role === "user");
    const title = firstUserMsg
      ? firstUserMsg.content.slice(0, 50) + (firstUserMsg.content.length > 50 ? "..." : "")
      : s.title;

    return {
      ...s,
      messages,
      title,
      updatedAt: Date.now(),
    };
  });
}

/** Delete a session by ID */
export function deleteSessionById(sessions: ChatSession[], sessionId: string): ChatSession[] {
  return sessions.filter((s) => s.id !== sessionId);
}

/** Format a timestamp for display */
export function formatDate(timestamp: number): string {
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;

  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}
