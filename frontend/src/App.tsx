import { useState, useEffect, useRef, useCallback } from "react";
import { useChat } from "./hooks/useChat";
import { useChatHistory } from "./hooks/useChatHistory";
import Header from "./components/layout/Header";
import ChatContainer from "./components/layout/ChatContainer";
import ChatInput from "./components/input/ChatInput";
import GlassCard from "./components/ui/GlassCard";
import type { ChatSession } from "./types";
import { formatDate } from "./utils/storage";

function App() {
  const chat = useChat();
  const history = useChatHistory();
  const [showHistory, setShowHistory] = useState(false);
  const historyRef = useRef<HTMLDivElement>(null);

  /* ── Load active session on mount ─────────────── */

  const initializedRef = useRef(false);
  useEffect(() => {
    if (initializedRef.current) return;
    if (history.sessions.length === 0 && history.activeSessionId === null) {
      // First visit ever — create initial session
      history.newSession();
      initializedRef.current = true;
      return;
    }
    if (history.activeSessionId) {
      const session = history.sessions.find((s) => s.id === history.activeSessionId);
      if (session && session.messages.length > 0) {
        chat.loadMessages(session.messages);
      } else {
        // Active session has no messages, just start fresh
        history.newSession();
      }
    } else if (history.sessions.length > 0) {
      // No active session but sessions exist — load the most recent
      history.switchSession(history.sessions[0].id);
      chat.loadMessages(history.sessions[0].messages);
    } else {
      history.newSession();
    }
    initializedRef.current = true;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [history.sessions, history.activeSessionId]);

  /* ── Auto-save messages to session ────────────── */

  useEffect(() => {
    if (!initializedRef.current) return;
    if (chat.messages.length > 0) {
      history.saveCurrentSession(chat.messages);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chat.messages]);

  /* ── Close history dropdown on outside click ──── */

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (historyRef.current && !historyRef.current.contains(e.target as Node)) {
        setShowHistory(false);
      }
    }
    if (showHistory) {
      document.addEventListener("mousedown", handleClick);
      return () => document.removeEventListener("mousedown", handleClick);
    }
  }, [showHistory]);

  /* ── New Chat handler ─────────────────────────── */

  const handleNewChat = useCallback(() => {
    // Save current session before starting new
    if (chat.messages.length > 0) {
      history.saveCurrentSession(chat.messages);
    }
    chat.clearChat();
    history.newSession();
    setShowHistory(false);
  }, [chat, history]);

  /* ── Session switch handler ───────────────────── */

  const handleSwitchSession = useCallback(
    (id: string) => {
      // Save current session
      if (chat.messages.length > 0) {
        history.saveCurrentSession(chat.messages);
      }
      // Switch
      history.switchSession(id);
      const session = history.sessions.find((s) => s.id === id);
      if (session) {
        chat.loadMessages(session.messages);
      } else {
        chat.clearChat();
      }
      setShowHistory(false);
    },
    [chat, history]
  );

  /* ── Session delete handler ───────────────────── */

  const handleDeleteSession = useCallback(
    (id: string) => {
      const wasActive = history.activeSessionId === id;
      history.deleteSession(id);

      if (wasActive) {
        chat.clearChat();
        // After delete, if sessions remain, load the first one
        const remaining = history.sessions.filter((s) => s.id !== id);
        if (remaining.length > 0) {
          history.switchSession(remaining[0].id);
          chat.loadMessages(remaining[0].messages);
        } else {
          history.newSession();
        }
      }
    },
    [chat, history]
  );

  return (
    <div className="flex flex-col h-dvh bg-slate-50 animate-fade-in">
      {/* ── Header ──────────────────────────────── */}
      <div className="relative">
        <Header
          onNewChat={handleNewChat}
          onToggleHistory={() => setShowHistory((v) => !v)}
          sessionCount={history.sessions.filter((s) => s.messages.length > 0).length}
          showHistory={showHistory}
        />

        {/* ── History Dropdown ──────────────────── */}
        {showHistory && (
          <div
            ref={historyRef}
            className="absolute top-full right-4 z-50 mt-2 w-80 animate-fade-in-down"
          >
            <GlassCard strong className="p-2 shadow-xl max-h-96 overflow-y-auto">
              <div className="px-3 py-2 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                Chat History
              </div>

              {history.sessions.length === 0 ? (
                <p className="px-3 py-4 text-sm text-slate-400 text-center">
                  No previous conversations
                </p>
              ) : (
                <div className="space-y-0.5">
                  {history.sessions.map((session: ChatSession) => (
                    <div
                      key={session.id}
                      className={`group flex items-center gap-2 px-3 py-2.5 rounded-xl cursor-pointer transition-colors ${
                        session.id === history.activeSessionId
                          ? "bg-blue-50 border border-blue-100"
                          : "hover:bg-slate-50"
                      }`}
                      onClick={() => handleSwitchSession(session.id)}
                    >
                      <div className="flex-1 min-w-0">
                        <p className="text-sm text-slate-800 truncate font-medium">
                          {session.title || "New Chat"}
                        </p>
                        <p className="text-[11px] text-slate-400 mt-0.5">
                          {session.messages.length} messages · {formatDate(session.updatedAt)}
                        </p>
                      </div>

                      {/* Delete button */}
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteSession(session.id);
                        }}
                        className="opacity-0 group-hover:opacity-100 p-1 rounded-lg
                          text-slate-400 hover:text-red-500 hover:bg-red-50
                          transition-all duration-150 shrink-0"
                        title="Delete conversation"
                      >
                        <svg
                          width="14"
                          height="14"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <polyline points="3 6 5 6 21 6" />
                          <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                        </svg>
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </GlassCard>
          </div>
        )}
      </div>

      {/* ── Chat Area ───────────────────────────── */}
      <ChatContainer
        messages={chat.messages}
        isStreaming={chat.isStreaming}
        onSend={chat.sendMessage}
        onRegenerate={chat.regenerateMessage}
        onDelete={chat.deleteMessage}
      />

      {/* ── Input Bar ───────────────────────────── */}
      <ChatInput
        onSend={chat.sendMessage}
        isStreaming={chat.isStreaming}
        onCancel={chat.cancelStream}
      />
    </div>
  );
}

export default App;
