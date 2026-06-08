import { useState, type FormEvent, type KeyboardEvent } from "react";
import GradientButton from "../ui/GradientButton";

interface ChatInputProps {
  onSend: (text: string) => void;
  disabled?: boolean;
  isStreaming?: boolean;
  onCancel?: () => void;
}

/* ── Inline SVG Icons ── */

function SendIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="ml-1.5">
      <line x1="3" y1="8" x2="13" y2="8" />
      <polyline points="9,4 13,8 9,12" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor" className="mr-1.5">
      <rect x="2" y="2" width="10" height="10" rx="2" />
    </svg>
  );
}

export default function ChatInput({
  onSend,
  disabled = false,
  isStreaming = false,
  onCancel,
}: ChatInputProps) {
  const [value, setValue] = useState("");

  const trimmed = value.trim();

  function handleSubmit(e?: FormEvent) {
    e?.preventDefault();
    if (trimmed && !disabled && !isStreaming) {
      onSend(trimmed);
      setValue("");
    }
  }

  function handleKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  return (
    <div className="glass-strong border-t border-slate-200/50 p-4 shrink-0">
      <form
        onSubmit={handleSubmit}
        className="flex gap-3 items-center max-w-3xl mx-auto"
      >
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder="Ask a question about calculus..."
          className="flex-1 rounded-2xl bg-white/60 backdrop-blur px-5 py-3.5 text-sm
            text-slate-900 placeholder:text-slate-400
            border border-slate-200/60 outline-none
            focus:border-blue-400 focus:shadow-lg focus:shadow-blue-500/10
            transition-all duration-200
            disabled:opacity-50 disabled:cursor-not-allowed"
        />

        {isStreaming ? (
          <button
            type="button"
            onClick={onCancel}
            className="flex items-center justify-center
              px-5 py-2.5 text-sm font-medium rounded-xl
              bg-red-50 text-red-600 border border-red-200
              hover:bg-red-100 hover:border-red-300
              transition-all duration-200"
          >
            <StopIcon />
            Stop
          </button>
        ) : (
          <GradientButton
            type="submit"
            disabled={!trimmed || disabled}
            className="flex items-center"
          >
            Send
            <SendIcon />
          </GradientButton>
        )}
      </form>
    </div>
  );
}
