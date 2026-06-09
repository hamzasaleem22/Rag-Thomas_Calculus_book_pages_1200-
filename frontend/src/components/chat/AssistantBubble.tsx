import type { Message } from "../../types";
import GlassCard from "../ui/GlassCard";
import AnswerSummary from "../widgets/AnswerSummary";
import KeyPointsList from "../widgets/KeyPointsList";
import FormulaBox from "../widgets/FormulaBox";
import SourcesWidget from "../widgets/SourcesWidget";
import { parseAnswer } from "../../utils/answerParser";

interface AssistantBubbleProps {
  message: Message;
  isStreaming?: boolean;
}

export default function AssistantBubble({ message, isStreaming = false }: AssistantBubbleProps) {
  // Don't render at all when content is empty and streaming —
  // the StreamingIndicator in ChatContainer handles the loading state.
  if (isStreaming && !message.content) return null;

  const time = new Date(message.timestamp).toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });

  const citations = message.citations ?? [];
  const parsed = !isStreaming ? parseAnswer(message.content) : null;
  const hasWidgets = parsed && (parsed.summary || parsed.keyPoints.length > 0 || parsed.formulas.length > 0);

  return (
    <div className="flex justify-start animate-slide-left">
      <div className="max-w-[85%] w-full">
        {/* Header: bot avatar + label */}
        <div className="flex items-center gap-2 mb-1.5">
          <span className="text-cyan-500 text-base font-bold">&#9670;</span>
          <span className="text-xs font-semibold text-slate-600">Assistant</span>
        </div>

        <GlassCard className="p-4 shadow-md">
          {isStreaming ? (
            /* ── Streaming text with blinking cursor ── */
            <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">
              {message.content}
              <span className="animate-blink text-cyan-500 font-bold ml-0.5">|</span>
            </p>
          ) : hasWidgets ? (
            /* ── Structured widgets with citations passed down ── */
            <div className="space-y-3 stagger-children">
              {parsed!.summary && (
                <AnswerSummary content={parsed!.summary} citations={citations} />
              )}
              {parsed!.keyPoints.length > 0 && (
                <KeyPointsList points={parsed!.keyPoints} citations={citations} />
              )}
              {parsed!.formulas.length > 0 && (
                <FormulaBox formulas={parsed!.formulas} />
              )}
            </div>
          ) : (
            /* ── Fallback: plain text ── */
            <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">
              {message.content}
            </p>
          )}

          {/* ── Sources section (shown after streaming completes) ── */}
          {!isStreaming && citations.length > 0 && (
            <div className="mt-3 pt-3 border-t border-slate-100">
              <SourcesWidget citations={citations} />
            </div>
          )}
        </GlassCard>

        <p className="text-[10px] text-slate-400 mt-1">{time}</p>
      </div>
    </div>
  );
}
