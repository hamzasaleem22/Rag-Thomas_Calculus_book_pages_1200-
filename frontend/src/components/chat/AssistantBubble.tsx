import type { ReactNode } from "react";
import type { Message } from "../../types";
import GlassCard from "../ui/GlassCard";
import DynamicSection from "../widgets/DynamicSection";
import SourcesWidget from "../widgets/SourcesWidget";
import { parseAnswer } from "../../utils/answerParser";
import RichTextRenderer from "../widgets/RichTextRenderer";

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
  const parsed = parseAnswer(message.content);

  let contentNode: ReactNode;

  if (parsed.sections.length > 0) {
    // ── Dynamic sections with KaTeX rendering ──
    contentNode = (
      <div className="space-y-3 stagger-children">
        {parsed.sections.map((section, i) => (
          <DynamicSection
            key={i}
            header={section.header}
            content={section.content}
            citations={citations}
            index={i}
          />
        ))}
      </div>
    );
  } else {
    // ── Fallback: rich text with math rendering ──
    contentNode = (
      <RichTextRenderer text={message.content} citations={citations} />
    );
  }

  return (
    <div className="flex justify-start animate-slide-left">
      <div className="max-w-[85%] w-full">
        {/* Header: bot avatar + label */}
        <div className="flex items-center gap-2 mb-1.5">
          <span className="text-cyan-500 text-base font-bold">&#9670;</span>
          <span className="text-xs font-semibold text-slate-600">Assistant</span>
        </div>

        <GlassCard className="p-4 shadow-md">
          {contentNode}

          {/* ── Streaming cursor ── */}
          {isStreaming && (
            <span className="animate-blink text-cyan-500 font-bold ml-0.5">|</span>
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
