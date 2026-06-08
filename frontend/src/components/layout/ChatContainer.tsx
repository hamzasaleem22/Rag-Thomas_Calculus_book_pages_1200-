import { useEffect, useRef, useCallback } from "react";
import type { Message } from "../../types";
import UserBubble from "../chat/UserBubble";
import AssistantBubble from "../chat/AssistantBubble";
import MessageActions from "../chat/MessageActions";
import StreamingIndicator from "../chat/StreamingIndicator";
import EmptyState from "../chat/EmptyState";

interface ChatContainerProps {
  messages: Message[];
  isStreaming: boolean;
  onSend: (text: string) => void;
  onRegenerate: (messageId: string) => void;
  onDelete: (messageId: string) => void;
}

export default function ChatContainer({
  messages,
  isStreaming,
  onSend,
  onRegenerate,
  onDelete,
}: ChatContainerProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleCopy = useCallback((content: string) => {
    navigator.clipboard.writeText(content).catch(() => {
      // Fallback for older browsers
      const textarea = document.createElement("textarea");
      textarea.value = content;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      document.body.removeChild(textarea);
    });
  }, []);

  if (messages.length === 0) {
    return <EmptyState onExampleClick={onSend} />;
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6">
      <div className="max-w-3xl mx-auto space-y-4">
        {messages.map((msg, _idx) => {
          const isLastAssistant =
            msg.role === "assistant" && msg.id === messages[messages.length - 1]?.id;
          const isCurrentlyStreaming = isStreaming && isLastAssistant;

          return (
            <div key={msg.id} className="group relative">
              {msg.role === "user" ? (
                <UserBubble message={msg} />
              ) : (
                <AssistantBubble
                  message={msg}
                  isStreaming={isCurrentlyStreaming}
                />
              )}

              {/* Message actions — visible on hover */}
              {!isCurrentlyStreaming && msg.content && (
                <div className={msg.role === "user" ? "flex justify-end" : "flex justify-start"}>
                  <MessageActions
                    role={msg.role}
                    content={msg.content}
                    onCopy={() => handleCopy(msg.content)}
                    onRegenerate={
                      msg.role === "assistant"
                        ? () => onRegenerate(msg.id)
                        : undefined
                    }
                    onDelete={() => onDelete(msg.id)}
                  />
                </div>
              )}
            </div>
          );
        })}

        {/* Streaming indicator — shown when streaming but the assistant message is still empty */}
        {isStreaming &&
          messages.length > 0 &&
          messages[messages.length - 1].role === "assistant" &&
          messages[messages.length - 1].content === "" && (
            <StreamingIndicator isVisible />
          )}

        {/* Scroll anchor */}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
