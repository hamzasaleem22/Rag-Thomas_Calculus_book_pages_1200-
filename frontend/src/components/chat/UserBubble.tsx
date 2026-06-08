import type { Message } from "../../types";

interface UserBubbleProps {
  message: Message;
}

export default function UserBubble({ message }: UserBubbleProps) {
  const time = new Date(message.timestamp).toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });

  return (
    <div className="flex justify-end animate-slide-right">
      <div className="max-w-[80%]">
        <div className="gradient-bg text-white rounded-2xl rounded-tr-md px-5 py-3 shadow-md shadow-blue-500/10">
          <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
        </div>
        <p className="text-[10px] text-slate-400 mt-1 text-right">{time}</p>
      </div>
    </div>
  );
}
