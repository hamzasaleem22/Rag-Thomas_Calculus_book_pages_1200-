interface StreamingIndicatorProps {
  isVisible: boolean;
}

export default function StreamingIndicator({ isVisible }: StreamingIndicatorProps) {
  if (!isVisible) return null;

  return (
    <div className="flex justify-start animate-fade-in">
      <div className="flex items-center gap-2 glass rounded-2xl px-4 py-3">
        <div className="flex gap-1">
          <span
            className="w-2 h-2 rounded-full bg-blue-400"
            style={{ animation: "dotPulse 1.4s ease-in-out infinite", animationDelay: "0s" }}
          />
          <span
            className="w-2 h-2 rounded-full bg-cyan-400"
            style={{ animation: "dotPulse 1.4s ease-in-out infinite", animationDelay: "0.2s" }}
          />
          <span
            className="w-2 h-2 rounded-full bg-violet-400"
            style={{ animation: "dotPulse 1.4s ease-in-out infinite", animationDelay: "0.4s" }}
          />
        </div>
        <span className="text-xs text-slate-500">Thinking</span>
      </div>
    </div>
  );
}
