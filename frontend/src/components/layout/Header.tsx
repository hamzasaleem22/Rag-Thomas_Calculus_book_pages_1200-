import IconButton from "../ui/IconButton";

interface HeaderProps {
  onNewChat: () => void;
  onToggleHistory: () => void;
  sessionCount: number;
  showHistory: boolean;
}

/* ── Inline SVG Icons (20×20, stroke-based) ── */

function PlusIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <line x1="10" y1="4" x2="10" y2="16" />
      <line x1="4" y1="10" x2="16" y2="10" />
    </svg>
  );
}

function HistoryIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="10" cy="10" r="7" />
      <polyline points="10,6 10,10 13,12" />
    </svg>
  );
}

export default function Header({
  onNewChat,
  onToggleHistory,
  sessionCount,
  showHistory,
}: HeaderProps) {
  return (
    <header className="glass-strong border-b border-slate-200/50 px-4 py-3 flex items-center justify-between shrink-0">
      {/* ── Left: Brand ── */}
      <div className="flex items-center gap-2.5">
        <span className="gradient-text text-xl select-none" aria-hidden="true">
          ◆
        </span>
        <div className="flex flex-col leading-tight">
          <span className="gradient-text text-lg font-bold">
            Thomas' Calculus AI
          </span>
          <span className="text-xs text-slate-400 -mt-0.5">14th Edition</span>
        </div>
      </div>

      {/* ── Right: Actions ── */}
      <div className="flex items-center gap-1.5">
        <IconButton onClick={onNewChat} tooltip="New Chat">
          <PlusIcon />
        </IconButton>

        <div className="relative">
          <IconButton
            onClick={onToggleHistory}
            tooltip={showHistory ? "Hide History" : "Chat History"}
          >
            <HistoryIcon />
          </IconButton>
          {sessionCount > 0 && (
            <span className="absolute -top-1 -right-1 min-w-[18px] h-[18px] flex items-center justify-center
              gradient-bg text-white text-[10px] font-bold rounded-full px-1
              shadow-sm shadow-blue-500/30 pointer-events-none">
              {sessionCount > 99 ? "99+" : sessionCount}
            </span>
          )}
        </div>
      </div>
    </header>
  );
}
