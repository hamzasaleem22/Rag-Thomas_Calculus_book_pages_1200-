import { useState } from "react";
import type { Citation } from "../../types";
import GlassCard from "../ui/GlassCard";
import Badge from "../ui/Badge";
import CitationCard from "./CitationCard";

interface SourcesWidgetProps {
  citations: Citation[];
  className?: string;
}

/* ──────────────────────────────────────────────────────────────────────────
   Chevron icon (inline SVG so we don't need an icon library)
   ────────────────────────────────────────────────────────────────────────── */

function ChevronIcon({ expanded }: { expanded: boolean }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={`text-slate-400 transition-transform duration-300 ${
        expanded ? "rotate-180" : "rotate-0"
      }`}
    >
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}

/* ──────────────────────────────────────────────────────────────────────────
   Component
   ────────────────────────────────────────────────────────────────────────── */

export default function SourcesWidget({
  citations,
  className = "",
}: SourcesWidgetProps) {
  const [expanded, setExpanded] = useState(false);

  if (!citations || citations.length === 0) return null;

  return (
    <GlassCard className={`p-4 animate-fade-in-up ${className}`}>
      {/* Header — clickable toggle */}
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        className="flex items-center justify-between w-full"
      >
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
            📚 Sources
          </span>
          <Badge variant="blue">{citations.length}</Badge>
        </div>

        <ChevronIcon expanded={expanded} />
      </button>

      {/* Collapsible body */}
      <div
        className={`overflow-hidden transition-all duration-300 ${
          expanded ? "max-h-[2000px] opacity-100" : "max-h-0 opacity-0"
        }`}
      >
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-3">
          {citations.map((citation, idx) => (
            <CitationCard key={idx} citation={citation} index={idx} />
          ))}
        </div>
      </div>
    </GlassCard>
  );
}
