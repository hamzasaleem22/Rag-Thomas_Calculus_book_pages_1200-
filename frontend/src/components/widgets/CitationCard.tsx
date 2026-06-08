import type { Citation } from "../../types";
import Badge from "../ui/Badge";

interface CitationCardProps {
  citation: Citation;
  index: number;
}

export default function CitationCard({ citation, index }: CitationCardProps) {
  return (
    <div className="bg-slate-50/80 rounded-xl p-3 border border-slate-100">
      {/* Top row: index + badges */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-slate-400 text-xs font-mono">[{index + 1}]</span>

        {citation.page != null && (
          <Badge variant="blue">p.{citation.page}</Badge>
        )}

        {citation.chapter != null && citation.chapter.trim() !== "" && (
          <Badge variant="violet">{citation.chapter}</Badge>
        )}

        {citation.section != null && citation.section.trim() !== "" && (
          <Badge variant="cyan">{citation.section}</Badge>
        )}
      </div>

      {/* Excerpt text — max 2 lines */}
      {citation.text && (
        <p className="text-xs text-slate-500 leading-relaxed mt-1.5 line-clamp-2">
          {citation.text}
        </p>
      )}
    </div>
  );
}
