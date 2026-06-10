import GlassCard from "../ui/GlassCard";
import type { Citation } from "../../types";
import { renderRichText } from "./RichTextRenderer";

interface DynamicSectionProps {
  header: string;
  content: string;
  citations?: Citation[];
  index?: number;
}

export default function DynamicSection({
  header,
  content,
  citations = [],
  index = 0,
}: DynamicSectionProps) {
  if (!content || !content.trim()) return null;

  return (
    <div style={{ animationDelay: `${index * 80}ms` }} className="animate-fade-in-up">
      <GlassCard className="p-4">
        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
          {header}
        </h3>
        <div className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">
          {renderRichText(content, citations)}
        </div>
      </GlassCard>
    </div>
  );
}
