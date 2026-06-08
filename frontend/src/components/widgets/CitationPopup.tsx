import { useState, useRef, useEffect, type ReactNode } from "react";
import type { Citation } from "../../types";
import Badge from "../ui/Badge";

interface CitationPopupProps {
  /** The citation index number (1-based, as it appears in the text) */
  index: number;
  /** Full citations array from the message */
  citations: Citation[];
}

export default function CitationPopup({ index, citations }: CitationPopupProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [position, setPosition] = useState<"above" | "below">("above");
  const triggerRef = useRef<HTMLSpanElement>(null);
  const popupRef = useRef<HTMLDivElement>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout>>(null);

  // Find the matching citation (index is 1-based, citations array is 0-based)
  const citation = citations[index - 1] ?? null;

  // Calculate popup position when opening
  useEffect(() => {
    if (isOpen && triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect();
      // If not enough space above (popup ~180px), show below
      if (rect.top < 200) {
        setPosition("below");
      } else {
        setPosition("above");
      }
    }
  }, [isOpen]);

  function handleMouseEnter() {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setIsOpen(true);
  }

  function handleMouseLeave() {
    timeoutRef.current = setTimeout(() => setIsOpen(false), 200);
  }

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

  return (
    <span
      className="relative inline-block"
      ref={triggerRef}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {/* Citation marker — styled as a clickable badge */}
      <sup className="cursor-help">
        <span className="inline-flex items-center justify-center min-w-[20px] h-[18px] px-1 rounded-md bg-blue-50 text-blue-600 text-[11px] font-bold border border-blue-100 hover:bg-blue-100 hover:border-blue-200 transition-colors duration-150">
          {index}
        </span>
      </sup>

      {/* Popup */}
      {isOpen && citation && (
        <div
          ref={popupRef}
          className={`absolute z-50 w-72 ${
            position === "above" ? "bottom-full mb-2" : "top-full mt-2"
          } left-1/2 -translate-x-1/2 animate-scale-in`}
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseLeave}
        >
          <div className="glass-strong rounded-xl shadow-xl shadow-slate-200/50 p-3 border border-slate-200/60">
            {/* Header row: badges */}
            <div className="flex items-center gap-1.5 mb-2 flex-wrap">
              {citation.page != null && (
                <Badge variant="blue">p.{citation.page}</Badge>
              )}
              {citation.chapter && (
                <Badge variant="violet">{citation.chapter}</Badge>
              )}
              {citation.section && (
                <Badge variant="cyan">{citation.section}</Badge>
              )}
              <span className="ml-auto text-[10px] text-slate-400 font-medium">
                Source [{index}]
              </span>
            </div>

            {/* Citation text */}
            <p className="text-xs text-slate-600 leading-relaxed line-clamp-4">
              {citation.text}
            </p>

            {/* Arrow pointer */}
            <div
              className={`absolute left-1/2 -translate-x-1/2 w-3 h-3 rotate-45 glass-strong border-slate-200/60 ${
                position === "above"
                  ? "-bottom-1.5 border-b border-r"
                  : "-top-1.5 border-t border-l"
              }`}
            />
          </div>
        </div>
      )}

      {/* Fallback when citation data is missing */}
      {isOpen && !citation && (
        <div
          className={`absolute z-50 ${
            position === "above" ? "bottom-full mb-2" : "top-full mt-2"
          } left-1/2 -translate-x-1/2 animate-scale-in`}
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseLeave}
        >
          <div className="glass-strong rounded-lg shadow-lg px-3 py-1.5 border border-slate-200/60">
            <span className="text-[11px] text-slate-400">Source [{index}]</span>
          </div>
        </div>
      )}
    </span>
  );
}

/**
 * Render text with interactive citation popups.
 * Replaces [N] markers with CitationPopup components.
 *
 * @param text - The text containing [N] citation markers
 * @param citations - The full citations array from the message
 * @param keyPrefix - Unique prefix for React keys
 */
export function renderCitationText(
  text: string,
  citations: Citation[],
  keyPrefix: string,
): ReactNode[] {
  const citationRe = /\[(\d+)\]/g;
  const parts: ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = citationRe.exec(text)) !== null) {
    // Push text before the citation
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }

    const refIndex = parseInt(match[1], 10);
    parts.push(
      <CitationPopup
        key={`${keyPrefix}-cite-${match.index}`}
        index={refIndex}
        citations={citations}
      />,
    );

    lastIndex = match.index + match[0].length;
  }

  // Remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  if (parts.length === 0) {
    parts.push(text);
  }

  return parts;
}
