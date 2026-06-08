import { type ReactNode } from "react";
import GlassCard from "../ui/GlassCard";
import { InlineMath } from "react-katex";
import type { Citation } from "../../types";
import { renderCitationText } from "./CitationPopup";
import { sanitizeLatex } from "../../utils/latexSanitizer";

interface KeyPointsListProps {
  points: string[];
  citations?: Citation[];
  className?: string;
}

/* ──────────────────────────────────────────────────────────────────────────
   Point text renderer with inline math + citation popups
   ────────────────────────────────────────────────────────────────────────── */

function renderPointText(
  text: string,
  pointIdx: number,
  citations: Citation[],
): ReactNode[] {
  const elements: ReactNode[] = [];

  // Split on display math first
  const displayParts = text.split(/(\$\$[\s\S]+?\$\$)/g);

  displayParts.forEach((part, i) => {
    if (part.startsWith("$$") && part.endsWith("$$") && part.length > 4) {
      const latex = sanitizeLatex(part.slice(2, -2).trim());
      elements.push(
        <InlineMath key={`kp-d-${pointIdx}-${i}`} math={latex} />,
      );
    } else {
      // Split on inline math $...$
      const inlineParts = part.split(/(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)/g);

      inlineParts.forEach((segment, j) => {
        if (j % 2 === 1) {
          elements.push(
            <InlineMath
              key={`kp-i-${pointIdx}-${i}-${j}`}
              math={sanitizeLatex(segment.trim())}
            />,
          );
        } else if (segment.length > 0) {
          if (citations.length > 0) {
            elements.push(
              <span key={`kp-t-${pointIdx}-${i}-${j}`}>
                {renderCitationText(segment, citations, `kp-${pointIdx}-${i}-${j}`)}
              </span>,
            );
          } else {
            elements.push(
              <span key={`kp-t-${pointIdx}-${i}-${j}`}>
                {renderSimpleCitations(segment, `kp-${pointIdx}-${i}-${j}`)}
              </span>,
            );
          }
        }
      });
    }
  });

  return elements;
}

function renderSimpleCitations(text: string, keyPrefix: string): ReactNode[] {
  const citationRe = /\[(\d+)\]/g;
  const parts: ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = citationRe.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    parts.push(
      <sup
        key={`${keyPrefix}-cite-${match.index}`}
        className="text-blue-500 font-semibold"
      >
        [{match[1]}]
      </sup>,
    );
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  if (parts.length === 0) {
    parts.push(text);
  }

  return parts;
}

/* ──────────────────────────────────────────────────────────────────────────
   Component
   ────────────────────────────────────────────────────────────────────────── */

export default function KeyPointsList({
  points,
  citations = [],
  className = "",
}: KeyPointsListProps) {
  if (!points || points.length === 0) return null;

  return (
    <GlassCard className={`p-4 animate-fade-in-up ${className}`}>
      <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
        Key Points
      </h3>

      <ul className="space-y-2.5">
        {points.map((point, idx) => (
          <li
            key={idx}
            className="flex gap-3 items-start animate-fade-in-up"
            style={{ animationDelay: `${idx * 80}ms` }}
          >
            {/* Gradient number circle */}
            <span className="w-6 h-6 rounded-full gradient-bg flex items-center justify-center text-white text-xs font-bold shrink-0">
              {idx + 1}
            </span>

            {/* Point text with inline math + citation popups */}
            <span className="text-sm text-slate-700 leading-relaxed">
              {renderPointText(point, idx, citations)}
            </span>
          </li>
        ))}
      </ul>
    </GlassCard>
  );
}
