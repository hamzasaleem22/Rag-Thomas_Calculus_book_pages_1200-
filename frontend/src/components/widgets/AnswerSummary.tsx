import { type ReactNode } from "react";
import GlassCard from "../ui/GlassCard";
import { InlineMath, BlockMath } from "react-katex";
import type { Citation } from "../../types";
import { renderCitationText } from "./CitationPopup";
import { sanitizeLatex } from "../../utils/latexSanitizer";
import { MathErrorBoundary } from "../ui/MathErrorBoundary";

interface AnswerSummaryProps {
  content: string;
  citations?: Citation[];
  className?: string;
}

/* ──────────────────────────────────────────────────────────────────────────
   Rich text renderer: handles $$...$$, $...$, and [N] citations.
   ────────────────────────────────────────────────────────────────────────── */

function renderRichText(
  text: string,
  citations: Citation[],
): ReactNode[] {
  const elements: ReactNode[] = [];

  // Step 1: Split on display math $$...$$
  const displayParts = text.split(/(\$\$[\s\S]+?\$\$)/g);

  displayParts.forEach((part, i) => {
    if (part.startsWith("$$") && part.endsWith("$$") && part.length > 4) {
      // Display math block
      const latex = sanitizeLatex(part.slice(2, -2).trim());
      elements.push(
        <span key={`display-${i}`} className="block my-3 overflow-x-auto">
          <MathErrorBoundary latex={latex}>
            <BlockMath math={latex} />
          </MathErrorBoundary>
        </span>,
      );
    } else {
      // Step 2: Split on inline math $...$  (not $$)
      const inlineParts = part.split(/(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)/g);

      inlineParts.forEach((segment, j) => {
        if (j % 2 === 1) {
          // Captured group — inline math
          const latex = sanitizeLatex(segment.trim());
          elements.push(
            <MathErrorBoundary key={`inline-${i}-${j}`} latex={latex}>
              <InlineMath math={latex} />
            </MathErrorBoundary>,
          );
        } else if (segment.length > 0) {
          // Step 3: Render plain text with interactive citation popups
          if (citations.length > 0) {
            elements.push(
              <span key={`text-${i}-${j}`}>
                {renderCitationText(segment, citations, `s-${i}-${j}`)}
              </span>,
            );
          } else {
            // No citations available — render as plain text with simple [N] badges
            elements.push(
              <span key={`text-${i}-${j}`}>
                {renderSimpleCitations(segment, `${i}-${j}`)}
              </span>,
            );
          }
        }
      });
    }
  });

  return elements;
}

/**
 * Fallback citation renderer when no citation data is available.
 * Renders [N] as simple styled superscript.
 */
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

export default function AnswerSummary({
  content,
  citations = [],
  className = "",
}: AnswerSummaryProps) {
  if (!content || !content.trim()) return null;

  return (
    <GlassCard className={`p-4 animate-fade-in-up ${className}`}>
      <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
        Summary
      </h3>
      <div className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">
        {renderRichText(content, citations)}
      </div>
    </GlassCard>
  );
}
