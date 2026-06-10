import { type ReactNode } from "react";
import { InlineMath, BlockMath } from "react-katex";
import type { Citation } from "../../types";
import { renderCitationText } from "./CitationPopup";
import { sanitizeLatex } from "../../utils/latexSanitizer";
import { MathErrorBoundary } from "../ui/MathErrorBoundary";

export function renderSimpleCitations(text: string, keyPrefix: string): ReactNode[] {
  const citationRe = /\[(\d+)\]/g;
  const parts: ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = citationRe.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    parts.push(
      <sup key={`${keyPrefix}-cite-${match.index}`} className="text-blue-500 font-semibold">
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

export function renderRichText(text: string, citations: Citation[]): ReactNode[] {
  const elements: ReactNode[] = [];

  const displayParts = text.split(/(\$\$[\s\S]+?\$\$)/g);

  displayParts.forEach((part, i) => {
    if (part.startsWith("$$") && part.endsWith("$$") && part.length > 4) {
      const latex = sanitizeLatex(part.slice(2, -2).trim());
      elements.push(
        <span key={`display-${i}`} className="block my-3 overflow-x-auto">
          <MathErrorBoundary latex={latex}>
            <BlockMath math={latex} />
          </MathErrorBoundary>
        </span>,
      );
    } else {
      const inlineParts = part.split(/(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)/g);

      inlineParts.forEach((segment, j) => {
        if (j % 2 === 1) {
          const latex = sanitizeLatex(segment.trim());
          elements.push(
            <MathErrorBoundary key={`inline-${i}-${j}`} latex={latex}>
              <InlineMath math={latex} />
            </MathErrorBoundary>,
          );
        } else if (segment.length > 0) {
          if (citations.length > 0) {
            elements.push(
              <span key={`text-${i}-${j}`}>
                {renderCitationText(segment, citations, `s-${i}-${j}`)}
              </span>,
            );
          } else {
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

interface RichTextRendererProps {
  text: string;
  citations?: Citation[];
  className?: string;
}

export default function RichTextRenderer({
  text,
  citations = [],
  className = "",
}: RichTextRendererProps) {
  if (!text || !text.trim()) return null;

  return (
    <div className={`text-sm text-slate-700 leading-relaxed whitespace-pre-wrap ${className}`}>
      {renderRichText(text, citations)}
    </div>
  );
}
