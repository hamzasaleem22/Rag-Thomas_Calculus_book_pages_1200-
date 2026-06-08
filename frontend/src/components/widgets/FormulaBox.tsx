import { Component, type ReactNode, type ErrorInfo } from "react";
import GlassCard from "../ui/GlassCard";
import { BlockMath, InlineMath } from "react-katex";
import { sanitizeLatex } from "../../utils/latexSanitizer";

interface FormulaBoxProps {
  formulas: string[];
  className?: string;
}

/* ──────────────────────────────────────────────────────────────────────────
   KaTeX Error Boundary
   ────────────────────────────────────────────────────────────────────────── */

interface MathErrorBoundaryProps {
  latex: string;
  children: ReactNode;
}

interface MathErrorBoundaryState {
  hasError: boolean;
}

class MathErrorBoundary extends Component<
  MathErrorBoundaryProps,
  MathErrorBoundaryState
> {
  constructor(props: MathErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): MathErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(_error: Error, _info: ErrorInfo) {
    // Silently handled
  }

  render() {
    if (this.state.hasError) {
      return (
        <code className="text-sm font-mono text-slate-500 block whitespace-pre-wrap">
          {this.props.latex}
        </code>
      );
    }
    return this.props.children;
  }
}

/* ──────────────────────────────────────────────────────────────────────────
   Helpers
   ────────────────────────────────────────────────────────────────────────── */

/**
 * Determine if a formula is "short" enough for inline rendering.
 * Short formulas like f'(x) = cos(x) render better inline.
 * Long formulas like integrals with limits need display mode.
 */
function isShortFormula(latex: string): boolean {
  // Count complexity indicators
  const hasFrac = /\\frac/.test(latex);
  const hasIntegral = /\\int/.test(latex);
  const hasSum = /\\sum/.test(latex);
  const hasMatrix = /\\begin\{/.test(latex);
  const hasMultiLine = latex.includes("\\\\") || latex.includes("\n");
  const charCount = latex.length;

  if (hasMatrix || hasMultiLine) return false;
  if (hasIntegral || hasSum) return false;
  if (hasFrac && charCount > 30) return false;
  if (charCount > 50) return false;

  return true;
}

/* ──────────────────────────────────────────────────────────────────────────
   Component
   ────────────────────────────────────────────────────────────────────────── */

export default function FormulaBox({
  formulas,
  className = "",
}: FormulaBoxProps) {
  if (!formulas || formulas.length === 0) return null;

  // Limit to 3 formulas max to keep the widget compact
  const displayFormulas = formulas.slice(0, 3);

  return (
    <GlassCard className={`gradient-border animate-fade-in-up ${className}`}>
      <div className="px-4 py-3">
        <h3 className="text-xs font-semibold text-blue-600 uppercase tracking-wider mb-2">
          Formula
        </h3>

        <div className="space-y-2">
          {displayFormulas.map((formula, idx) => {
            const sanitized = sanitizeLatex(formula);
            return (
              <div
                key={idx}
                className="bg-slate-50/60 rounded-lg px-3 py-2.5 overflow-x-auto"
              >
                <MathErrorBoundary latex={sanitized}>
                  {isShortFormula(sanitized) ? (
                    <span className="text-base">
                      <InlineMath math={sanitized} />
                    </span>
                  ) : (
                    <div className="katex-compact">
                      <BlockMath math={sanitized} />
                    </div>
                  )}
                </MathErrorBoundary>
              </div>
            );
          })}
        </div>

        {/* Show count if there are more formulas beyond the 3 shown */}
        {formulas.length > 3 && (
          <p className="text-[11px] text-slate-400 mt-1.5">
            +{formulas.length - 3} more formula{formulas.length - 3 > 1 ? "s" : ""}
          </p>
        )}
      </div>
    </GlassCard>
  );
}
