/**
 * LaTeX Sanitizer
 * ───────────────
 * Fixes common LaTeX malformation in LLM output before passing to KaTeX.
 *
 * Handles:
 * - \fracf → \frac{f}  (missing braces)
 * - \frac{a}b → \frac{a}{b}  (missing denominator braces)
 * - d¸ots → \dots  (garbled commands)
 * - Bare LaTeX commands without $...$ wrapping
 * - Duplicate Unicode-rendered formulas
 */

/**
 * Sanitize a LaTeX string for KaTeX compatibility.
 * Call this on every math segment before passing to <InlineMath> or <BlockMath>.
 */
export function sanitizeLatex(latex: string): string {
  if (!latex) return latex;

  let result = latex;

  // ── 0. Fix spacing primitives (defense-in-depth) ──────────────────
  // \tmspace + 3mu.1667em → \, (thin space implementation detail)
  result = result.replace(/\\tmspace[^]*?(?=\\|\$|$)/g, "\\,");
  // \kern 3.0pt → (strip)
  result = result.replace(/\\kern\s*[+-]?\s*[\d.]+\s*(?:pt|pc|in|bp|cm|mm|em|ex|mu)/g, " ");
  // \hskip, \vskip, \mskip, \mkern → (strip)
  result = result.replace(/\\(?:hskip|vskip|mskip|mkern)\s*[+-]?\s*[\d.]*(?:\s*(?:pt|pc|in|bp|cm|mm|em|ex|mu))?/g, " ");
  // Collapse multiple spaces
  result = result.replace(/  +/g, " ");

  // ── 1. Fix garbled characters ──────────────────────────────────────
  // d¸ots → \dots (cedilla corruption from PDF extraction)
  result = result.replace(/d¸\s*ots/g, "\\dots");
  result = result.replace(/¸/g, ""); // Remove stray cedillas

  // ── 2. Fix \frac without braces ────────────────────────────────────
  // \fracf → \frac{f}
  result = result.replace(/\\frac([a-zA-Z])(?!\{)/g, "\\frac{$1}");
  // \frac2 → \frac{2}
  result = result.replace(/\\frac(\d)(?!\{)/g, "\\frac{$1}");
  // \frac{a}b → \frac{a}{b}
  result = result.replace(/\\frac\{([^}]+)\}([a-zA-Z0-9])/g, "\\frac{$1}{$2}");

  // ── 3. Fix sub/superscript without braces ──────────────────────────
  // \sum_k=0 → \sum_{k=0} (only for simple single-char subscripts)
  // Be careful: \sum_{k=0} is already correct
  result = result.replace(/\\(sum|prod|int|lim)_([a-zA-Z])(?=[\s=])/g, "\\$1_{$2}");

  // ── 4. Fix f (k) → f^{(k)} (derivative notation with spaces) ─────
  result = result.replace(/f\s+\(([a-zA-Z])\)/g, "f^{($1)}");

  // ── 5. Fix stray spaces in primes ─────────────────────────────────
  // f ' → f'
  result = result.replace(/f\s+'/g, "f'");
  result = result.replace(/f\s+''/g, "f''");

  // ── 6. Fix common Unicode replacements that slipped through ───────
  result = result.replace(/→/g, "\\to");
  result = result.replace(/∞/g, "\\infty");
  result = result.replace(/≤/g, "\\leq");
  result = result.replace(/≥/g, "\\geq");
  result = result.replace(/≠/g, "\\neq");
  result = result.replace(/·/g, "\\cdot");

  // ── 7. Fix \sum k=0 \infty → \sum_{k=0}^{\infty} ───────────────
  result = result.replace(
    /\\sum\s+(\w+)\s*=\s*(\d+)\s+\\infty/g,
    "\\sum_{$1=$2}^{\\infty}"
  );

  // ── 8. Fix double backslash issues ────────────────────────────────
  // \\\\frac → \\frac (sometimes escaped too much)
  result = result.replace(/\\\\(frac|sum|int|lim|sin|cos|tan|ln|log|sqrt|dots|cdots|ldots|to|infty|pi|alpha|beta|gamma|delta)/g, "\\$1");

  // ── 9. Remove duplicate formulas (raw + Unicode side by side) ─────
  result = removeDuplicateFormulas(result);

  return result.trim();
}

/**
 * Sanitize text content that may contain inline LaTeX.
 * Fixes LaTeX patterns within $...$ delimiters in running text.
 */
export function sanitizeTextWithMath(text: string): string {
  if (!text) return text;

  // First, fix garbled characters in the whole text
  let result = text.replace(/d¸\s*ots/g, "\\dots").replace(/¸/g, "");

  // Process $...$ segments through sanitizeLatex
  result = result.replace(
    /(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)/g,
    (_match, inner) => `$${sanitizeLatex(inner)}$`
  );

  // Process $$...$$ segments
  result = result.replace(
    /\$\$([\s\S]+?)\$\$/g,
    (_match, inner) => `$$${sanitizeLatex(inner)}$$`
  );

  return result;
}

/**
 * Remove lines that appear to be Unicode-rendered duplicates
 * of a preceding LaTeX formula line.
 */
function removeDuplicateFormulas(text: string): string {
  const lines = text.split("\n");
  const result: string[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const prevLine = i > 0 ? result[result.length - 1] : "";

    // Check if this line is a Unicode duplicate of the previous
    const unicodeMathChars = (line.match(/[∑∫∞≤≥≠→√∂∇]/g) || []).length;

    if (unicodeMathChars >= 2 && prevLine && prevLine.includes("\\")) {
      // Extract words from both lines
      const prevWords = new Set(prevLine.match(/[a-zA-Z]+/g) || []);
      const currWords = line.match(/[a-zA-Z]+/g) || [];
      const overlap = currWords.filter((w) => prevWords.has(w)).length;

      if (overlap >= 2 && currWords.length > 0 && overlap / currWords.length > 0.5) {
        continue; // Skip this duplicate
      }
    }

    result.push(line);
  }

  return result.join("\n");
}
