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

  // ── 0.5 Fix truncated/abbreviated LaTeX commands ──────────────────
  // LLMs sometimes output \f instead of \frac, \i instead of \int, etc.
  // Only fix when the truncated form is followed by { or _ — patterns
  // that could NEVER appear in valid LaTeX commands.
  result = result
    // \f{numerator}{denominator} → \frac{numerator}{denominator}
    .replace(/\\f(?=\{)/g, "\\frac")
    // \sq{ → \sqrt{  (\sqcap, \sqcup, \square use letters, not {)
    .replace(/\\sq(?=\{)/g, "\\sqrt")
    // \i_{lower}^{upper} → \int_{lower}^{upper}  (valid cmds start with \in, not \i_)
    .replace(/\\i(?=_)/g, "\\int")
    // \s_{lower}^{upper} → \sum_{lower}^{upper}  (valid cmds start with \si, \su, not \s_)
    .replace(/\\s(?=_)/g, "\\sum")
    // \p_{lower}^{upper} → \prod_{lower}^{upper}  (valid cmds start with \pi, \pa, not \p_)
    .replace(/\\p(?=_)/g, "\\prod")
    // \inf} , \inf  → \infty  (\infty is the only cmd starting with \if)
    .replace(/\\inf(?=[\s,}\]])/g, "\\infty");

  // ── 1. Fix garbled characters ──────────────────────────────────────
  // d¸ots → \dots (cedilla corruption from PDF extraction)
  result = result.replace(/d¸\s*ots/g, "\\dots");
  result = result.replace(/¸/g, ""); // Remove stray cedillas

  // ── 2. Fix \frac with multi-char differentials (must run before single-char frac fix) ──
  // \fracdydx → \frac{dy}{dx}
  const diffTokens = ["dx","dy","du","dv","dt","dr","ds","dz","dθ","dφ","dψ"];
  // Sort by length descending so "dx" doesn't match before "dθ"
  diffTokens.sort((a,b) => b.length - a.length);
  const diffPattern = diffTokens.join("|");
  const diffRe = new RegExp("\\\\frac(" + diffPattern + ")(" + diffPattern + ")", "g");
  result = result.replace(diffRe, (_m, num, den) => `\\frac{${num}}{${den}}`);

  // Broad pattern: \frac followed by two 2-letter lowercase groups
  result = result.replace(/\\frac([a-z]{2})([a-z]{2})/g, "\\frac{$1}{$2}");

  // ── 3. Fix \frac without braces ────────────────────────────────────
  // \fracf → \frac{f}
  result = result.replace(/\\frac([a-zA-Z])(?!\{)/g, "\\frac{$1}");
  // \frac2 → \frac{2}
  result = result.replace(/\\frac(\d)(?!\{)/g, "\\frac{$1}");
  // \frac{a}b → \frac{a}{b}
  result = result.replace(/\\frac\{([^}]+)\}([a-zA-Z0-9])/g, "\\frac{$1}{$2}");

  // ── 4. Fix sub/superscript without braces ──────────────────────────
  // \sum_k=0 → \sum_{k=0} (only for simple single-char subscripts)
  result = result.replace(/\\(sum|prod|int|lim)_([a-zA-Z])(?=[\s=])/g, "\\$1_{$2}");

  // ── 5. Fix missing operator space: \intf(x) → \int f(x) ───────────
  result = result.replace(/\\(int|iint|iiint|oint|sum|prod)([a-zA-Z])/g, "\\$1 $2");

  // ── 6. Fix f (k) → f^{(k)} (derivative notation with spaces) ──────
  result = result.replace(/f\s+\(([a-zA-Z])\)/g, "f^{($1)}");

  // ── 7. Fix stray spaces in primes ─────────────────────────────────
  result = result.replace(/f\s+'/g, "f'");
  result = result.replace(/f\s+''/g, "f''");

  // ── 8. Fix superscript braces: f^(k)(x) → f^{(k)}(x), x^2y → x^{2}y ──
  result = result.replace(/f\^\((\w)\)/g, "f^{($1)}");
  result = result.replace(/(\w)\^(\w{2,})(?!\})/g, "$1^{$2}");

  // ── 9. Fix missing backslash on math functions (not preceded by \ ) ──
  const mathFns = ["sin","cos","tan","cot","sec","csc","arcsin","arccos","arctan",
    "sinh","cosh","tanh","ln","log","lg","exp","lim","det","dim","sup","inf","max","min",
    "cdot","dots","to","infty","pi","nabla","partial","times","div","circ","propto"];
  for (const fn of mathFns) {
    // Match function name that is NOT preceded by backslash (i.e. not already \sin etc.)
    result = result.replace(
      new RegExp("(?<!\\\\)\\b(" + fn + ")\\b", "g"),
      "\\$1"
    );
  }

  // ── 10. Fix common Unicode replacements that slipped through ───────
  result = result.replace(/→/g, "\\to");
  result = result.replace(/∞/g, "\\infty");
  result = result.replace(/≤/g, "\\leq");
  result = result.replace(/≥/g, "\\geq");
  result = result.replace(/≠/g, "\\neq");
  result = result.replace(/·/g, "\\cdot");
  // U+22C5 DOT OPERATOR (different from U+00B7 MIDDLE DOT)
  result = result.replace(/⋅/g, "\\cdot");
  // The word "dot" between math expressions → \cdot
  result = result.replace(/}([.\s]*)dot([\s]*){/gi, "}\\cdot{");
  result = result.replace(/}([.\s]*)dot([\s]*)\\frac/gi, "}\\cdot\\frac");

  // ── 11. Fix \sum k=0 \infty → \sum_{k=0}^{\infty} ─────────────────
  result = result.replace(/\\sum\s+(\w+)\s*=\s*(\d+)\s+\\infty/g, "\\sum_{$1=$2}^{\\infty}");
  // \sum infty → \sum_{k=0}^{\infty}
  result = result.replace(/\\sum\s+infty/g, "\\sum_{k=0}^{\\infty}");
  // \sum_infty → \sum_{k=0}^{\infty}
  result = result.replace(/\\sum\s*_\s*infty/g, "\\sum_{k=0}^{\\infty}");
  // \sum_{k=0}\infty → \sum_{k=0}^{\infty}
  result = result.replace(/\\sum_\{([^}]+)\}\s*infty/g, "\\sum_{$1}^{\\infty}");

  // ── 12. Fix double backslash issues ────────────────────────────────
  // \\frac → \frac (double backslash to single)
  result = result.replace(/\\\\(frac|sum|int|iint|iiint|oint|prod|lim|sin|cos|tan|ln|log|sqrt|dots|cdots|ldots|to|infty|pi|alpha|beta|gamma|delta|theta|lambda|mu|sigma|phi|psi|omega|nabla|partial|times|cdot|circ|text|mathbf|mathcal|mathbb|overrightarrow|rightarrow|Rightarrow|mapsto|implies|iff)/g, "\\$1");

  // ── 13. Remove duplicate formulas (raw + Unicode side by side) ────
  result = removeDuplicateFormulas(result);

  return result.trim();
}

/**
 * Sanitize text content that may contain inline LaTeX.
 * Fixes LaTeX patterns within $...$ delimiters in running text.
 */
export function sanitizeTextWithMath(text: string): string {
  if (!text) return text;

  // First, fix garbled characters and Unicode artifacts in the whole text
  let result = text
    .replace(/d¸\s*ots/g, "\\dots")
    .replace(/¸/g, "")
    // Unicode math symbols that LaTeX should handle
    .replace(/⋅/g, "\\cdot")   // U+22C5 DOT OPERATOR
    .replace(/·/g, "\\cdot")   // U+00B7 MIDDLE DOT
    .replace(/→/g, "\\to")
    .replace(/∞/g, "\\infty")
    // The word "dot" in place of \cdot
    .replace(/}([.\s]*)dot([\s]*){/gi, "}\\cdot{")
    .replace(/}([.\s]*)dot([\s]*)\\frac/gi, "}\\cdot\\frac");

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
