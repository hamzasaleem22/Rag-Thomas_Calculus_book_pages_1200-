import type { ParsedAnswer } from "../types";
import { sanitizeTextWithMath } from "./latexSanitizer";

/**
 * Parse the LLM's structured answer text into a ParsedAnswer object.
 *
 * Handles these section headers:
 *   **Answer:** / **Key Points:** / ### Key Points / **Formula:** / ### Formula
 *
 * Falls back to treating the entire text as the summary when no headers match.
 * Also sanitizes LaTeX and normalizes plain-text math.
 */
export function parseAnswer(text: string): ParsedAnswer {
  if (!text || !text.trim()) {
    return { summary: "", keyPoints: [], formulas: [], rawText: text ?? "" };
  }

  // Pre-process: sanitize LaTeX + normalize plain-text math
  const sanitized = sanitizeTextWithMath(text);
  const normalized = normalizeMathText(sanitized);
  const rawText = text;

  // ── 1. Try to split by structured headers ──────────────────────────
  const summaryMatch = normalized.match(
    /\*\*Answer:\*\*\s*([\s\S]*?)(?=\*\*Key Points:\*\*|### Key Points|\*\*Formula:\*\*|### Formula|$)/i,
  );
  const keyPointsMatch = normalized.match(
    /(?:\*\*Key Points:\*\*|### Key Points)\s*([\s\S]*?)(?=\*\*Formula:\*\*|### Formula|$)/i,
  );
  const formulaMatch = normalized.match(
    /(?:\*\*Formula:\*\*|### Formula[s]?)([\s\S]*?)$/i,
  );

  // ── 2. Build summary ───────────────────────────────────────────────
  let summary: string;
  if (summaryMatch) {
    summary = summaryMatch[1].trim();
  } else if (!keyPointsMatch && !formulaMatch) {
    summary = normalized.trim();
  } else {
    const firstHeader = normalized.search(
      /(?:\*\*Key Points:\*\*|### Key Points|\*\*Formula:\*\*|### Formula)/i,
    );
    summary = firstHeader > 0 ? normalized.slice(0, firstHeader).trim() : normalized.trim();
  }

  // ── 3. Build key points ────────────────────────────────────────────
  let keyPoints: string[] = [];
  if (keyPointsMatch) {
    const kpBlock = keyPointsMatch[1].trim();
    keyPoints = extractBulletPoints(kpBlock);
  }

  if (keyPoints.length === 0 && summary) {
    const bullets = extractBulletPoints(summary);
    if (bullets.length >= 2) {
      const nonBulletLines = summary
        .split("\n")
        .filter(
          (l) =>
            l.trim() !== "" &&
            !/^\s*[•\-*]\s/.test(l) &&
            !/^\s*\d+\.\s/.test(l),
        );
      if (nonBulletLines.length > 0) {
        keyPoints = bullets;
        summary = nonBulletLines.join("\n").trim();
      }
    }
  }

  // ── 4. Build formulas ──────────────────────────────────────────────
  // ONLY collect from the explicit **Formula:** section.
  // Do NOT extract inline $...$ from summary/keypoints — those are already
  // rendered inline by AnswerSummary and KeyPointsList via KaTeX.
  // This prevents the FormulaBox from being bloated with duplicates.
  let formulas: string[] = [];

  if (formulaMatch) {
    // From the formula section, collect display math ($$...$$)
    formulas = extractDisplayFormulas(formulaMatch[1]);

    // If no display math found, the formula section likely contains inline $...$
    // — extract those as a fallback (the LLM wrote the formula inline)
    if (formulas.length === 0) {
      formulas = extractInlineFormulas(formulaMatch[1]);
    }
  }

  // De-duplicate and filter
  formulas = [...new Set(formulas.map((f) => f.trim()))].filter(Boolean);

  return { summary, keyPoints, formulas, rawText };
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/**
 * Normalize plain-text math expressions into LaTeX with $...$ delimiters.
 *
 * This handles the case where the LLM outputs math without LaTeX wrapping,
 * e.g., "d/dx[sin(x)] = cos(x)" → "$\frac{d}{dx}[\sin(x)] = \cos(x)$"
 *
 * The function is conservative — it only wraps patterns that are clearly
 * mathematical and not already inside $...$ or $$...$$ delimiters.
 */
export function normalizeMathText(text: string): string {
  let result = text;

  // 1. Protect existing LaTeX delimiters by temporarily replacing them
  const placeholders: string[] = [];
  // Protect $$...$$
  result = result.replace(/\$\$([\s\S]+?)\$\$/g, (match) => {
    placeholders.push(match);
    return `%%MATH_BLOCK_${placeholders.length - 1}%%`;
  });
  // Protect $...$
  result = result.replace(/(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)/g, (match) => {
    placeholders.push(match);
    return `%%MATH_INLINE_${placeholders.length - 1}%%`;
  });

  // 2. Convert common plain-text math patterns to LaTeX

  // d/dx or d²/dx² → \frac{d}{dx} or \frac{d^2}{dx^2}
  result = result.replace(/d(\^?\d*)\/d([a-z])/g, (_match, power, varName) => {
    const pow = power === "^2" ? "^2" : power === "²" ? "^2" : "";
    return `\\frac{d${pow}}{d${varName}}`;
  });

  // d/dx[...] → \frac{d}{dx}[...]
  result = result.replace(/\\frac\{d\}\{d([a-z])\}\[/g, "\\frac{d}{d$1}\\left[");
  result = result.replace(/\](\s*=)/g, "\\right]$1");

  // ∫ integral sign without delimiters
  result = result.replace(/(?<!\$)∫(?!\$)/g, "\\int");

  // Σ sigma without delimiters
  result = result.replace(/(?<!\$)Σ(?!\$)/g, "\\sum");

  // π pi without delimiters
  result = result.replace(/(?<![\\a-zA-Z])π(?![a-zA-Z])/g, "\\pi");

  // ∞ infinity
  result = result.replace(/(?<![\\a-zA-Z])∞(?![a-zA-Z])/g, "\\infty");

  // Common function names that should be LaTeX: sin, cos, tan, ln, log, sec, csc, cot, sinh, cosh, tanh, arcsin, arccos, arctan
  // Only wrap if they look like function calls: followed by ( or [
  const mathFns = [
    "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan",
    "sin", "cos", "tan", "sec", "csc", "cot",
    "ln", "log", "exp", "lim",
  ];
  for (const fn of mathFns) {
    // Match function name NOT preceded by \ (already LaTeX) or letters (part of word)
    const fnRe = new RegExp(`(?<![\\\\a-zA-Z])${fn}(?=[\\s(\\[])`, "g");
    result = result.replace(fnRe, `\\${fn}`);
  }

  // x^n patterns like x² → x^2, x³ → x^3
  result = result.replace(/([a-zA-Z0-9)])²/g, "$1^2");
  result = result.replace(/([a-zA-Z0-9)])³/g, "$1^3");

  // sqrt(x) → \sqrt{x}  (only if not already \sqrt)
  result = result.replace(/(?<!\\)sqrt\(([^)]+)\)/gi, "\\sqrt{$1}");

  // 3. Now wrap any line/segment that contains LaTeX commands but isn't
  //    already delimited in $...$
  // We look for lines containing \frac, \int, \sum, \sqrt, etc. that
  // aren't already in $...$
  result = wrapOrphanedLatex(result);

  // 4. Restore protected math blocks
  result = result.replace(/%%MATH_BLOCK_(\d+)%%/g, (_, idx) => placeholders[parseInt(idx)]);
  result = result.replace(/%%MATH_INLINE_(\d+)%%/g, (_, idx) => placeholders[parseInt(idx)]);

  return result;
}

/**
 * Find LaTeX commands that aren't inside $...$ delimiters and wrap them.
 * Looks for segments containing \frac, \int, \sum, \sqrt, \lim, \sin, etc.
 */
function wrapOrphanedLatex(text: string): string {
  const latexCmdRe = /(?:\\frac|\\int|\\sum|\\sqrt|\\lim|\\sin|\\cos|\\tan|\\ln|\\log|\\exp|\\pi|\\infty|\\left|\\right)/;

  const lines = text.split("\n");
  const result: string[] = [];

  for (const line of lines) {
    // If the line contains LaTeX commands but no $ delimiters at all,
    // try to wrap the mathematical portion
    if (latexCmdRe.test(line) && !line.includes("$")) {
      // Find the start of the math expression
      const match = line.match(/^([\s]*)([\s\S]*)$/);
      if (match) {
        const [, leading, content] = match;
        // If the whole line is math, wrap it
        result.push(`${leading}$${content}$`);
        continue;
      }
    }
    result.push(line);
  }

  return result.join("\n");
}

/**
 * Extract bullet-point items from a block of text.
 * Recognises lines starting with •, -, *, or a number followed by a dot.
 */
function extractBulletPoints(block: string): string[] {
  const lines = block.split("\n");
  const points: string[] = [];

  for (const line of lines) {
    const trimmed = line.trim();
    const bulletMatch = trimmed.match(/^[•\-*]\s+(.+)/);
    const numMatch = trimmed.match(/^\d+\.\s+(.+)/);

    if (bulletMatch) {
      points.push(bulletMatch[1].trim());
    } else if (numMatch) {
      points.push(numMatch[1].trim());
    }
  }

  return points;
}

/**
 * Extract only display math formulas ($$...$$).
 * Returns the LaTeX content WITHOUT the $$ delimiters.
 */
function extractDisplayFormulas(text: string): string[] {
  const formulas: string[] = [];
  const re = /\$\$([\s\S]+?)\$\$/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    formulas.push(m[1].trim());
  }
  return formulas;
}

/**
 * Extract inline math formulas ($...$) — fallback when no display math exists.
 * Returns the LaTeX content WITHOUT the $ delimiters.
 */
function extractInlineFormulas(text: string): string[] {
  const formulas: string[] = [];
  const re = /(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    formulas.push(m[1].trim());
  }
  return formulas;
}

/**
 * Find all citation reference numbers [N] in text and return them.
 * Useful for cross-referencing with the Citation[] array.
 */
export function extractCitationRefs(text: string): number[] {
  const refs = new Set<number>();
  const re = /\[(\d+)\]/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    refs.add(parseInt(m[1], 10));
  }
  return [...refs].sort((a, b) => a - b);
}
