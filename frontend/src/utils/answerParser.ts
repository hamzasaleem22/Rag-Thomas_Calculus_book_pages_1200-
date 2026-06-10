import type { AnswerSection, ParsedAnswer } from "../types";
import { sanitizeTextWithMath } from "./latexSanitizer";

export function parseAnswer(text: string): ParsedAnswer {
  if (!text || !text.trim()) {
    return { summary: "", keyPoints: [], formulas: [], sections: [], rawText: text ?? "" };
  }

  const sanitized = sanitizeTextWithMath(text);
  const normalized = normalizeMathText(sanitized);

  const sections = extractSections(normalized);

  if (sections.length === 0) {
    return {
      summary: normalized.trim(),
      keyPoints: [],
      formulas: [],
      sections: [{ header: "Answer", content: normalized.trim() }],
      rawText: text,
    };
  }

  const summary = sections[0]?.content ?? "";
  const keyPoints = extractBulletPoints(sections.map((s) => s.content).join("\n"));
  const formulas = collectFormulas(normalized);
  const rawText = text;

  return { summary, keyPoints, formulas, sections, rawText };
}

function extractSections(text: string): AnswerSection[] {
  const lines = text.split("\n");
  const sections: AnswerSection[] = [];
  let currentHeader = "";
  let currentContent: string[] = [];
  let foundHeader = false;

  for (const line of lines) {
    const match = line.match(/^\*\*([^*]+)\*\*:\s*$/);
    if (match) {
      if (foundHeader) {
        sections.push({ header: currentHeader, content: currentContent.join("\n").trim() });
      }
      currentHeader = match[1].trim();
      currentContent = [];
      foundHeader = true;
    } else if (foundHeader) {
      currentContent.push(line);
    }
  }

  if (foundHeader) {
    sections.push({ header: currentHeader, content: currentContent.join("\n").trim() });
  }

  return sections;
}

function collectFormulas(text: string): string[] {
  const formulas: string[] = [];
  const re = /\$\$([\s\S]+?)\$\$/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    formulas.push(m[1].trim());
  }
  if (formulas.length === 0) {
    const inlineRe = /(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)/g;
    while ((m = inlineRe.exec(text)) !== null) {
      formulas.push(m[1].trim());
    }
  }
  return [...new Set(formulas.map((f) => f.trim()))].filter(Boolean);
}

export function normalizeMathText(text: string): string {
  let result = text;

  const placeholders: string[] = [];
  result = result.replace(/\$\$([\s\S]+?)\$\$/g, (match) => {
    placeholders.push(match);
    return `%%MATH_BLOCK_${placeholders.length - 1}%%`;
  });
  result = result.replace(/(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)/g, (match) => {
    placeholders.push(match);
    return `%%MATH_INLINE_${placeholders.length - 1}%%`;
  });

  result = result.replace(/d(\^?\d*)\/d([a-z])/g, (_match, power, varName) => {
    const pow = power === "^2" ? "^2" : power === "²" ? "^2" : "";
    return `\\frac{d${pow}}{d${varName}}`;
  });
  result = result.replace(/\\frac\{d\}\{d([a-z])\}\[/g, "\\frac{d}{d$1}\\left[");
  result = result.replace(/\](\s*=)/g, "\\right]$1");
  result = result.replace(/(?<!\$)∫(?!\$)/g, "\\int");
  result = result.replace(/(?<!\$)Σ(?!\$)/g, "\\sum");
  result = result.replace(/(?<![\\a-zA-Z])π(?![a-zA-Z])/g, "\\pi");
  result = result.replace(/(?<![\\a-zA-Z])∞(?![a-zA-Z])/g, "\\infty");

  const mathFns = [
    "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan",
    "sin", "cos", "tan", "sec", "csc", "cot",
    "ln", "log", "exp", "lim",
  ];
  for (const fn of mathFns) {
    const fnRe = new RegExp(`(?<![\\\\a-zA-Z])${fn}(?=[\\s(\\[])`, "g");
    result = result.replace(fnRe, `\\${fn}`);
  }

  result = result.replace(/([a-zA-Z0-9)])²/g, "$1^2");
  result = result.replace(/([a-zA-Z0-9)])³/g, "$1^3");
  result = result.replace(/(?<!\\)sqrt\(([^)]+)\)/gi, "\\sqrt{$1}");

  result = wrapOrphanedLatex(result);

  result = result.replace(/%%MATH_BLOCK_(\d+)%%/g, (_, idx) => placeholders[parseInt(idx)]);
  result = result.replace(/%%MATH_INLINE_(\d+)%%/g, (_, idx) => placeholders[parseInt(idx)]);

  return result;
}

function wrapOrphanedLatex(text: string): string {
  const latexCmdRe = /(?:\\frac|\\int|\\sum|\\sqrt|\\lim|\\sin|\\cos|\\tan|\\ln|\\log|\\exp|\\pi|\\infty|\\to|\\cdot|\\left|\\right)/;
  const lines = text.split("\n");
  const result: string[] = [];
  for (const line of lines) {
    if (latexCmdRe.test(line) && !line.includes("$")) {
      const match = line.match(/^([\s]*)([\s\S]*)$/);
      if (match) {
        const [, leading, content] = match;
        result.push(`${leading}$${content}$`);
        continue;
      }
    }
    result.push(line);
  }
  return result.join("\n");
}

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

export function extractCitationRefs(text: string): number[] {
  const refs = new Set<number>();
  const re = /\[(\d+)\]/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    refs.add(parseInt(m[1], 10));
  }
  return [...refs].sort((a, b) => a - b);
}
