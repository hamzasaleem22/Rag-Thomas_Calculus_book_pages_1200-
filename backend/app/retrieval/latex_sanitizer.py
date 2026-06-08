"""
latex_sanitizer.py
──────────────────
Post-processes LLM-generated answer text to fix common LaTeX malformation.

The LLM sometimes outputs:
  - Bare \\frac, \\sum, etc. without $...$ delimiters
  - \\fracf instead of \\frac{f} (missing braces)
  - d¸ots instead of \\dots (garbled commands)
  - Duplicate formulas (raw LaTeX + rendered Unicode side by side)

This module cleans these patterns before the answer reaches the frontend.
"""

import re


def sanitize_answer(text: str) -> str:
    """Fix malformed LaTeX and wrap bare math in $...$ delimiters."""
    if not text:
        return text

    result = text

    # ── 1. Fix garbled LaTeX commands ──────────────────────────────────
    # d¸ots, d¸ ots → \dots
    result = re.sub(r'd¸\s*ots', r'\\dots', result)
    # ¸o → o (stray cedilla)
    result = result.replace('¸', '')

    # \to without backslash in some contexts
    result = re.sub(r'(?<!\\)(\\to)(?!\w)', r'\\to', result)

    # Fix \infty without backslash: "infty" → "\infty" (only when standalone)
    result = re.sub(r'(?<!\\)(?<![a-zA-Z])infty(?![a-zA-Z])', r'\\infty', result)

    # ── 2. Fix \frac without braces ────────────────────────────────────
    # \fracf → \frac{f}  (single char numerator)
    result = re.sub(r'\\frac([a-zA-Z])(?!\{)', r'\\frac{\1}', result)
    # \frac2 → \frac{2}  (single digit numerator)
    result = re.sub(r'\\frac(\d)(?!\{)', r'\\frac{\1}', result)

    # \frac{...} followed by non-brace → wrap next token in braces
    # e.g. \frac{a}b → \frac{a}{b}
    result = re.sub(r'\\frac\{([^}]+)\}([a-zA-Z0-9])', r'\\frac{\1}{\2}', result)

    # ── 3. Fix \sum, \prod, \int without proper subscript/superscript ──
    # \sum k=0 \infty → \sum_{k=0}^{\infty}
    # This is a heuristic — catches common patterns
    result = re.sub(
        r'\\sum\s+(\w+)\s*=\s*(\d+)\s+\\infty',
        r'\\sum_{\1=\2}^{\\infty}',
        result,
    )
    result = re.sub(
        r'\\sum\s+\{?(\w+)\s*=\s*(\d+)\}?\s*\{?\\infty\}?',
        r'\\sum_{\1=\2}^{\\infty}',
        result,
    )

    # ── 4. Fix f^{(k)} patterns ────────────────────────────────────────
    # f (k) → f^{(k)}  when it looks like a derivative notation
    result = re.sub(r'f\s+\((\w)\)', r'f^{(\1)}', result)
    # f ' → f' (stray space)
    result = re.sub(r"f\s+'", "f'", result)
    result = re.sub(r"f\s+''", "f''", result)

    # ── 5. Remove duplicate Unicode-rendered formulas ──────────────────
    # The LLM sometimes outputs BOTH the LaTeX and a Unicode fallback.
    # Detect lines that are near-duplicates and remove the Unicode version.
    # This is handled by detecting lines with Unicode math symbols (∑, ∫, ∞)
    # that appear right after a line with the equivalent LaTeX.
    result = _remove_unicode_duplicates(result)

    # ── 6. Wrap bare LaTeX blocks in $...$ ────────────────────────────
    result = _wrap_bare_latex(result)

    # ── 7. Clean up excessive whitespace ───────────────────────────────
    result = re.sub(r'\n{3,}', '\n\n', result)
    result = re.sub(r'[^\S\n]+', ' ', result)

    return result


def _remove_unicode_duplicates(text: str) -> str:
    """Remove lines that are Unicode-rendered versions of a preceding LaTeX line."""
    lines = text.split('\n')
    result = []
    skip_next = False

    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        # Check if this line contains Unicode math duplicates of LaTeX above
        # Unicode math indicators: ∑, ∫, ∞, ≤, ≥, ≠, → used alongside
        # the same formula in LaTeX on the previous line
        unicode_math_chars = sum(1 for c in line if c in '∑∫∞≤≥≠→√∂∇')

        if unicode_math_chars >= 2 and i > 0:
            prev = result[-1] if result else ""
            # If previous line has LaTeX commands and this line has similar
            # content in Unicode, skip this line
            if '\\' in prev and len(line) > 5:
                # Rough similarity: check if key terms overlap
                prev_terms = set(re.findall(r'[a-zA-Z]+', prev))
                curr_terms = set(re.findall(r'[a-zA-Z]+', line))
                overlap = len(prev_terms & curr_terms)
                if overlap >= 2 and len(curr_terms) > 0:
                    overlap_ratio = overlap / len(curr_terms)
                    if overlap_ratio > 0.5:
                        continue  # Skip this Unicode duplicate

        result.append(line)

    return '\n'.join(result)


def _wrap_bare_latex(text: str) -> str:
    """Find lines that are entirely bare LaTeX and wrap them in $$...$$.

    Only wraps whole lines that are clearly standalone math expressions
    (mostly LaTeX commands, few English words). Does NOT wrap individual
    LaTeX commands within prose lines — that causes malformed output.
    """
    latex_cmd_re = re.compile(
        r'\\(?:frac|sum|prod|int|lim|sqrt|sin|cos|tan|ln|log|exp|left|right|'
        r'begin|end|cdot|ldots|cdots|dots|to|infty|pi|alpha|beta|gamma|delta|'
        r'theta|lambda|mu|sigma|phi|psi|omega|partial|nabla|forall|exists|'
        r'in|notin|subset|cup|cap|times|div|pm|mp|le|ge|ne|approx|equiv|sim)\b'
    )

    lines = text.split('\n')
    result = []

    for line in lines:
        # Skip lines that already have $ delimiters
        if '$' in line:
            result.append(line)
            continue

        # Count LaTeX commands in this line
        latex_matches = latex_cmd_re.findall(line)

        # Only wrap if the line has 2+ LaTeX commands (clearly a formula line)
        # AND is mostly math (few English words)
        if len(latex_matches) >= 2:
            # Count English words (3+ letter sequences that aren't LaTeX)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', line)
            # Filter out LaTeX-related words
            non_latex_words = [w for w in words if w.lower() not in
                {'sin', 'cos', 'tan', 'log', 'lim', 'exp', 'frac', 'sum',
                 'int', 'sqrt', 'dots', 'infty', 'pi', 'alpha', 'beta',
                 'gamma', 'delta', 'theta', 'lambda', 'left', 'right'}]

            # If mostly math (<=2 non-LaTeX English words), wrap it
            if len(non_latex_words) <= 2:
                stripped = line.strip()
                if stripped:
                    result.append(f"$${stripped}$$")
                    continue

        result.append(line)

    return '\n'.join(result)
