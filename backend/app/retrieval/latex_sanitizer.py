"""
latex_sanitizer.py
──────────────────
Post-processes LLM-generated answer text to fix common LaTeX malformation.

The LLM sometimes outputs:
  - Bare \\frac, \\sum, etc. without $...$ delimiters
  - \\fracf instead of \\frac{f} (missing braces)
  - d¸ots instead of \\dots (garbled commands)
  - Duplicate formulas (raw LaTeX + rendered Unicode side by side)
  - Unpaired $ delimiters
  - Missing braces on superscripts

This module cleans these patterns before the answer reaches the frontend.
"""

import re


# Complete list of LaTeX commands for bare-LaTeX detection
_LATEX_COMMANDS = [
    r'\\frac', r'\\sum', r'\\prod', r'\\int', r'\\iint', r'\\iiint', r'\\oint',
    r'\\lim', r'\\sqrt', r'\\sin', r'\\cos', r'\\tan', r'\\ln', r'\\log',
    r'\\exp', r'\\left', r'\\right', r'\\begin', r'\\end', r'\\cdot',
    r'\\ldots', r'\\cdots', r'\\dots', r'\\to', r'\\infty', r'\\pi',
    r'\\alpha', r'\\beta', r'\\gamma', r'\\delta', r'\\theta', r'\\lambda',
    r'\\mu', r'\\sigma', r'\\phi', r'\\psi', r'\\omega', r'\\partial',
    r'\\nabla', r'\\forall', r'\\exists', r'\\in', r'\\notin', r'\\subset',
    r'\\cup', r'\\cap', r'\\times', r'\\div', r'\\pm', r'\\mp', r'\\le',
    r'\\ge', r'\\ne', r'\\approx', r'\\equiv', r'\\sim', r'\\rightarrow',
    r'\\leftarrow', r'\\Rightarrow', r'\\Leftarrow', r'\\mapsto',
    r'\\mathbf', r'\\overrightarrow', r'\\vec', r'\\hat', r'\\tilde',
    r'\\bar', r'\\overline', r'\\underline', r'\\circ', r'\\propto',
    r'\\implies', r'\\iff', r'\\hookrightarrow', r'\\hookleftarrow',
    r'\\uparrow', r'\\downarrow', r'\\updownarrow', r'\\langle', r'\\rangle',
    r'\\lfloor', r'\\rfloor', r'\\lceil', r'\\rceil', r'\\mathcal',
    r'\\mathbb', r'\\mathrm', r'\\textbf', r'\\textit', r'\\text',
    r'\\quad', r'\\qquad', r'\\colon', r'\\cup', r'\\cap', r'\\setminus',
]

_LATEX_CMD_RE = re.compile(r'(?:' + '|'.join(_LATEX_COMMANDS) + r')\b')

_MATH_FUNCTIONS = {
    'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan',
    'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch',
    'log', 'ln', 'lg', 'exp', 'det', 'dim', 'ker', 'hom', 'tr', 'deg',
    'lim', 'limsup', 'liminf', 'sup', 'max', 'min', 'inf', 'arg',
    'cdot', 'dots', 'cdots', 'ldots', 'to', 'infty', 'pi',
    'alpha', 'beta', 'gamma', 'delta', 'theta', 'lambda',
    'nabla', 'partial', 'times', 'div', 'circ', 'propto',
}

# Differential tokens used in \frac numerator/denominator
_DIFF_TOKENS = {
    'dx', 'dy', 'du', 'dv', 'dt', 'dr', 'ds', 'dz',
    'dθ', 'dφ', 'dψ',
}


def sanitize_answer(text: str) -> str:
    """Fix malformed LaTeX and wrap bare math in $...$ delimiters."""
    if not text:
        return text

    result = text

    result = _fix_double_backslash(result)
    result = _fix_garbled_commands(result)
    result = _fix_spacing_primitives(result)
    result = _fix_frac_multichar(result)
    result = _fix_frac_braces(result)
    result = _fix_frac_with_superscript(result)
    result = _fix_missing_frac_denominator(result)
    result = _fix_sum_limits(result)
    result = _fix_missing_op_space(result)
    result = _fix_derivative_notation(result)
    result = _fix_superscript_braces(result)
    result = _fix_missing_backslash(result)
    result = _fix_missing_cdot(result)
    result = _fix_bare_cmd_outside_math(result)
    result = _remove_unicode_duplicates(result)
    result = _fix_unpaired_dollars(result)
    result = _wrap_bare_latex(result)
    result = _cleanup_whitespace(result)

    return result


def _fix_spacing_primitives(text: str) -> str:
    """Remove or convert TeX spacing primitives that are implementation details.

    These commands (\\tmspace, \\kern, \\hskip, \\vskip, \\mskip, \\mkern) are
    low-level spacing directives that should never appear in user-facing output.
    They originate from the LLM's training data and are always artifacts.
    """
    result = text

    # 1. \\tmspace — thin/medium/thick/negative thin space implementation detail
    # Patterns:
    #   \\tmspace + 3mu.1667em   → \\,
    #   \\tmspace-{5mu}{.277em}  → \\;
    #   \\tmspace- {5mu}{.277em} → \\;
    #   \\tmspace  {3mu}         → \\,
    # Strip the entire \\tmspace expression including trailing brace-groups
    result = re.sub(
        r'\\tmspace\s*[+-]?\s*(?:\d*\.?\d*mu\.?\d*em[a-z]*|\{[^}]*\}\s*\{[^}]*\}|\{[^}]*\})',
        r'\\,',
        result,
    )
    # Catch any remaining bare \\tmspace references
    result = re.sub(r'\\tmspace', r'\\,', result)

    # 2. \\kern — explicit kern with dimension
    # \\kern 3.0pt  \\kern -2em  \\kern 1ex
    result = re.sub(
        r'\\kern\s*[+-]?\s*[\d.]+\s*(pt|pc|in|bp|cm|mm|em|ex|mu)',
        r' ',
        result,
    )

    # 3. \\hskip, \\vskip, \\mskip, \\mkern — with or without dimension
    # These all specify explicit spacing that should be rendered as normal space
    result = re.sub(
        r'\\(?:hskip|vskip|mskip|mkern)\s*[+-]?\s*[\d.]*\s*(pt|pc|in|bp|cm|mm|em|ex|mu)?',
        r' ',
        result,
    )

    # Collapse multiple spaces that may result from the above replacements
    result = re.sub(r'  +', ' ', result)

    return result


def _fix_missing_frac_denominator(text: str) -> str:
    """Fix \\frac where the denominator expression is missing braces.

    Pattern: \\frac{(numerator)}k!  →  \\frac{(numerator)}{k!}
    Pattern: \\frac{(numerator)}x!  →  \\frac{(numerator)}{x!}
    Pattern: \\frac{(numerator)}n   →  \\frac{(numerator)}{n}   (if followed by end or non-word)
    """
    result = text

    # After a braced numerator, capture single letter + ! as denominator
    result = re.sub(
        r'\\frac\{([^}]+)\}([a-zA-Z])!',
        r'\\frac{\1}{\2!}',
        result,
    )

    # After a braced numerator, capture single digit + ! as denominator
    result = re.sub(
        r'\\frac\{([^}]+)\}(\d)!',
        r'\\frac{\1}{\2!}',
        result,
    )

    # Fallback: after a braced numerator, a single letter/digit followed by
    # space, end-of-string, or non-alphanumeric may be the denominator
    result = re.sub(
        r'\\frac\{([^}]+)\}([a-zA-Z0-9])(\s|$|\.|,|;|!|\\|\))',
        r'\\frac{\1}{\2}\3',
        result,
    )

    return result


def _fix_frac_with_superscript(text: str) -> str:
    """Fix \\frac where numerator has superscript but denominator is missing braces.

    Pattern: \\frac{f}^{(n)}2!  →  \\frac{f^{(n)}}{2!}
    Pattern: \\frac{x}^2 + 1    →  \\frac{x^2}{+1}
    
    Handles cases where a superscript follows the numerator and
    additional content that should be the denominator follows.
    """
    result = text

    # Pattern: \frac{numerator}^{superscript}suffix
    # where suffix is digit(s) followed by ! or similar
    # Rewrite as: \frac{numerator^{superscript}}{suffix}
    result = re.sub(
        r'\\frac\{([^}]+)\}\^(\{[^}]*\}|\S)([0-9]+!)',
        lambda m: f'\\frac{{{m.group(1)}^{m.group(2)}}}{{{m.group(3)}}}',
        result,
    )

    # Pattern: \frac{numerator}^{superscript} followed by a single digit/letter
    # This is more ambiguous, so only do this if it's followed by ! or space/punc
    result = re.sub(
        r'\\frac\{([^}]+)\}\^(\{[^}]*\})\s*([a-zA-Z])!',
        lambda m: f'\\frac{{{m.group(1)}^{m.group(2)}}}{{{m.group(3)}!}}',
        result,
    )

    return result


def _fix_missing_op_space(text: str) -> str:
    """Fix missing space after operators like \\int, \\sum, \\prod before a letter.

    Pattern: \\intf(x)  →  \\int f(x)
    Pattern: \\sumk=1    →  \\sum k=1
    """
    result = text
    ops_pattern = r'\\(int|iint|iiint|oint|sum|prod)([a-zA-Z])'
    result = re.sub(ops_pattern, r'\\\1 \2', result)
    return result


def _fix_garbled_commands(text: str) -> str:
    """Fix garbled LaTeX commands from PDF extraction."""
    result = text
    result = re.sub(r'd¸\s*ots', r'\\dots', result)
    result = result.replace('¸', '')
    result = re.sub(r'(?<!\\)(\\to)(?!\w)', r'\\to', result)
    result = re.sub(r'(?<!\\)(?<![a-zA-Z])infty(?![a-zA-Z])', r'\\infty', result)
    return result


def _fix_frac_multichar(text: str) -> str:
    """Fix \\frac with multi-char numerator/denominator and no braces.
    
    Handles differential notation like \\fracdydx → \\frac{dy}{dx}.
    Must run before _fix_frac_braces to intercept multi-char patterns first.
    """
    result = text

    # Pattern: \frac + two differential tokens (dx, dy, du, dv, dt, etc.)
    # e.g., \fracdydx → \frac{dy}{dx}, \fracdudv → \frac{du}{dv}
    result = re.sub(
        r'\\frac(' + '|'.join(sorted(_DIFF_TOKENS, key=len, reverse=True)) + r')'
        r'(' + '|'.join(sorted(_DIFF_TOKENS, key=len, reverse=True)) + r')',
        lambda m: f'\\frac{{{m.group(1)}}}{{{m.group(2)}}}',
        result,
    )

    # Broader pattern: \frac + two groups of 2-4 lowercase letters
    # Only applies when both groups look like common math variables
    # Examples: \fracsinx → \frac{sin}{x}, \frac{ab}{cd}
    result = re.sub(
        r'\\frac([a-z]{2})([a-z]{2})',
        lambda m: f'\\frac{{{m.group(1)}}}{{{m.group(2)}}}',
        result,
    )

    return result


def _fix_missing_cdot(text: str) -> str:
    r"""Fix 'dot' or Unicode ⋅/· used in place of \cdot in math contexts."""
    result = text
    # Unicode ⋅ (U+22C5 DOT OPERATOR) → \cdot
    result = result.replace('\u22C5', '\\cdot')
    # Unicode · (U+00B7 MIDDLE DOT) → \cdot
    result = result.replace('\u00B7', '\\cdot')
    # Between two math expressions (curly braces or parens): } dot { → } \cdot {
    result = re.sub(r'}([.\s]*)dot([\s]*){', r'}\\cdot{', result, flags=re.IGNORECASE)
    # After a closing brace before a new frac: }dot\frac → }\cdot\frac
    result = re.sub(r'}([.\s]*)dot([\s]*)\\frac', r'}\\cdot\\frac', result, flags=re.IGNORECASE)
    return result


def _fix_frac_braces(text: str) -> str:
    """Fix \\frac without braces."""
    result = text
    result = re.sub(r'\\frac([a-zA-Z])(?!\{)', r'\\frac{\1}', result)
    result = re.sub(r'\\frac(\d)(?!\{)', r'\\frac{\1}', result)
    result = re.sub(r'\\frac\{([^}]+)\}([a-zA-Z0-9])', r'\\frac{\1}{\2}', result)
    return result


def _fix_sum_limits(text: str) -> str:
    """Fix \\sum, \\prod, \\int without proper subscript/superscript."""
    result = text
    
    # Pattern 1: \sum k=0 \infty (with spaces)
    result = re.sub(
        r'\\sum\s+(\w+)\s*=\s*(\d+)\s+\\infty',
        r'\\sum_{\1=\2}^{\\infty}',
        result,
    )
    
    # Pattern 2: \sum with optional braces around k=0 and \infty
    result = re.sub(
        r'\\sum\s+\{?(\w+)\s*=\s*(\d+)\}?\s*\{?\\infty\}?',
        r'\\sum_{\1=\2}^{\\infty}',
        result,
    )
    
    # Pattern 3: \sum_infty or \sum _infty (missing k=0, assume k=0)
    result = re.sub(r'\\sum\s*_\s*infty', r'\\sum_{k=0}^{\\infty}', result)
    result = re.sub(r'\\sum\s+infty(?![a-zA-Z])', r'\\sum_{k=0}^{\\infty}', result)
    
    # Pattern 4: Fix bare \infty to \\infty in sum context
    result = re.sub(r'\\sum_\{([^}]+)\}\s*infty', r'\\sum_{\1}^{\\infty}', result)
    
    return result


def _fix_derivative_notation(text: str) -> str:
    """Fix f (k) → f^{(k)} and related derivative patterns."""
    result = text
    result = re.sub(r'f\s+\((\w)\)', r'f^{(\1)}', result)
    result = re.sub(r"f\s+'", "f'", result)
    result = re.sub(r"f\s+''", "f''", result)
    return result


def _fix_superscript_braces(text: str) -> str:
    """Ensure superscripts with multiple chars have braces: x^2y → x^{2}y, f^(k)(x) → f^{(k)}(x)"""
    result = text
    result = re.sub(r'f\^\((\w)\)', r'f^{(\1)}', result)
    result = re.sub(r'(\w)\^(\w{2,})(?!\})', r'\1^{\2}', result)
    return result


def _fix_missing_backslash(text: str) -> str:
    """Add backslashes to known math functions used without them inside math contexts."""
    result = text

    def _fix_math_fn(match):
        prefix = match.group(1)
        name = match.group(2)
        if name in _MATH_FUNCTIONS:
            if prefix.endswith('\\'):
                return match.group(0)
            return prefix + '\\' + name
        return match.group(0)

    result = re.sub(r'(\$[^$]*?)\b(' + '|'.join(sorted(_MATH_FUNCTIONS, key=len, reverse=True)) + r')\b', _fix_math_fn, result)
    return result


def _fix_double_backslash(text: str) -> str:
    r"""Fix \\cdot, \\to, \\pi etc. -> \cdot, \to, \pi (single backslash).

    The LLM sometimes doubles the backslash in display math: \\cdot.
    KaTeX interprets \\ as a line break, breaking the formula.
    """
    result = text
    result = re.sub(
        r'\\\\(cdot|to|infty|pi|alpha|beta|gamma|delta|theta|lambda|mu|sigma|phi|psi|omega|nabla|partial|times|div|circ|dots|cdots|ldots|sqrt|frac|int|sum|prod|lim|sin|cos|tan|ln|log|exp|left|right|text|mathbf|mathrm|mathcal|mathbb|overrightarrow|rightarrow|Rightarrow|mapsto|implies|iff)',
        r'\\\1',
        result,
    )
    return result


def _fix_bare_cmd_outside_math(text: str) -> str:
    """Wrap bare LaTeX commands like \\to that appear outside $...$ delimiters.

    The LLM sometimes outputs \\to in running text (not inside $...$):
        "with respect \\to $x$"  ->  "with respect $\\to$ $x$"

    This function tracks math mode ($ and $$) and wraps known math-only
    commands when they appear outside it.
    """
    _BARE_CMDS = {
        'to', 'cdot', 'pi', 'infty', 'alpha', 'beta', 'gamma', 'delta',
        'theta', 'lambda', 'mu', 'sigma', 'phi', 'psi', 'omega',
        'nabla', 'partial', 'times', 'div', 'circ', 'approx', 'equiv',
        'sim', 'propto', 'le', 'ge', 'ne', 'pm', 'mp',
    }
    _BARE_PATTERN = re.compile(
        r'\\(' + '|'.join(sorted(_BARE_CMDS, key=len, reverse=True)) + r')\b'
    )

    lines = text.split('\n')
    result = []

    for line in lines:
        if '\\' not in line:
            result.append(line)
            continue

        new_line = []
        i = 0
        in_math = False

        while i < len(line):
            ch = line[i]

            # Toggle math mode for $$ and $
            if line[i:i+2] == '$$':
                in_math = not in_math
                new_line.append('$$')
                i += 2
                continue
            if ch == '$':
                in_math = not in_math
                new_line.append('$')
                i += 1
                continue

            if not in_math and ch == '\\':
                m = _BARE_PATTERN.match(line, i)
                if m:
                    new_line.append('$' + m.group(0) + '$')
                    i += len(m.group(0))
                    continue

            new_line.append(ch)
            i += 1

        result.append(''.join(new_line))

    return '\n'.join(result)


def _remove_unicode_duplicates(text: str) -> str:
    """Remove lines that are Unicode-rendered versions of a preceding LaTeX line."""
    lines = text.split('\n')
    result = []
    skip_next = False

    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        unicode_math_chars = sum(1 for c in line if c in '∑∫∞≤≥≠→√∂∇')

        if unicode_math_chars >= 2 and i > 0:
            prev = result[-1] if result else ""
            if '\\' in prev and len(line) > 5:
                prev_terms = set(re.findall(r'[a-zA-Z]+', prev))
                curr_terms = set(re.findall(r'[a-zA-Z]+', line))
                overlap = len(prev_terms & curr_terms)
                if overlap >= 2 and len(curr_terms) > 0:
                    overlap_ratio = overlap / len(curr_terms)
                    if overlap_ratio > 0.5:
                        continue

        result.append(line)

    return '\n'.join(result)


def _fix_unpaired_dollars(text: str) -> str:
    """Fix unpaired $ delimiters by ensuring balanced pairs."""
    lines = text.split('\n')
    result = []

    for line in lines:
        dollar_count = line.count('$')
        if dollar_count % 2 == 0:
            result.append(line)
            continue

        display_count = line.count('$$')
        if display_count == 1:
            dollar_count -= 2

        if dollar_count % 2 != 0:
            result.append(line)
            continue

        result.append(line)

    return '\n'.join(result)


def _wrap_bare_latex(text: str) -> str:
    """Find lines that are entirely bare LaTeX and wrap them in $$...$$."""
    lines = text.split('\n')
    result = []

    for line in lines:
        # Skip lines that already have math delimiters
        if '$' in line or '\\[' in line or '\\]' in line:
            result.append(line)
            continue

        latex_matches = _LATEX_CMD_RE.findall(line)

        # More aggressive: wrap if ANY LaTeX command found + line looks math-heavy
        if len(latex_matches) >= 1:
            # Check if line is mostly LaTeX (backslashes, braces, parens)
            latex_density = len([c for c in line if c in '\\{}()^_']) / max(len(line), 1)
            
            # If ≥40% of characters are LaTeX syntax, wrap it
            if latex_density >= 0.4:
                stripped = line.strip()
                if stripped and not stripped.endswith('$$'):
                    result.append(f"$${stripped}$$")
                    continue
            
            # Also wrap if has 2+ LaTeX commands regardless of density
            if len(latex_matches) >= 2:
                words = re.findall(r'\b[a-zA-Z]{3,}\b', line)
                non_latex_words = [
                    w for w in words
                    if w.lower() not in {
                        'sin', 'cos', 'tan', 'log', 'lim', 'exp', 'frac', 'sum',
                        'int', 'sqrt', 'dots', 'infty', 'pi', 'alpha', 'beta',
                        'gamma', 'delta', 'theta', 'lambda', 'left', 'right',
                        'nabla', 'partial', 'times', 'cdot', 'circ', 'propto',
                        'mathbf', 'overrightarrow', 'rightarrow', 'Rightarrow',
                        'mapsto', 'implies', 'iff', 'langle', 'rangle',
                        'mathcal', 'mathbb', 'mathrm', 'textbf', 'textit',
                        'prod', 'oint', 'iint', 'iiint', 'cdots', 'ldots',
                    }
                ]

                # Lower threshold: wrap if <= 3 non-LaTeX words (was 2)
                if len(non_latex_words) <= 3:
                    stripped = line.strip()
                    if stripped and not stripped.endswith('$$'):
                        result.append(f"$${stripped}$$")
                        continue

        result.append(line)

    return '\n'.join(result)


def _cleanup_whitespace(text: str) -> str:
    """Clean up excessive whitespace."""
    result = text
    result = re.sub(r'\n{3,}', '\n\n', result)
    result = re.sub(r'[^\S\n]+', ' ', result)
    return result
