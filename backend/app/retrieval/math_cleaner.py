"""
math_cleaner.py
───────────────
Cleans garbled math from PDF-extracted chunks before sending to the LLM.
See also: app/ingestion/cleaner.py (overlapping Unicode artifact cleanup).

NOTE: This module runs on chunks BEFORE they reach the LLM. It must NOT
replace LaTeX commands with Unicode (e.g. backslash-infty to ∞) because the LLM
needs to see LaTeX to produce correct output. Such replacements are
handled by app/retrieval/latex_sanitizer.py on the LLM output side.

The source PDF was extracted with a lossy tool that produces:
  - /uniXXXX  Adobe glyph codes (e.g. /uni2206 = Δ)
  - Spaced-out function names: "l i m", "s i n", "c o s"
  - ƒ  (U+0192) instead of f
  - S  used as → (arrow / "approaches")
  - p  used as π in math contexts
  - Various /H20841, ≠.alt, .alt artifacts
  - Bare LaTeX fragments like \\infty without delimiters

This module converts these to readable math that the LLM can then
re-express in clean LaTeX for the frontend.
"""

import re


# ── Adobe glyph → Unicode/LaTeX mapping ──────────────────────────────────

_GLYPH_MAP = {
    "/uni00B0": "°",
    "/uni0393": "\\Gamma",
    "/uni0394": "\\Delta",
    "/uni2034": "'''",
    "/uni211D": "\\mathbb{R}",
    "/uni2205": "\\emptyset",
    "/uni2206": "\\Delta",
    "/uni2207": "\\nabla",
    "/uni2202": "\\partial",
    "/uni2209": "\\notin",
    "/uni220A": "\\in",
    "/uni2218": "\\circ",
    "/uni2219": "\\cdot",
    "/uni2220A": "\\angle",
    "/uni2223": "|",
    "/uni222A": "\\cup",
    "/uni222B": "\\int",
    "/uni222C": "\\iint",
    "/uni222D": "\\iiint",
    "/uni223C": "\\sim",
    "/uni27E8": "\\langle",
    "/uni27E9": "\\rangle",
    "/uni2A0F": "\\oint",
    "/uni00D7": "\\times",
}


def clean_chunk(text: str) -> str:
    """Clean garbled math from a single PDF-extracted text chunk.

    This does NOT produce perfect LaTeX — it makes the text readable enough
    that the LLM can understand the math and re-express it cleanly.
    """
    result = text

    # ── 1. Remove pure noise artifacts ──────────────────────────────────
    # /H20841, /H20840 etc. — PDF bookmark noise
    result = re.sub(r"/H\d+", "", result)
    # .alt11, .altx suffixes on symbols
    result = re.sub(r"\.alt\d+[a-z]?", "", result)
    # ≠.alt patterns
    result = result.replace("≠.alt", "≠")

    # ── 2. Replace /uniXXXX glyph codes ─────────────────────────────────
    # First handle codes with trailing alphanumeric (e.g. /uni2206ABC → ΔABC)
    for code, replacement in sorted(_GLYPH_MAP.items(), key=lambda x: -len(x[0])):
        # Match the code possibly followed by alphanumeric chars that are
        # actually part of the NEXT token (not the glyph code itself)
        result = result.replace(code, replacement)

    # Clean up any remaining /uni codes we don't have mapped
    # These are usually /uni003D... patterns (= sign + trailing hex garbage)
    result = re.sub(r"/uni003D[a-fA-F0-9]*", "=", result)
    result = re.sub(r"/uni[0-9A-Fa-f]+", "", result)  # Remove unmapped

    # ── 3. Fix spaced-out function names ────────────────────────────────
    # "l i m" → "lim", "s i n" → "sin", "c o s" → "cos", etc.
    _spaced_fns = {
        "l i m": "lim",
        "s i n": "sin",
        "c o s": "cos",
        "t a n": "tan",
        "s e c": "sec",
        "c s c": "csc",
        "c o t": "cot",
        "l n": "ln",
        "l o g": "log",
        "e x p": "exp",
    }
    for spaced, normal in _spaced_fns.items():
        result = result.replace(spaced, normal)

    # Fix "o r" used as logical OR in equations
    result = re.sub(r"\bo r\b(?=\s*\w)", " or ", result)

    # ── 4. Fix ƒ → f ───────────────────────────────────────────────────
    result = result.replace("ƒ", "f")

    # ── 5. Fix "S" used as arrow (→) ────────────────────────────────────
    # Pattern: letter/number S letter/number/{ → "→"
    # e.g. "x S c", "h S 0", "x S {q"
    result = re.sub(r"(\w)\s+S\s+(\w|\{)", r"\1 → \2", result)
    # Also "x S {q" pattern for ±∞
    result = result.replace("→ {q", "→ ±∞")
    result = result.replace("→ {∞", "→ ±∞")

    # ── 6. Fix "q" used as ∞ ───────────────────────────────────────────
    # Only in math contexts: after comma, in brackets, after →
    # Be careful — "q" is also a normal letter. Only fix in clear math contexts.
    result = re.sub(r"(?<=\s)q(?=\s*[),;\]\}])", "∞", result)
    result = re.sub(r"(?<=\[)q(?=\s*\))", "∞", result)

    # ── 7. Fix "p" used as π ────────────────────────────────────────────
    # Only in clear math contexts: p/2, pr, p·, (p)
    # This is risky — only fix very obvious patterns
    result = re.sub(r"\bp/2\b", "π/2", result)
    result = re.sub(r"\bp/4\b", "π/4", result)
    result = re.sub(r"\bp/3\b", "π/3", result)
    result = re.sub(r"\bp/6\b", "π/6", result)
    result = re.sub(r"\b2p\b", "2π", result)
    result = re.sub(r"\b3p\b", "3π", result)

    # ── 8. Fix "Ú" used as ≥ and "…" used as ≤ ────────────────────────
    result = result.replace(" Ú ", " ≥ ")
    result = result.replace(" … ", " ≤ ")

    # ── 9. Clean up extra whitespace ───────────────────────────────────
    # Collapse multiple spaces (but preserve newlines)
    result = re.sub(r"[^\S\n]+", " ", result)
    # Remove trailing spaces on each line
    result = "\n".join(line.rstrip() for line in result.split("\n"))

    return result


def clean_context_chunks(documents) -> list:
    """Clean a list of Document objects' page_content in-place.

    Args:
        documents: list of langchain Document objects

    Returns:
        The same list (modified in-place) for chaining.
    """
    for doc in documents:
        doc.page_content = clean_chunk(doc.page_content)
    return documents
