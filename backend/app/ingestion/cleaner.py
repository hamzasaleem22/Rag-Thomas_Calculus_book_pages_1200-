import re

from langchain_core.documents import Document
from app.retrieval.math_cleaner import clean_chunk as math_clean_chunk


POUND_SIGN = re.compile(r"#")
UNI00A0 = re.compile(r"/?uni00A0")
MULTI_NEWLINE = re.compile(r"\n{3,}")
EXTRA_SPACES = re.compile(r"[ \t]{2,}")
LEADING_WS = re.compile(r"^[ \t]+", re.MULTILINE)

UNICODE_ARTIFACTS = {
    r"/uni2032": "'",
    r"/uni2033": "''",
    r"/uni2260\.alt10": "≠",
    r"/uni2260\.alt1a": "≠",
    r"/uni2260": "≠",
    r"/uni2212\.boldH": "−",
    r"/uni2212": "−",
    r"/u2202": "∂",
    r"/uni00A0": " ",
    r"/uni2192": "→",
    r"/uni2264": "≤",
    r"/uni2265": "≥",
    r"/uni222B": "∫",
    r"/uni03B1": "α",
    r"/uni03B2": "β",
    r"/uni03B8": "θ",
    r"/uni03C0": "π",
    r"/uni03A3": "Σ",
    r"/uni03A0": "Π",
    r"/uni2207": "∇",
    r"/uni221E": "∞",
    r"/uni2248": "≈",
    r"/uni00D7": "×",
    r"/uni00F7": "÷",
    r"/uni00B1": "±",
    r"/uni221A": "√",
    r"/uni22C5": "⋅",
    r"/\.alt\d+": "",
}
UNICODE_PATTERNS = [
    (re.compile(p, re.IGNORECASE), r)
    for p, r in UNICODE_ARTIFACTS.items()
]


def clean_text(text: str) -> str:
    text = POUND_SIGN.sub("", text)
    text = UNI00A0.sub(" ", text)
    text = MULTI_NEWLINE.sub("\n\n", text)
    text = EXTRA_SPACES.sub(" ", text)
    text = LEADING_WS.sub("", text)
    return text.strip()


def fix_unicode_artifacts(text: str) -> str:
    for pattern, replacement in UNICODE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


NORMALIZE_LATEX = {
    r"\\\[": "$$",
    r"\\\]": "$$",
    r"\\\(": "$",
    r"\\\)": "$",
}

LATEX_REPLACEMENTS = [
    (re.compile(pattern), replacement)
    for pattern, replacement in NORMALIZE_LATEX.items()
]


def normalize_latex(text: str) -> str:
    for pattern, replacement in LATEX_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text


FOOTER_PATTERNS = [
    re.compile(r"^\d+\s*$", re.MULTILINE),
    re.compile(r"^Thomas' Calculus.*$", re.MULTILINE),
]


def remove_footers(text: str) -> str:
    for pattern in FOOTER_PATTERNS:
        text = pattern.sub("", text)
    return text


GARBLED_LINES = re.compile(r"^[A-Z]\d{2}_[A-Z]+.*\.indd.*$", re.MULTILINE)


def remove_garbled_lines(text: str) -> str:
    return GARBLED_LINES.sub("", text)


def clean_document(doc: Document) -> Document:
    text = doc.page_content
    text = remove_footers(text)
    text = remove_garbled_lines(text)
    text = normalize_latex(text)
    text = fix_unicode_artifacts(text)
    text = math_clean_chunk(text)
    text = clean_text(text)
    doc.page_content = text
    return doc


def clean_documents(docs: list[Document]) -> list[Document]:
    return [clean_document(doc) for doc in docs]
