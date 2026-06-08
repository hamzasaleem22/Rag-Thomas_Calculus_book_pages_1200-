import re

from langchain_core.documents import Document


POUND_SIGN = re.compile(r"#")
UNI00A0 = re.compile(r"/?uni00A0")
MULTI_NEWLINE = re.compile(r"\n{3,}")
EXTRA_SPACES = re.compile(r"[ \t]{2,}")
LEADING_WS = re.compile(r"^[ \t]+", re.MULTILINE)


def clean_text(text: str) -> str:
    text = POUND_SIGN.sub("", text)
    text = UNI00A0.sub(" ", text)
    text = MULTI_NEWLINE.sub("\n\n", text)
    text = EXTRA_SPACES.sub(" ", text)
    text = LEADING_WS.sub("", text)
    return text.strip()


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


def clean_document(doc: Document) -> Document:
    text = doc.page_content
    text = remove_footers(text)
    text = normalize_latex(text)
    text = clean_text(text)
    doc.page_content = text
    return doc


def clean_documents(docs: list[Document]) -> list[Document]:
    return [clean_document(doc) for doc in docs]
