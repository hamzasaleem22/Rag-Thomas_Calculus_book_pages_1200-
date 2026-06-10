import re
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


SYMBOL_MAP = {
    "\u0391": "A", "\u0392": "B", "\u0393": r"\Gamma", "\u0394": r"\Delta",
    "\u0398": r"\Theta", "\u039B": r"\Lambda", "\u03A0": r"\Pi",
    "\u03A3": r"\Sigma", "\u03A6": r"\Phi", "\u03A8": r"\Psi",
    "\u03A9": r"\Omega",
    "\u03B1": r"\alpha", "\u03B2": r"\beta", "\u03B3": r"\gamma",
    "\u03B4": r"\delta", "\u03B5": r"\epsilon", "\u03B6": r"\zeta",
    "\u03B7": r"\eta", "\u03B8": r"\theta", "\u03BA": r"\kappa",
    "\u03BB": r"\lambda", "\u03BC": r"\mu", "\u03BD": r"\nu",
    "\u03BE": r"\xi", "\u03C0": r"\pi", "\u03C1": r"\rho",
    "\u03C2": r"\sigma", "\u03C3": r"\sigma", "\u03C4": r"\tau",
    "\u03C5": r"\upsilon", "\u03C6": r"\phi", "\u03C7": r"\chi",
    "\u03C8": r"\psi", "\u03C9": r"\omega",
    "\u2202": r"\partial", "\u2207": r"\nabla",
    "\u2211": r"\sum", "\u220F": r"\prod",
    "\u222B": r"\int", "\u222E": r"\oint",
    "\u221E": r"\infty", "\u2192": r"\to",
    "\u2264": r"\le", "\u2265": r"\ge",
    "\u2260": r"\ne", "\u2248": r"\approx",
    "\u00D7": r"\times", "\u00F7": r"\div",
    "\u00B1": r"\pm", "\u221A": r"\sqrt",
    "\u2225": r"\parallel", "\u22C5": r"\cdot",
}

SYMBOL_RE = re.compile("|".join(re.escape(k) for k in SYMBOL_MAP))


def _replace_unicode_math(text: str) -> str:
    return SYMBOL_RE.sub(lambda m: SYMBOL_MAP[m.group(0)], text)


def convert_pdf_to_markdown(pdf_path: str, output_dir: str) -> str:
    pdf_path = str(Path(pdf_path).resolve())
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text(extraction_mode="layout")
        text = _replace_unicode_math(text)
        pages.append((i + 1, text))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{Path(pdf_path).stem}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        for page_num, text in pages:
            f.write(f"<!-- Page {page_num} -->\n")
            f.write(text.strip() + "\n\n")
    return str(md_path)


def parse_marker_markdown(md_path: str) -> list[Document]:
    from app.ingestion.pdf_structure import classify_page

    md_path = Path(md_path)
    text = md_path.read_text(encoding="utf-8")

    PAGE_PATTERN = re.compile(r"<!-- Page (\d+) -->")
    parts = PAGE_PATTERN.split(text)
    docs = []

    for i in range(1, len(parts) - 1, 2):
        page_num = int(parts[i].strip())
        content = parts[i + 1].strip()
        if content:
            section = classify_page(page_num)
            docs.append(Document(
                page_content=content,
                metadata={
                    "page": page_num,
                    "source": str(md_path.name),
                    "file_path": str(md_path),
                    "section": section,
                }
            ))

    if not docs:
        docs.append(Document(
            page_content=text,
            metadata={"source": str(md_path.name), "file_path": str(md_path), "section": "unknown"}
        ))

    return docs


class MarkerDocumentLoader:
    def __init__(self, pdf_path: str, output_dir: str = "data/markdown"):
        self.pdf_path = pdf_path
        self.output_dir = output_dir

    def load(self) -> list[Document]:
        md_path = convert_pdf_to_markdown(self.pdf_path, self.output_dir)
        return parse_marker_markdown(md_path)
