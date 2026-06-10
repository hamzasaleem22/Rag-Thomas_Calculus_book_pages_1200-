import re
from typing import Optional

from langchain_core.documents import Document

from app.ingestion.pdf_structure import get_chapter_number, CHAPTER_START_PAGE

HAS_EQUATION = re.compile(r"\$\$|\$|\\\[|\\\(|\\begin\{equation|\\begin\{align|\\begin\{cases")
SECTION_NUM = re.compile(r"(?:Section\s+)?(?:CHAPTER\s+)?(\d+)\.(\d+)")
CHAPTER_NUM = re.compile(r"(?:Chapter\s+)(\d+)", re.IGNORECASE)


def extract_section_number(chapter: str, page_content: str) -> str:
    m = SECTION_NUM.search(page_content)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    m = CHAPTER_NUM.search(chapter)
    if m:
        return f"{m.group(1)}.0"
    return ""


def extract_chapter_number(chapter: str, page: int) -> int:
    if "Answers to Odd-Numbered Exercises" in chapter:
        return 0
    m = CHAPTER_NUM.search(chapter)
    if m:
        return int(m.group(1))
    return get_chapter_number(page)


def compute_equation_density(text: str) -> float:
    if not text.strip():
        return 0.0
    matches = HAS_EQUATION.findall(text)
    total_words = len(text.split())
    return round(len(matches) / max(total_words, 1), 4)


def extract_key_terms(text: str, max_terms: int = 10) -> list[str]:
    stopwords = {
        "the", "is", "at", "which", "what", "how", "do", "does", "a", "an",
        "and", "or", "of", "to", "for", "in", "on", "by", "with", "from", "its",
        "are", "be", "has", "have", "was", "were", "been", "being",
        "that", "this", "these", "those", "it", "we", "they", "not", "no",
        "can", "will", "may", "but", "all", "each", "every", "some", "any",
        "there", "their", "them", "than", "then", "also", "very", "just",
        "so", "if", "as", "when", "where", "why", "who", "whom",
    }
    words = re.findall(r"[A-Za-z]\w+", text.lower())
    word_freq = {}
    for w in words:
        if w not in stopwords and len(w) > 2:
            word_freq[w] = word_freq.get(w, 0) + 1

    sorted_terms = sorted(word_freq.items(), key=lambda x: -x[1])
    return [term for term, _ in sorted_terms[:max_terms]]


def enrich_metadata(chunk: Document) -> Document:
    meta = chunk.metadata
    page = meta.get("page", 0)
    chapter = meta.get("chapter", "")

    content_type = meta.get("content_type", "prose")
    section_number = extract_section_number(chapter, chunk.page_content)
    chapter_number = extract_chapter_number(chapter, page)
    equation_density = compute_equation_density(chunk.page_content)
    has_formula = equation_density > 0.0
    key_terms = extract_key_terms(chunk.page_content)

    chunk_size_tokens = len(chunk.page_content) // 4

    meta["section_number"] = section_number
    meta["chapter_number"] = chapter_number
    meta["equation_density"] = equation_density
    meta["has_formula"] = has_formula
    meta["key_terms"] = key_terms
    meta["chunk_size_tokens"] = chunk_size_tokens
    meta["content_type"] = content_type

    return chunk


def enrich_chunks(chunks: list[Document]) -> list[Document]:
    return [enrich_metadata(chunk) for chunk in chunks]
