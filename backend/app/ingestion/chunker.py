import re
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from app.ingestion.pdf_structure import classify_page, get_chapter_number, get_chapter_name, get_book_page, TOC_CHAPTER_NAMES, CHAPTER_START_PAGE

SECTION_PATTERN = re.compile(r"^(1[0-6]|\d)\.(\d+)\s+([A-Z][A-Za-z0-9 ,;:\-']+?)\s+(\d+)$")
SECTION_SOLUTION_EXCLUDE = re.compile(r"^\d+\.\s+[a-z]+\s*=")
CHAPTER_PAGE_PATTERN = re.compile(r"^\d{1,4}\s+Chapter\s+(1[0-6]|\d)\s+(.+)$")
ANSWER_KEY_EXCLUDE = re.compile(r"^\d+\s+\d+\..*[a-zA-Z]\s*=")
CHAPTER_STANDALONE = re.compile(r"^Chapter\s+(1[0-6]|\d)\s*(:|\.)?\s*([A-Z][A-Za-z\s]+)$")
SECTION_APPENDIX = re.compile(r"^Section\s+(1[0-6]|\d)\.(\d+)\s*[,\s]\s*(?:pp\.)?\s*\d+", re.IGNORECASE)

TOC_ENTRY = re.compile(r"^(\d+)\.(\d+)\s+(.+?)\s+(\d+)$")

THEOREM_BOUNDARIES = re.compile(
    r"(^|\n)(?=("
    r"THEOREM\s+\d+"
    r"|Definition\s+\d+"
    r"|EXAMPLE\s+\d+"
    r"|Rule\s+\d+"
    r"))",
    re.MULTILINE | re.IGNORECASE,
)

EQUATION_DELIMITER = re.compile(r"\$\$|\$|\\\[|\\\]|\\\(|\\\)")
HAS_EQUATION = re.compile(r"\$\$|\$|\\\[|\\\(|\\begin\{equation|\\begin\{align|\\begin\{cases")


def _is_exercise_line(stripped: str) -> bool:
    return bool(re.match(r"^\d+\.\s*[a-zA-Z]", stripped)) and "=" in stripped


def _inject_markdown_headers(text: str) -> str:
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()

        if _is_exercise_line(stripped) or ANSWER_KEY_EXCLUDE.match(stripped):
            result.append(line)
            continue

        m = SECTION_PATTERN.match(stripped)
        if m:
            result.append(f"## {m.group(1)}.{m.group(2)} {m.group(3)}")
            continue
        m = SECTION_APPENDIX.match(stripped)
        if m:
            result.append(f"## {m.group(1)}.{m.group(2)}")
            continue
        if len(stripped) < 80:
            m = CHAPTER_PAGE_PATTERN.match(stripped)
            if m:
                result.append(f"# Chapter {m.group(1)} {m.group(2)}")
                continue
            m = CHAPTER_STANDALONE.match(stripped)
            if m:
                result.append(f"# Chapter {m.group(1)} {m.group(3)}")
                continue
        result.append(line)
    return "\n".join(result)


def _scan_chapters(docs: list[Document]) -> dict[int, dict]:
    sorted_docs = sorted(docs, key=lambda d: d.metadata.get("page", 0))
    page_info = {}
    current_chapter = ""

    for doc in sorted_docs:
        page_num = doc.metadata.get("page", 0)
        text = doc.page_content
        section = doc.metadata.get("section", "")

        if "Answers to Odd-Numbered Exercises" in text:
            current_chapter = "Answers to Odd-Numbered Exercises"

        for line in text.split("\n"):
            stripped = line.strip()
            if len(stripped) < 80 and not _is_exercise_line(stripped) and not ANSWER_KEY_EXCLUDE.match(stripped):
                m = CHAPTER_PAGE_PATTERN.match(stripped)
                if m and "=" not in m.group(2):
                    ch_num = int(m.group(1))
                    ch_name = m.group(2).strip()
                    current_chapter = f"Chapter {ch_num} {ch_name}"
                    continue
                m = CHAPTER_STANDALONE.match(stripped)
                if m:
                    ch_num = int(m.group(1))
                    ch_name = m.group(3).strip()
                    current_chapter = f"Chapter {ch_num} {ch_name}"
                    continue

        if not current_chapter and page_num >= CHAPTER_START_PAGE:
            fallback = get_chapter_name(page_num)
            if fallback:
                current_chapter = fallback

        page_info[page_num] = {"chapter": current_chapter}

    return page_info


def _classify_content_type(text: str) -> str:
    first_line = text.strip().split("\n")[0].strip().upper()
    if first_line.startswith("THEOREM"):
        return "theorem"
    if first_line.startswith("DEFINITION"):
        return "definition"
    if first_line.startswith("EXAMPLE"):
        return "example"
    if first_line.startswith("RULE"):
        return "rule"

    eq_count = len(HAS_EQUATION.findall(text))
    total_words = len(text.split())
    eq_density = eq_count / max(total_words, 1)

    if eq_density > 0.05:
        return "equation_heavy"

    total_chars = len(text)
    newline_count = text.count("\n")
    avg_line_len = total_chars / max(newline_count, 1)

    if avg_line_len > 80 and newline_count > 5:
        return "table"

    return "prose"


def _has_unmatched_latex(text: str) -> bool:
    cleaned = re.sub(r"\$\s*\d[\d,.\s]*", "", text)
    cleaned = re.sub(r"[a-zA-Z]\$", "", cleaned)
    cleaned = re.sub(r"(?:^|\s)\$(?:$|\s)", "", cleaned)
    singles = cleaned.count("$") % 2 != 0
    doubles = cleaned.count("$$") % 2 != 0
    has_unmatched_bracket = (cleaned.count("\\[") - cleaned.count("\\]")) != 0
    has_unmatched_paren = (cleaned.count("\\(") - cleaned.count("\\)")) != 0
    return singles or doubles or has_unmatched_bracket or has_unmatched_paren


def _fix_broken_latex(chunks: list[Document]) -> list[Document]:
    merged = []
    buffer = None
    for chunk in chunks:
        if buffer is not None:
            combined_text = buffer.page_content + "\n" + chunk.page_content
            combined_meta = {**buffer.metadata}
            if "chunk_index" in buffer.metadata:
                combined_meta["chunk_index"] = buffer.metadata["chunk_index"]
            merged.append(Document(page_content=combined_text, metadata=combined_meta))
            buffer = None
        elif _has_unmatched_latex(chunk.page_content):
            buffer = chunk
        else:
            merged.append(chunk)
    if buffer is not None:
        merged.append(buffer)
    return merged


def _enforce_theorem_boundary(chunks: list[Document]) -> list[Document]:
    result = []
    for chunk in chunks:
        splits = THEOREM_BOUNDARIES.split(chunk.page_content)
        if len(splits) == 1:
            result.append(chunk)
        else:
            head = Document(page_content=splits[0], metadata={**chunk.metadata})
            result.append(head)
            for i in range(2, len(splits), 3):
                prefix = splits[i - 1] if i - 1 < len(splits) else ""
                content = splits[i] if i < len(splits) else ""
                if prefix or content:
                    new_doc = Document(
                        page_content=(prefix + content).strip(),
                        metadata={**chunk.metadata},
                    )
                    result.append(new_doc)
    return result


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _smart_split_at_boundary(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    boundary = re.search(r"(?<=\n)(?=THEOREM|Definition|EXAMPLE|Rule|\.\s)", text)
    if boundary:
        split_pos = boundary.start()
        if split_pos > max_chars * 0.3:
            return [text[:split_pos].strip(), text[split_pos:].strip()]
    split_at_para = re.search(r"\n\n", text[max_chars // 2:])
    if split_at_para:
        split_pos = max_chars // 2 + split_at_para.start()
        return [text[:split_pos].strip(), text[split_pos:].strip()]
    return [text[:max_chars].strip(), text[max_chars:].strip()]


def _merge_small_chunks(chunks: list[Document], min_chars: int = 100) -> list[Document]:
    merged = []
    buffer = None
    for chunk in chunks:
        text = chunk.page_content
        if len(text) < min_chars:
            if buffer is not None:
                buffer.page_content += "\n" + text
                buffer.metadata["page"] = min(buffer.metadata.get("page", 0), chunk.metadata.get("page", 0))
                buffer.metadata["chunk_index"] = buffer.metadata.get("chunk_index", 0)
            else:
                buffer = Document(page_content=text, metadata={**chunk.metadata})
        else:
            if buffer is not None:
                buffer.page_content += "\n" + text
                buffer.metadata["page"] = min(buffer.metadata.get("page", 0), chunk.metadata.get("page", 0))
                merged.append(buffer)
                buffer = None
            else:
                merged.append(chunk)
    if buffer is not None:
        if merged:
            merged[-1].page_content += "\n" + buffer.page_content
        else:
            merged.append(buffer)

    result = []
    for chunk in merged:
        if len(chunk.page_content) < min_chars and result:
            result[-1].page_content += "\n" + chunk.page_content
        else:
            result.append(chunk)
    return result


def _enforce_max_chunk_size(chunks: list[Document], max_chars: int = 3000) -> list[Document]:
    def _split(text: str, meta: dict) -> list[Document]:
        if len(text) <= max_chars:
            return [Document(page_content=text.strip(), metadata={**meta})]
        splits = _smart_split_at_boundary(text, max_chars)
        result = []
        for s in splits:
            result.extend(_split(s, meta))
        merged = []
        for doc in result:
            if len(doc.page_content) < 100 and merged:
                merged[-1].page_content += "\n" + doc.page_content
            else:
                merged.append(doc)
        return merged

    result = []
    for chunk in chunks:
        if len(chunk.page_content) < 100 and result:
            result[-1].page_content += "\n" + chunk.page_content
        else:
            result.append(chunk)

    return [doc for c in result for doc in _split(c.page_content, c.metadata)]


def _get_chunk_size_for_content(content_type: str) -> int:
    sizes = {
        "theorem": 768,
        "definition": 768,
        "example": 768,
        "rule": 768,
        "equation_heavy": 512,
        "table": 1024,
        "prose": 1000,
        "answer_key": 1024,
    }
    return sizes.get(content_type, 1000)


def _get_overlap_for_content(content_type: str) -> int:
    overlaps = {
        "equation_heavy": 32,
        "theorem": 64,
        "definition": 64,
        "prose": 64,
        "table": 64,
    }
    return overlaps.get(content_type, 64)


def get_header_splitter() -> MarkdownHeaderTextSplitter:
    return MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "chapter"),
            ("##", "section"),
            ("###", "subsection"),
        ]
    )


def _chunk_toc(docs: list[Document], metadata_base: dict) -> list[Document]:
    import json
    from collections import OrderedDict

    chapters: dict[int, list[dict]] = {}
    for doc in docs:
        for line in doc.page_content.split("\n"):
            stripped = line.strip()
            m = TOC_ENTRY.match(stripped)
            if m:
                ch_num = int(m.group(1))
                sec_num = m.group(2)
                sec_name = m.group(3).strip()
                page = m.group(4)
                if ch_num not in chapters:
                    chapters[ch_num] = []
                chapters[ch_num].append({
                    "section": f"{ch_num}.{sec_num}",
                    "name": sec_name,
                    "page": int(page),
                })

    result = []
    for ch_num in sorted(chapters.keys()):
        sections = chapters[ch_num]
        ch_name = TOC_CHAPTER_NAMES.get(ch_num, f"Chapter {ch_num}")
        lines = [f"Chapter {ch_num}: {ch_name}"]
        for s in sections:
            lines.append(f"  {s['section']}  {s['name']}  (p. {s['page']})")
        content = "\n".join(lines)
        chunk = Document(
            page_content=content,
            metadata={
                **metadata_base,
                "chapter": f"Chapter {ch_num} {ch_name}",
                "chapter_number": ch_num,
                "content_type": "table_of_contents",
                "toc_sections": json.dumps(sections),
                "chunk_index": ch_num,
            },
        )
        result.append(chunk)
    return result


def _chunk_preface(content: str, metadata: dict, content_splitter) -> list[Document]:
    chunks = []
    doc = Document(page_content=content, metadata={**metadata})
    content_chunks = content_splitter.split_documents([doc])
    for i, cc in enumerate(content_chunks):
        cc.metadata = {**metadata, "chunk_index": i}
        chunks.append(cc)
    return chunks


def _chunk_answer_key(content: str, metadata: dict) -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=64,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=_estimate_tokens,
    )
    chunks = []
    doc = Document(page_content=content, metadata={**metadata})
    content_chunks = splitter.split_documents([doc])
    for i, cc in enumerate(content_chunks):
        cc.metadata = {**metadata, "chunk_index": i}
        chunks.append(cc)
    return chunks


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    chapter: Optional[str] = None,
) -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    header_splitter = get_header_splitter()
    sorted_docs = sorted(docs, key=lambda d: d.metadata.get("page", 0))
    page_chapters = _scan_chapters(sorted_docs)

    all_chunks = []

    toc_docs = [d for d in sorted_docs if d.metadata.get("section", "") == "table_of_contents"]
    if toc_docs:
        all_chunks.extend(_chunk_toc(toc_docs, {"section": "table_of_contents"}))

    for doc in sorted_docs:
        page_num = doc.metadata.get("page", 0)
        section = doc.metadata.get("section", "")
        chapter_meta = page_chapters.get(page_num, {}).get("chapter", "")

        if section in ("table_of_contents",):
            continue

        if section == "preface":
            preface_chunks = _chunk_preface(doc.page_content, {
                **doc.metadata,
                "chapter": "",
                "section": doc.metadata.get("section", "preface"),
                "content_type": "preface",
            }, RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=_estimate_tokens,
            ))
            all_chunks.extend(preface_chunks)
            continue

        if section == "answer_key":
            answer_key_chunks = _chunk_answer_key(doc.page_content, {
                **doc.metadata,
                "chapter": "Answers to Odd-Numbered Exercises",
                "section": doc.metadata.get("section", "answer_key"),
                "content_type": "answer_key",
            })
            all_chunks.extend(answer_key_chunks)
            continue

        md_text = _inject_markdown_headers(doc.page_content)

        header_chunks = header_splitter.split_text(md_text)
        if not header_chunks:
            header_chunks = [Document(page_content=md_text, metadata={})]

        for hc in header_chunks:
            content_type = _classify_content_type(hc.page_content)
            adaptive_chunk_size = _get_chunk_size_for_content(content_type)
            adaptive_overlap = _get_overlap_for_content(content_type)

            content_splitter = RecursiveCharacterTextSplitter(
                chunk_size=adaptive_chunk_size,
                chunk_overlap=adaptive_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=_estimate_tokens,
            )

            section_meta = {
                **doc.metadata,
                "chapter": chapter_meta or hc.metadata.get("chapter", ""),
                **hc.metadata,
                "content_type": content_type,
            }
            content_chunks = content_splitter.split_documents([hc])
            for cc in content_chunks:
                cc.metadata = {**section_meta, **cc.metadata}

            content_chunks = _enforce_theorem_boundary(content_chunks)
            content_chunks = _fix_broken_latex(content_chunks)

            for i, cc in enumerate(content_chunks):
                cc.metadata["chunk_index"] = i
                cc.metadata["is_equation"] = bool(HAS_EQUATION.search(cc.page_content))
                cc.metadata["equation_density"] = round(
                    len(HAS_EQUATION.findall(cc.page_content)) / max(len(cc.page_content.split()), 1), 4
                )

            all_chunks.extend(content_chunks)

    all_chunks = [c for c in all_chunks if len(c.page_content.strip()) >= 50]
    all_chunks = _merge_small_chunks(all_chunks, min_chars=100)
    all_chunks = _enforce_max_chunk_size(all_chunks, max_chars=3000)

    from app.ingestion.metadata_enricher import enrich_chunks
    all_chunks = enrich_chunks(all_chunks)

    for i, chunk in enumerate(all_chunks):
        chunk.metadata["chunk_id"] = i
        page = chunk.metadata.get("page", 0)
        chunk.metadata["book_page"] = get_book_page(page)

    return all_chunks
