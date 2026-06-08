import re
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

SECTION_PATTERN = re.compile(r"^(1[0-6]|\d)\.(\d+)\s+([A-Z][A-Za-z0-9 ,;:\-']+?)\s+(\d+)$")
CHAPTER_PAGE_PATTERN = re.compile(r"^\d{1,4}\s+Chapter\s+(1[0-6]|\d)\s+(.+)$")
CHAPTER_STANDALONE = re.compile(r"^Chapter\s+(1[0-6]|\d)\s*(:|\.)?\s*(.+)$")
SECTION_APPENDIX = re.compile(r"^Section\s+(1[0-6]|\d)\.(\d+)\s*[,\s]\s*(?:pp\.)?\s*\d+", re.IGNORECASE)

THEOREM_BOUNDARIES = re.compile(
    r"(^|\n)(?=("
    r"THEOREM\s+\d+"
    r"|Definition\s+\d+"
    r"|EXAMPLE\s+\d+"
    r"|Rule\s+\d+"
    r"))",
    re.MULTILINE | re.IGNORECASE,
)


def _inject_markdown_headers(text: str) -> str:
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
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

        for line in text.split("\n"):
            stripped = line.strip()
            if len(stripped) < 80:
                m = CHAPTER_PAGE_PATTERN.match(stripped)
                if m:
                    current_chapter = f"Chapter {m.group(1)} {m.group(2)}"
                    continue
                m = CHAPTER_STANDALONE.match(stripped)
                if m:
                    current_chapter = f"Chapter {m.group(1)} {m.group(3).strip()}"
                    continue

        page_info[page_num] = {"chapter": current_chapter}

    return page_info


def _has_unmatched_latex(text: str) -> bool:
    singles = text.count("$") % 2 != 0
    doubles = text.count("$$") % 2 != 0
    return singles or doubles


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
    return [text[:max_chars].strip(), text[max_chars:].strip()]


def get_header_splitter() -> MarkdownHeaderTextSplitter:
    return MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "chapter"),
            ("##", "section"),
            ("###", "subsection"),
        ]
    )


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    chapter: Optional[str] = None,
) -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    header_splitter = get_header_splitter()
    content_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=_estimate_tokens,
    )
    sorted_docs = sorted(docs, key=lambda d: d.metadata.get("page", 0))
    page_chapters = _scan_chapters(sorted_docs)

    all_chunks = []
    for doc in sorted_docs:
        page_num = doc.metadata.get("page", 0)
        chapter = page_chapters.get(page_num, {}).get("chapter", "")
        md_text = _inject_markdown_headers(doc.page_content)

        header_chunks = header_splitter.split_text(md_text)
        if not header_chunks:
            header_chunks = [Document(page_content=md_text, metadata={})]

        for hc in header_chunks:
            section_meta = {
                **doc.metadata,
                "chapter": chapter or hc.metadata.get("chapter", ""),
                **hc.metadata,
            }
            content_chunks = content_splitter.split_documents([hc])
            for cc in content_chunks:
                cc.metadata = {**section_meta, **cc.metadata}

            content_chunks = _enforce_theorem_boundary(content_chunks)
            content_chunks = _fix_broken_latex(content_chunks)

            for i, cc in enumerate(content_chunks):
                cc.metadata["chunk_index"] = i
                cc.metadata["is_equation"] = (
                    "$$" in cc.page_content
                    or "\\[" in cc.page_content
                    or "\\(" in cc.page_content
                )

            all_chunks.extend(content_chunks)

    all_chunks = [c for c in all_chunks if len(c.page_content.strip()) >= 50]

    for i, chunk in enumerate(all_chunks):
        chunk.metadata["chunk_id"] = i

    return all_chunks
