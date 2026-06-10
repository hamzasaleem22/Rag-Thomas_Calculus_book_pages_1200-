"""Contextual compression: extract only relevant sentences from retrieved chunks."""
import time
import re
import hashlib
from typing import Optional
from langchain_core.documents import Document
from app.config import settings

_client: Optional = None
_last_request_time = 0.0
_compression_cache: dict[str, str] = {}
_MAX_CACHE_SIZE = 1000

EXTRACT_PROMPT = """Given a question and a passage from a calculus textbook, extract ONLY the sentences that are directly relevant to answering the question. Preserve mathematical notation exactly as it appears. If no sentences are relevant, return "NO_RELEVANT_CONTENT".

Question: {query}

Passage: {passage}

Relevant sentences (preserve notation, cite [N]):"""

BATCH_EXTRACT_PROMPT = """Given a question and several passages from a calculus textbook, extract ONLY the sentences from ALL passages that are directly relevant to answering the question. Preserve mathematical notation exactly as it appears. If no sentences are relevant, return "NO_RELEVANT_CONTENT".

Question: {query}

Passages:
{passages}

Relevant sentences (preserve notation, cite [N]):"""


def _cache_key(query: str, doc: Document) -> str:
    content_preview = doc.page_content[:200]
    raw = f"{query}:::{content_preview}"
    return hashlib.md5(raw.encode()).hexdigest()


def _query_key_terms(query: str) -> set[str]:
    stopwords = {'the', 'is', 'at', 'which', 'what', 'how', 'do', 'does', 'a', 'an',
                 'and', 'or', 'of', 'to', 'for', 'in', 'on', 'by', 'with', 'from', 'its',
                 'state', 'find', 'use', 'using', 'give', 'what', 'explain', 'show'}
    return set(re.findall(r'\b[a-zA-Z]\w+\b', query.lower())) - stopwords


def _chunk_has_key_terms(query: str, passage: str) -> bool:
    key_terms = _query_key_terms(query)
    if not key_terms:
        return False
    passage_lower = passage.lower()
    return all(term in passage_lower for term in key_terms)


def _evict_cache_if_needed():
    global _compression_cache
    if len(_compression_cache) > _MAX_CACHE_SIZE:
        items = list(_compression_cache.items())
        items = items[-(_MAX_CACHE_SIZE // 2):]
        _compression_cache = dict(items)


def compress_chunk(query: str, doc: Document, doc_index: int) -> Optional[Document]:
    global _last_request_time
    passage = doc.page_content

    if len(passage) < 300:
        return doc

    if _chunk_has_key_terms(query, passage):
        return doc

    ckey = _cache_key(query, doc)
    if ckey in _compression_cache:
        cached = _compression_cache[ckey]
        if cached == "NO_RELEVANT_CONTENT":
            return doc
        return Document(
            page_content=cached,
            metadata={**doc.metadata, "compressed": True, "original_length": len(passage)},
        )

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=settings.openai_api_key or "no-key",
            base_url=settings.openai_base_url,
        )

        elapsed = time.time() - _last_request_time
        min_delay = min(settings.request_delay * 0.1, 0.5)
        if elapsed < min_delay:
            time.sleep(min_delay - elapsed)
        _last_request_time = time.time()

        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": "You extract relevant sentences from textbook passages. Preserve all math notation."},
                {"role": "user", "content": EXTRACT_PROMPT.format(query=query, passage=passage)},
            ],
            temperature=0.0,
            max_tokens=512,
        )

        extracted = response.choices[0].message.content.strip()
        _compression_cache[ckey] = extracted
        _evict_cache_if_needed()

        if extracted == "NO_RELEVANT_CONTENT" or len(extracted) < 10:
            return doc

        return Document(
            page_content=extracted,
            metadata={**doc.metadata, "compressed": True, "original_length": len(passage)},
        )
    except Exception as e:
        print(f"  Compression failed for chunk {doc_index}: {e}")
        return doc


def _compress_batch(query: str, docs: list[Document]) -> list[Document]:
    """Compress all chunks in a single LLM call."""
    passages = []
    for i, doc in enumerate(docs):
        passages.append(f"[{i+1}] {doc.page_content}")

    combined_passages = "\n\n".join(passages)

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=settings.openai_api_key or "no-key",
            base_url=settings.openai_base_url,
        )

        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": "You extract relevant sentences from textbook passages. Preserve all math notation."},
                {"role": "user", "content": BATCH_EXTRACT_PROMPT.format(query=query, passages=combined_passages)},
            ],
            temperature=0.0,
            max_tokens=1024,
        )

        extracted = response.choices[0].message.content.strip()
        if extracted == "NO_RELEVANT_CONTENT" or len(extracted) < 50:
            return docs

        return [Document(
            page_content=extracted,
            metadata={**docs[0].metadata, "compressed": True, "original_length": sum(len(d.page_content) for d in docs)},
        )]
    except Exception as e:
        print(f"  Batch compression failed: {e}")
        return docs


def compress_documents(query: str, docs: list[Document]) -> list[Document]:
    if not docs:
        return docs

    total_chars = sum(len(d.page_content) for d in docs)
    if total_chars < 8000:
        return docs

    if settings.compression_mode == "skip":
        return docs

    if settings.compression_mode == "batch":
        return _compress_batch(query, docs)

    compressed = []
    for i, doc in enumerate(docs):
        result = compress_chunk(query, doc, i)
        if result is not None:
            compressed.append(result)
    return compressed
