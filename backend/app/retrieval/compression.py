"""Contextual compression: extract only relevant sentences from retrieved chunks."""
import time
from typing import Optional
from langchain_core.documents import Document
from app.config import settings

_client: Optional = None
_last_request_time = 0.0

EXTRACT_PROMPT = """Given a question and a passage from a calculus textbook, extract ONLY the sentences that are directly relevant to answering the question. Preserve mathematical notation exactly as it appears. If no sentences are relevant, return "NO_RELEVANT_CONTENT".

Question: {query}

Passage: {passage}

Relevant sentences (preserve notation, cite [N]):"""


def compress_chunk(query: str, doc: Document, doc_index: int) -> Optional[Document]:
    global _last_request_time
    passage = doc.page_content

    if len(passage) < 300:
        return doc

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
        if extracted == "NO_RELEVANT_CONTENT" or len(extracted) < 10:
            return doc

        return Document(
            page_content=extracted,
            metadata={**doc.metadata, "compressed": True, "original_length": len(passage)},
        )
    except Exception as e:
        print(f"  Compression failed for chunk {doc_index}: {e}")
        return doc


def compress_documents(query: str, docs: list[Document]) -> list[Document]:
    compressed = []
    for i, doc in enumerate(docs):
        result = compress_chunk(query, doc, i)
        if result is not None:
            compressed.append(result)
    return compressed
