"""Unit tests for chunking, cleaning, BM25, and reranking."""
import json
import tempfile
import pickle
import pytest
from pathlib import Path

from langchain_core.documents import Document
from app.ingestion.cleaner import (
    clean_text, normalize_latex, fix_unicode_artifacts, remove_footers,
    remove_garbled_lines, clean_document,
)
from app.ingestion.chunker import (
    _inject_markdown_headers, _has_unmatched_latex, _fix_broken_latex,
    _enforce_theorem_boundary, chunk_documents,
)
from app.retrieval.vector_store import BM25SparseEmbeddings
from app.retrieval.expansion import expand_query, expand_query_text


def test_clean_text():
    assert clean_text("  hello   world  ") == "hello world"
    assert clean_text("\n\n\nhello\n\nworld\n\n\n") == "hello\n\nworld"
    assert clean_text("#remove #pound") == "remove pound"


def test_normalize_latex():
    assert normalize_latex(r"\[x^2\]") == "$$x^2$$"
    assert normalize_latex(r"\(x\)") == "$x$"
    assert normalize_latex(r"\[") == "$$"
    assert normalize_latex(r"\]") == "$$"


def test_fix_unicode_artifacts():
    assert fix_unicode_artifacts("/uni2032") == "'"
    assert fix_unicode_artifacts("/uni2033") == "''"
    assert fix_unicode_artifacts("/uni2260.alt10") == "≠"
    assert fix_unicode_artifacts("/uni2260.alt1a") == "≠"
    assert fix_unicode_artifacts("/uni2212.boldH") == "−"
    assert fix_unicode_artifacts("hello /uni00A0 world") == "hello   world"
    assert "alt" not in fix_unicode_artifacts("text /.alt10 more")


def test_remove_footers():
    assert remove_footers("text\n123\nmore") == "text\n\nmore"
    assert remove_footers("Thomas' Calculus Early Transcendentals") == ""
    assert remove_footers("normal text") == "normal text"


def test_remove_garbled_lines():
    text = "normal text\nM04_HASS9020_14_SE_C04_221-299.indd 256 23/08/16 3:40 PM\nmore text"
    result = remove_garbled_lines(text)
    assert "M04_HASS9020" not in result
    assert "normal text" in result
    assert "more text" in result


def test_clean_document():
    doc = Document(
        page_content=r"test \[x^2\] /uni2032 #hash",
        metadata={"page": 1},
    )
    cleaned = clean_document(doc)
    assert "$$" in cleaned.page_content
    assert "#" not in cleaned.page_content.split("test")[0]
    assert "'" in cleaned.page_content


def test_inject_markdown_headers():
    text = "3.5 Derivatives of Trigonometric Functions 180\nsome content"
    result = _inject_markdown_headers(text)
    assert "## 3.5 Derivatives of Trigonometric Functions" in result

    text = "10 Chapter 1 Functions\nsome content"
    result = _inject_markdown_headers(text)
    assert "# Chapter 1 Functions" in result

    text = "Chapter 5 Integrals\nsome content"
    result = _inject_markdown_headers(text)
    assert "# Chapter 5 Integrals" in result


def test_has_unmatched_latex():
    assert not _has_unmatched_latex("x^2")
    assert not _has_unmatched_latex("$x$")
    assert not _has_unmatched_latex("$$x^2$$")
    assert not _has_unmatched_latex("$x$ and $y$")
    assert _has_unmatched_latex("$x")
    assert _has_unmatched_latex("$$x^2")


def test_fix_broken_latex():
    doc1 = Document(page_content="equation $x^2", metadata={"page": 1})
    doc2 = Document(page_content="+ y^2$ continues", metadata={"page": 1})
    fixed = _fix_broken_latex([doc1, doc2])
    assert len(fixed) == 1
    assert "$x^2" in fixed[0].page_content
    assert "y^2$" in fixed[0].page_content


def test_enforce_theorem_boundary():
    doc = Document(
        page_content="some text\nTHEOREM 6\n\nIf f is continuous...\nmore text",
        metadata={"page": 1},
    )
    results = _enforce_theorem_boundary([doc])
    assert len(results) >= 2


def test_chunk_documents():
    docs = [
        Document(
            page_content="Chapter 1 Functions\n\n## 1.1 Linear Functions\n\nContent about linear functions.\n\n$$f(x) = mx + b$$\n\nMore explanation.",
            metadata={"page": 1, "source": "test"},
        )
    ]
    chunks = chunk_documents(docs, chunk_size=512, chunk_overlap=50)
    assert len(chunks) > 0
    for c in chunks:
        assert len(c.page_content.strip()) >= 50
        assert "chapter" in c.metadata


def test_bm25_sparse_embeddings():
    bm25 = BM25SparseEmbeddings()
    texts = [
        "the derivative of sine is cosine",
        "the integral of cosine is sine",
        "the chain rule for composite functions",
    ]
    sparse_vectors = bm25.embed_documents(texts)
    assert len(sparse_vectors) == 3
    assert bm25.fitted
    assert len(bm25.vocab) > 0
    assert bm25.num_docs == 3

    q_sparse = bm25.embed_query("derivative of sine")
    assert len(q_sparse.indices) > 0
    assert len(q_sparse.values) > 0


def test_bm25_from_pickle():
    bm25 = BM25SparseEmbeddings()
    bm25.embed_documents(["test document with math content derivative integral"])
    state = {
        "vocab": bm25.vocab, "doc_freqs": bm25.doc_freqs,
        "num_docs": bm25.num_docs, "avg_dl": bm25.avg_dl,
        "doc_lengths": bm25.doc_lengths, "fitted": bm25.fitted,
        "k1": bm25.k1, "b": bm25.b,
    }
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(state, f)
        tmp_path = f.name

    loaded = BM25SparseEmbeddings.from_pickle(tmp_path)
    assert loaded.fitted
    assert len(loaded.vocab) > 0

    q = loaded.embed_query("derivative")
    assert len(q.indices) > 0


def test_query_expansion():
    results = expand_query("L'Hôpital's rule")
    assert len(results) >= 2

    results = expand_query("derivative of sin")
    assert len(results) >= 1

    text = expand_query_text("L'Hôpital's rule")
    assert "l'hopital" in text.lower() or "l'hospital" in text.lower()


def test_normalize_latex_consistency():
    test_cases = [
        (r"\[", "$$"),
        (r"\]", "$$"),
        (r"\(", "$"),
        (r"\)", "$"),
    ]
    for inp, expected in test_cases:
        assert normalize_latex(inp) == expected


def test_chunk_no_tiny():
    docs = [
        Document(
            page_content="Chapter 1 Functions\n\nContent here.",
            metadata={"page": 1, "source": "test"},
        )
    ]
    chunks = chunk_documents(docs)
    for c in chunks:
        assert len(c.page_content.strip()) >= 50
