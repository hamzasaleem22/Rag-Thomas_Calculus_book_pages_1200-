#!/usr/bin/env python3
"""
Phase 1 Migration: Re-chunk and re-index with improved chunker + cleaner.
Re-reads the existing markdown file, applies improved cleaning + chunking, re-indexes.
"""
import json, sys, time, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from langchain_core.documents import Document
from app.ingestion.pdf_loader import parse_marker_markdown
from app.ingestion.cleaner import clean_documents
from app.ingestion.chunker import chunk_documents
from app.ingestion.indexer import index_documents_and_return_bm25

MD_FILE = "data/markdown/Thomas' Calculus Early Transcendentals, 14th Edition.md"
OUTPUT_JSONL = "data/markdown/parsed_docs.jsonl"
QDRANT_DB_PATH = "/tmp/qdrant_calculus_db"
BM25_PATH = "/tmp/qdrant_calculus_bm25.pkl"

print("=" * 60)
print("Phase 1: Data Quality & Chunking Migration")
print("=" * 60)

print(f"\n1. Loading pages from {MD_FILE}...")
docs = parse_marker_markdown(MD_FILE)
print(f"   Loaded {len(docs)} page documents")

print("\n2. Cleaning with improved cleaner (Unicode fix + math notation)...")
docs = clean_documents(docs)
print(f"   Cleaned {len(docs)} documents")

print("\n3. Chunking with improved chunker (token-based, theorem boundaries)...")
chunks = chunk_documents(docs, chunk_size=1024, chunk_overlap=128)
print(f"   Created {len(chunks)} chunks")

lengths = [len(c.page_content) for c in chunks]
tiny = sum(1 for l in lengths if l < 50)
no_chapter = sum(1 for c in chunks if not c.metadata.get("chapter", ""))
print(f"   Avg length: {sum(lengths)/len(lengths):.0f} chars | Max: {max(lengths)} | <50: {tiny} | No chapter: {no_chapter}")

for i, c in enumerate(chunks):
    if "THEOREM 6" in c.page_content and "L'Hôpital" in c.page_content:
        print(f"\n   ✅ L'Hôpital Theorem in chunk {i}: {len(c.page_content)} chars")
        print(f"   Has limit formula: {'lim' in c.page_content.lower()}")
        print(f"   Preview: {c.page_content[:200]}...")
        break

print("\n4. Saving chunks to JSONL...")
with open(OUTPUT_JSONL, "w") as f:
    for chunk in chunks:
        f.write(chunk.model_dump_json() + "\n")
print(f"   Saved {len(chunks)} chunks to {OUTPUT_JSONL}")

print("\n5. Re-building Qdrant index (persistent)...")
client = QdrantClient(path=QDRANT_DB_PATH)
existing = {c.name for c in client.get_collections().collections}
if "calculus_book" in existing:
    print("   Dropping existing collection...")
    client.delete_collection("calculus_book")

t0 = time.time()
count, bm25 = index_documents_and_return_bm25(chunks, client=client, batch_size=64)
elapsed = time.time() - t0
print(f"   Indexed {count} vectors in {elapsed:.1f}s ({count/elapsed:.0f} chunks/s)")

print("\n6. Saving BM25 state...")
bm25_state = {
    "vocab": bm25.vocab, "doc_freqs": bm25.doc_freqs, "num_docs": bm25.num_docs,
    "avg_dl": bm25.avg_dl, "doc_lengths": bm25.doc_lengths, "fitted": bm25.fitted,
    "k1": bm25.k1, "b": bm25.b,
}
with open(BM25_PATH, "wb") as f:
    pickle.dump(bm25_state, f)
print(f"   Saved BM25 ({len(bm25.vocab)} vocab, {bm25.num_docs} docs)")

print(f"\n{'='*60}")
print(f"✅ Phase 1 migration complete!")
print(f"   Chunks: {len(chunks)} (was 8082)")
print(f"   Qdrant: {QDRANT_DB_PATH}")
print(f"   BM25:   {BM25_PATH}")
print(f"{'='*60}")
