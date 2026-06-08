#!/usr/bin/env python3
"""Diagnostic evaluation of the RAG pipeline: retrieval accuracy + answer quality."""
import json, sys, math, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from app.retrieval.vector_store import HybridVectorStore
from app.retrieval.reranker import Reranker
from app.generation.generator import Generator

QDRANT_DB_PATH = "/tmp/qdrant_calculus_db"
BM25_PATH = "/tmp/qdrant_calculus_bm25.pkl"

client = QdrantClient(path=QDRANT_DB_PATH)
store = HybridVectorStore(client=client, bm25_path=BM25_PATH)
reranker = Reranker()
generator = Generator()

# ── Debug: Check BM25 state ──
bm25 = store.sparse_embedding
print(f"── BM25 Diagnostic ──")
print(f"  Fitted: {bm25.fitted}")
print(f"  Vocab size: {len(bm25.vocab)}")
print(f"  Num docs: {bm25.num_docs}")
print(f"  Avg DL: {bm25.avg_dl:.2f}")

# Test query BM25
query = "derivative of sin"
q_sparse = bm25.embed_query(query)
print(f"  Query '{query}' sparse indices: {len(q_sparse.indices)}, values: {len(q_sparse.values)}")
if len(q_sparse.indices) == 0:
    print("  ⚠️  BM25 QUERY PRODUCES EMPTY SPARSE VECTOR! Hybrid search degraded.")
else:
    print("  ✅ BM25 query produces sparse vectors - hybrid search active")

# ── Test queries across chapters ──
TEST_QUERIES = [
    ("What is the derivative of sin(x)?", "derivative of sin", ["3", "Derivative"]),
    ("State the Fundamental Theorem of Calculus, Part 1.", "FTC Part 1", ["5", "Integral"]),
    ("What is L'Hôpital's rule and when do you use it?", "L'Hopital", ["4", "L'Hôpital"]),
    ("How do you compute the area between two curves?", "Area between curves", ["5", "6", "Area"]),
    ("What is the chain rule for differentiation?", "Chain rule", ["3", "Chain"]),
    ("Explain integration by parts with formula.", "Integration by parts", ["8", "Integration by parts"]),
    ("What is a Riemann sum and how is it used to define the definite integral?", "Riemann sum", ["5", "Riemann"]),
    ("How do you find the volume of a solid of revolution using disks?", "Volume disks", ["6", "Volume"]),
    ("What is a Taylor series and how do you find it?", "Taylor series", ["10", "Taylor"]),
    ("Explain the concept of partial derivatives.", "Partial derivatives", ["14", "Partial"]),
    ("How do you evaluate a double integral over a rectangular region?", "Double integral", ["15", "Double"]),
    ("What is the divergence theorem?", "Divergence theorem", ["16", "Divergence"]),
    ("Describe Newton's method for finding roots.", "Newton's method", ["4", "Newton"]),
    ("What is the limit definition of the derivative?", "Limit definition derivative", ["2", "3", "Limit"]),
    ("How do you solve separable differential equations?", "Separable DE", ["7", "9", "Separable"]),
    ("What is the formula for arc length of a curve?", "Arc length", ["6", "11", "Arc"]),
    ("Explain the concept of a vector field.", "Vector field", ["16", "Vector field"]),
    ("How do you use the second derivative test for local extrema?", "Second derivative test", ["14", "Second derivative test"]),
    ("What is the ratio test for series convergence?", "Ratio test", ["10", "Ratio test"]),
    ("Describe how to find the area of a surface of revolution.", "Surface area revolution", ["6", "Surface"]),
]

print(f"\n── Running {len(TEST_QUERIES)} diagnostic queries ──\n")

results_summary = []
all_retrieved_has_topic = []
all_reranked_has_topic = []

def content_matches(docs_list, kw_list):
    match_count = 0
    for d in docs_list:
        content_lower = d.page_content.lower()
        for kw in kw_list:
            if kw.lower() in content_lower:
                match_count += 1
                break
    return match_count, len(docs_list)

all_docs_for_dedup_check = []

for idx, (query, topic, keywords) in enumerate(TEST_QUERIES):
    docs = store.similarity_search(query, k=30)
    all_docs_for_dedup_check.extend(docs)

    retrieved_match, retrieved_total = content_matches(docs, keywords)
    all_retrieved_has_topic.append(retrieved_match > 0)

    texts = [d.page_content for d in docs]
    reranked = reranker.rerank(query, texts, top_k=5)
    kept = {t for t, _ in reranked}
    reranked_docs = [d for d in docs if d.page_content in kept]

    reranked_match, reranked_total = content_matches(reranked_docs, keywords)
    all_reranked_has_topic.append(reranked_match > 0)

    try:
        result = generator.generate(query, reranked_docs)
        answer = result["answer"]
        answer_has_citations = bool(result["citations"])
        answer_length = len(answer)
    except Exception as e:
        answer = f"ERROR: {e}"
        answer_has_citations = False
        answer_length = 0

    status = "✅" if retrieved_match > 0 and reranked_match > 0 else "⚠️"
    results_summary.append({
        "query": query[:60],
        "topic": topic,
        "retrieved_match": f"{retrieved_match}/{retrieved_total}",
        "reranked_match": f"{reranked_match}/{reranked_total}",
        "has_citations": answer_has_citations,
        "answer_length": answer_length,
        "status": status,
    })
    print(f"  [{idx+1:2d}] {status} {query[:55]:55s} | R:{retrieved_match}/{retrieved_total} RR:{reranked_match}/{reranked_total} | {'📎' if answer_has_citations else '❌'}cites")

total = len(TEST_QUERIES)
retrieval_accuracy = sum(all_retrieved_has_topic) / total * 100
rerank_accuracy = sum(all_reranked_has_topic) / total * 100

print(f"\n{'='*60}")
print(f"DIAGNOSTIC RESULTS")
print(f"{'='*60}")
print(f"  Total queries:              {total}")
print(f"  Retrieval accuracy (top-30): {retrieval_accuracy:.1f}%")
print(f"  Reranked accuracy (top-5):   {rerank_accuracy:.1f}%")
print(f"  Queries with citations:      {sum(1 for r in results_summary if r['has_citations'])}/{total}")
print(f"  Avg answer length:           {sum(r['answer_length'] for r in results_summary)/total:.0f} chars")

print(f"\n── Detailed breakdown ──")
for r in results_summary:
    print(f"  {r['status']} [{r['topic'][:25]:25s}] R:{r['retrieved_match']:>8s} RR:{r['reranked_match']:>8s} | {r['query'][:50]}")

# ── Deep checks ──
print(f"\n── Deep-Dive Critical Checks ──")

# Chunk quality
print(f"\n  Chunk quality:")
chunks = []
with open("data/markdown/parsed_docs.jsonl") as f:
    for line in f:
        data = json.loads(line)
        chunks.append(data)

lengths = [len(c.get("page_content","")) for c in chunks]
print(f"    Count: {len(chunks)}")
print(f"    Avg length: {sum(lengths)/len(lengths):.0f} chars")
print(f"    Min length: {min(lengths)}")
print(f"    Max length: {max(lengths)}")
tiny = sum(1 for l in lengths if l < 50)
print(f"    Chunks < 50 chars: {tiny}")
print(f"    Chunks < 100 chars: {sum(1 for l in lengths if l < 100)}")
no_chapter = sum(1 for c in chunks if not c.get("metadata",{}).get("chapter",""))
print(f"    Chunks without chapter: {no_chapter}")

# BM25 check
if len(q_sparse.indices) == 0:
    print(f"\n  ⚠️  BM25 not fitted → sparse query empty → hybrid degraded to dense-only")
else:
    print(f"\n  ✅ BM25 fitted and producing sparse vectors")

# Rerank dedup check
if all_docs_for_dedup_check:
    dup_count = len(set(d.page_content for d in all_docs_for_dedup_check))
    print(f"  Rerank dedup: unique content keys = {dup_count} vs {len(all_docs_for_dedup_check)} total (collisions: {len(all_docs_for_dedup_check) - dup_count})")
    if len(all_docs_for_dedup_check) != dup_count:
        print(f"  ⚠️  Content-key dedup may lose unique chunks!")

# Generation check
print(f"\n  Generation:")
sample_docs = store.similarity_search("derivative", k=2)
if sample_docs:
    msgs = generator._build_messages("test", sample_docs)
    print(f"    System prompt mentions citations: {'cite' in msgs[0]['content'].lower()}")
    print(f"    Messages: {len(msgs)}")
    print(f"    User prompt length: {len(msgs[-1]['content'])} chars")

# Final verdict
print(f"\n{'='*60}")
overall = (retrieval_accuracy + rerank_accuracy) / 2
if overall >= 95:
    print(f"✅ OVERALL: {overall:.1f}% — PASSES 95% threshold")
else:
    print(f"❌ OVERALL: {overall:.1f}% — BELOW 95% threshold. Optimization needed.")
print(f"{'='*60}")
