#!/usr/bin/env python3
"""
Eval Accuracy — focused on generation output quality using NLI-based claim metrics.
Runs a subset of queries and reports:
  - Claim-level accuracy (NLI-verified claims / total claims)
  - Hallucination rate (contradicted claims / total claims)
  - Average confidence (compound UQ score)
  - Abstention rate
  - Citation accuracy
  - LaTeX / equation fidelity
"""
import json, sys, math, re, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["OPENAI_BASE_URL"] = os.environ.get("OPENAI_BASE_URL", "http://localhost:20128/v1")

import numpy as np
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from app.config import settings
from app.retrieval.vector_store import HybridVectorStore
from app.retrieval.reranker import Reranker
from app.retrieval.expansion import expand_query_text
from app.generation.generator import Generator

# ── Shared Resources ──
client = QdrantClient(path="/tmp/qdrant_calculus_db")
store = HybridVectorStore(client=client, bm25_path="/tmp/qdrant_calculus_bm25.pkl")
reranker = Reranker()
generator = Generator()

settings.request_delay = 2.0
settings.llm_model = "kr/claude-haiku-4.5"

# ── Test Queries ──
STANDARD_QUERIES = [
    {"id": "ch02", "query": "What is the limit definition of the derivative?",
     "truth": "f'(x) = lim_{h→0} (f(x+h) - f(x))/h, provided the limit exists."},
    {"id": "ch03", "query": "What is the derivative of sin(x) and how do you find it using the chain rule when the argument is x²?",
     "truth": "d/dx sin(x) = cos(x). d/dx sin(x²) = cos(x²) · 2x by chain rule."},
    {"id": "ch04", "query": "State L'Hôpital's rule and find lim_{x→0} sin(x)/x.",
     "truth": "L'Hôpital: if f(a)=g(a)=0, lim f(x)/g(x) = lim f'(x)/g'(x). lim_{x→0} sin(x)/x = lim_{x→0} cos(x)/1 = 1."},
    {"id": "ch05", "query": "State the Fundamental Theorem of Calculus, Part 1.",
     "truth": "If f is continuous on [a,b] and F is an antiderivative, then ∫_a^b f(x)dx = F(b) - F(a)."},
    {"id": "ch10", "query": "State the ratio test for convergence of infinite series.",
     "truth": "Let ρ = lim |a_{n+1}/a_n|. If ρ < 1: converges absolutely. ρ > 1: diverges. ρ = 1: inconclusive."},
    {"id": "ch14", "query": "What is the second derivative test for local extrema of functions of two variables?",
     "truth": "Let D = f_xx f_yy - (f_xy)². D > 0 and f_xx > 0: local min. D > 0 and f_xx < 0: local max. D < 0: saddle point."},
    {"id": "ch16", "query": "State the divergence theorem (Gauss' theorem).",
     "truth": "∬_S F·n dσ = ∭_D ∇·F dV. The outward flux of F through closed surface S equals the triple integral of div F over the enclosed volume D."},
]

def retrieve_docs(query: str, top_k: int = 5):
    expanded = expand_query_text(query) if query.strip() else query
    docs = store.similarity_search(expanded, k=30)
    if not docs:
        return []
    texts = [d.page_content for d in docs]
    reranked = reranker.rerank(expanded, texts, top_k=top_k)
    kept_set = {t for t, _ in reranked}
    return [d for d in docs if d.page_content in kept_set]


def latex_fidelity(answer: str) -> float:
    if "$" not in answer and "$$" not in answer:
        return 1.0
    issues = 0
    total = 0
    singles = answer.count("$") - 2 * answer.count("$$")
    if singles % 2 != 0:
        issues += 1
    total += 1
    frac_bare = len(re.findall(r'\\frac(?!\{)', answer))
    if frac_bare > 0:
        issues += frac_bare
    total += 1
    sum_bare = len(re.findall(r'\\sum\s+[a-z]', answer))
    if sum_bare > 0:
        issues += sum_bare
    total += 1
    return max(0.0, 1.0 - issues / max(total, 1))


def compute_nli_metrics(answer: str, docs: list[Document]) -> dict:
    """Compute NLI-based claim-level metrics."""
    from app.retrieval.citation_verifier import verify_answer_claims
    from app.retrieval.confidence import ConfidenceScorer

    claim_results = verify_answer_claims(answer, docs)
    if not claim_results:
        return {"claim_count": 0, "claim_accuracy": 0.0, "hallucination_rate": 0.0,
                "average_confidence": 0.0, "abstention_count": 0, "per_claim": []}

    scorer = ConfidenceScorer()
    total = len(claim_results)
    verified = sum(1 for r in claim_results if r.verified)
    contradicted = sum(1 for r in claim_results if r.nli_label == "contradiction")
    
    confidences = []
    abstentions = 0
    for r in claim_results:
        c = scorer.score_claim(r.claim_text, r.nli_score, r.nli_label,
                                is_mathematical=r.is_mathematical)
        confidences.append(c.composite_confidence)
        if c.abstain:
            abstentions += 1

    per_claim = [
        {
            "claim": r.claim_text[:80],
            "verified": r.verified,
            "score": r.composite_score,
            "nli_label": r.nli_label,
            "confidence": confidences[i],
        }
        for i, r in enumerate(claim_results)
    ]

    return {
        "claim_count": total,
        "claim_accuracy": verified / max(total, 1),
        "hallucination_rate": contradicted / max(total, 1),
        "average_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "abstention_count": abstentions,
        "abstention_rate": abstentions / max(total, 1),
        "per_claim": per_claim,
    }


def evaluate_queries(queries: list[dict], label: str = "standard"):
    results = []
    for item in queries:
        q = item["query"]
        print(f"  [{item['id']}] {q[:60]}...", end=" ")
        sys.stdout.flush()

        t0 = time.perf_counter()
        docs = retrieve_docs(q)
        if not docs:
            print("NO DOCS")
            continue

        try:
            gen_result = generator.generate(q, docs)
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        answer = gen_result.get("answer", "")
        t1 = time.perf_counter()

        metrics = compute_nli_metrics(answer, docs)
        latex = latex_fidelity(answer)
        metrics["latex_fidelity"] = latex
        metrics["latency_s"] = round(t1 - t0, 2)
        metrics["answer_len"] = len(answer)
        metrics["id"] = item["id"]
        metrics["query"] = q[:80]

        results.append(metrics)
        print(f"claims={metrics['claim_count']} acc={metrics['claim_accuracy']:.2f} hal={metrics['hallucination_rate']:.2f} conf={metrics['average_confidence']:.2f} latex={latex:.2f} ({metrics['latency_s']}s)")

    return results


def print_summary(results, label: str):
    if not results:
        print(f"\n  No results for {label}")
        return
    
    avg_acc = np.mean([r["claim_accuracy"] for r in results])
    avg_hal = np.mean([r["hallucination_rate"] for r in results])
    avg_conf = np.mean([r["average_confidence"] for r in results])
    avg_abstain = np.mean([r["abstention_rate"] for r in results])
    avg_latex = np.mean([r["latex_fidelity"] for r in results])
    total_claims = sum(r["claim_count"] for r in results)
    total_verified = sum(r["claim_count"] * r["claim_accuracy"] for r in results)

    print(f"\n  ── {label.upper()} RESULTS ({len(results)} queries) ──")
    print(f"  Total claims:           {total_claims}")
    print(f"  Verified claims:        {int(total_verified)}/{total_claims}")
    print(f"  Claim Accuracy:         {avg_acc:.4f}")
    print(f"  Hallucination Rate:     {avg_hal:.4f} (lower=better)")
    print(f"  Average Confidence:     {avg_conf:.4f}")
    print(f"  Abstention Rate:        {avg_abstain:.4f}")
    print(f"  LaTeX Fidelity:         {avg_latex:.4f}")
    print(f"  Total Latency:          {sum(r['latency_s'] for r in results):.1f}s")

    return {
        "label": label,
        "queries": len(results),
        "total_claims": int(total_claims),
        "verified_claims": int(total_verified),
        "claim_accuracy": round(avg_acc, 4),
        "hallucination_rate": round(avg_hal, 4),
        "average_confidence": round(avg_conf, 4),
        "abstention_rate": round(avg_abstain, 4),
        "latex_fidelity": round(avg_latex, 4),
        "total_latency_s": round(sum(r['latency_s'] for r in results), 1),
        "per_query": results,
    }


if __name__ == "__main__":
    print("=" * 65)
    print("  RAG BOOK — ACCURACY EVALUATION (NLI-based)")
    print(f"  Model: {settings.llm_model}")
    print("=" * 65)
    print()
    print(f"Settings: use_nli={settings.use_nli_verifier}, use_sc={settings.use_self_consistency}")
    print()

    t_start = time.perf_counter()
    results = evaluate_queries(STANDARD_QUERIES, "standard")
    summary = print_summary(results, "standard")
    elapsed = time.perf_counter() - t_start

    print(f"\n  Total evaluation time: {elapsed:.0f}s")

    # Save report
    report_path = Path("eval_results/accuracy_report.json")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Report saved to {report_path}")

    print("\n" + "=" * 65)
    print("  EVALUATION COMPLETE")
    print("=" * 65)
