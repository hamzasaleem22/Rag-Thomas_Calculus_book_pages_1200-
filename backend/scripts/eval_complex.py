#!/usr/bin/env python3
"""Quick accuracy eval: 4 complex queries, with before/after comparison."""
import json, sys, re, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["OPENAI_BASE_URL"] = os.environ.get("OPENAI_BASE_URL", "http://localhost:20128/v1")

import numpy as np
from qdrant_client import QdrantClient
from app.config import settings
from app.retrieval.vector_store import HybridVectorStore
from app.retrieval.reranker import Reranker
from app.retrieval.expansion import expand_query_text
from app.generation.generator import Generator

client = QdrantClient(path="/tmp/qdrant_calculus_db")
store = HybridVectorStore(client=client, bm25_path="/tmp/qdrant_calculus_bm25.pkl")
reranker = Reranker()

QUERIES = [
    {
        "id": "ch02",
        "query": "Using the limit definition of the derivative, prove that d/dx[x^3] = 3x^2. Show each step clearly.",
        "truth": "f'(x) = lim_{h→0} (f(x+h)-f(x))/h = lim_{h→0} ((x+h)^3 - x^3)/h = lim_{h→0} (3x^2h + 3xh^2 + h^3)/h = lim_{h→0} (3x^2 + 3xh + h^2) = 3x^2.",
    },
    {
        "id": "ch06",
        "query": "Derive the formula for the volume of a sphere of radius R using the disk method of solids of revolution. What is the formula and how do you obtain it by revolving the semicircle y = sqrt(R^2 - x^2) about the x-axis?",
        "truth": "V = π∫_{-R}^{R} (sqrt(R^2 - x^2))^2 dx = π∫_{-R}^{R} (R^2 - x^2) dx = 2π∫_0^R (R^2 - x^2) dx = 2π[R^2x - x^3/3]_0^R = 2π(R^3 - R^3/3) = 4/3 πR^3.",
    },
    {
        "id": "ch12",
        "query": "Find an equation for the plane through the point P(1,2,3) that is perpendicular to the vector n = (2,-1,4). Then determine the distance from the origin to this plane.",
        "truth": "Plane equation: 2(x-1) - 1(y-2) + 4(z-3) = 0 → 2x - y + 4z = 12. Distance from origin: |2(0) - 0 + 4(0) - 12|/√(4+1+16) = 12/√21 ≈ 2.618.",
    },
    {
        "id": "ch16",
        "query": "State Green's Theorem in both circulation-curl and flux-divergence forms. Explain how the flux-divergence form relates to the divergence theorem (Gauss's theorem) in the plane.",
        "truth": "Circulation-curl: ∮_C F·dr = ∬_R curl F dA = ∬_R (∂N/∂x - ∂M/∂y) dxdy. Flux-divergence: ∮_C F·n ds = ∬_R div F dA = ∬_R (∂M/∂x + ∂N/∂y) dxdy. The flux-divergence form is the 2D version of the divergence theorem: the net outward flux through a closed curve equals the double integral of div F over the enclosed region.",
    },
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

def compute_nli_metrics(answer: str, docs):
    from app.retrieval.citation_verifier import verify_answer_claims
    from app.retrieval.confidence import ConfidenceScorer
    claim_results = verify_answer_claims(answer, docs)
    if not claim_results:
        return {"claim_count": 0, "claim_accuracy": 0.0, "hallucination_rate": 0.0, "average_confidence": 0.0}
    scorer = ConfidenceScorer()
    total = len(claim_results)
    verified = sum(1 for r in claim_results if r.verified)
    contradicted = sum(1 for r in claim_results if r.nli_label == "contradiction")
    confidences = [scorer.score_claim(r.claim_text, r.nli_score, r.nli_label, is_mathematical=r.is_mathematical).composite_confidence for r in claim_results]
    return {
        "claim_count": total,
        "claim_accuracy": verified / max(total, 1),
        "hallucination_rate": contradicted / max(total, 1),
        "average_confidence": float(np.mean(confidences)) if confidences else 0.0,
    }

def latex_fidelity(answer: str) -> float:
    if "$" not in answer and "$$" not in answer:
        return 1.0
    singles = answer.count("$") - 2 * answer.count("$$")
    issues = 1 if singles % 2 != 0 else 0
    t = 1
    frac_bare = len(re.findall(r'\\frac(?!\{)', answer))
    if frac_bare > 0:
        issues += frac_bare
    t += 1
    return max(0.0, 1.0 - issues / max(t, 1))

def evaluate_config(config_overrides: dict, label: str) -> dict:
    """Run eval with specific config overrides."""
    for k, v in config_overrides.items():
        setattr(settings, k, v)

    gen = Generator()
    print(f"\n{'='*60}")
    print(f"  CONFIG: {label}")
    print(f"  NLI={settings.use_nli_verifier}, SC={settings.use_self_consistency}")
    print(f"{'='*60}")

    results = []
    for item in QUERIES:
        q = item["query"]
        print(f"\n  [{item['id']}] {q[:60]}...", end=" ")
        sys.stdout.flush()

        docs = retrieve_docs(q)
        if not docs:
            print("NO DOCS"); continue

        try:
            gen_result = gen.generate(q, docs)
        except Exception as e:
            print(f"ERR: {e}"); continue

        answer = gen_result.get("answer", "")
        metrics = compute_nli_metrics(answer, docs)
        metrics["latex"] = latex_fidelity(answer)
        metrics["id"] = item["id"]
        metrics["answer_len"] = len(answer)
        results.append(metrics)
        print(f"claims={metrics['claim_count']} acc={metrics['claim_accuracy']:.2f} hal={metrics['hallucination_rate']:.2f} conf={metrics['average_confidence']:.2f} latex={metrics['latex']:.2f}")

    if not results:
        return {"label": label, "error": "no results"}

    summary = {
        "label": label,
        "claim_accuracy": round(np.mean([r["claim_accuracy"] for r in results]), 4),
        "hallucination_rate": round(np.mean([r["hallucination_rate"] for r in results]), 4),
        "average_confidence": round(np.mean([r["average_confidence"] for r in results]), 4),
        "latex_fidelity": round(np.mean([r["latex"] for r in results]), 4),
        "total_claims": sum(r["claim_count"] for r in results),
        "per_query": results,
    }
    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("  COMPLEX QUERY ACCURACY EVAL")
    print(f"  Model: {settings.llm_model}")
    print("=" * 60)

    # "Before" — disable NLI and self-consistency
    before = evaluate_config(
        {"use_nli_verifier": False, "use_self_consistency": False},
        "BEFORE (no NLI, no SC)"
    )

    # "After" — full pipeline with NLI and self-consistency
    after = evaluate_config(
        {"use_nli_verifier": True, "use_self_consistency": True},
        "AFTER (NLI + SC enabled)"
    )

    # Comparison table
    print(f"\n{'='*60}")
    print(f"  COMPARISON: BEFORE vs AFTER")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Before':>10} {'After':>10} {'Δ':>10}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10}")
    for metric in ["claim_accuracy", "hallucination_rate", "average_confidence", "latex_fidelity"]:
        b_val = before.get(metric, 0)
        a_val = after.get(metric, 0)
        delta = a_val - b_val
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "→")
        print(f"  {metric:<25} {b_val:>10.4f} {a_val:>10.4f} {arrow}{abs(delta):>8.4f}")

    total_b = before.get("total_claims", 0)
    total_a = after.get("total_claims", 0)
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'total_claims':<25} {total_b:>10} {total_a:>10}")

    # Save report
    report = {"before": before, "after": after}
    report_path = Path("eval_results/complex_accuracy_report.json")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report: {report_path}")
    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")
