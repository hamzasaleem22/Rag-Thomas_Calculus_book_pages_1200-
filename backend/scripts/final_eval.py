#!/usr/bin/env python3
"""Final optimized evaluation with rate-limited calls to kr/claude-sonnet-4.5."""
import json, sys, time, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from app.retrieval.vector_store import HybridVectorStore
from app.retrieval.reranker import Reranker
from app.retrieval.expansion import expand_query_text
from app.generation.generator import Generator
from app.config import settings

client = QdrantClient(path="/tmp/qdrant_calculus_db")
store = HybridVectorStore(client=client, bm25_path="/tmp/qdrant_calculus_bm25.pkl")
reranker = Reranker()
generator = Generator()

GROUND_TRUTHS = [
    {
        "query": "What is the derivative of sin(x)?",
        "truth": "The derivative of sin(x) with respect to x is cos(x).",
        "keywords": ["derivative", "sin", "cos"],
    },
    {
        "query": "State the Fundamental Theorem of Calculus, Part 1.",
        "truth": "If f is continuous on [a,b], then the function F(x) = ∫_a^x f(t) dt is continuous on [a,b] and differentiable on (a,b) and F'(x) = f(x).",
        "keywords": ["fundamental theorem", "continuous", "antiderivative"],
    },
    {
        "query": "What does L'Hôpital's rule state for evaluating limits of indeterminate forms?",
        "truth": "Suppose f(a)=g(a)=0, f and g differentiable, then lim f(x)/g(x) = lim f'(x)/g'(x) as x→a.",
        "keywords": ["l'hôpital", "l'hopital", "indeterminate", "limit"],
    },
    {
        "query": "What is the chain rule for differentiating composite functions?",
        "truth": "dy/dx = dy/du · du/dx = f'(g(x)) · g'(x)",
        "keywords": ["chain rule", "f'(g(x))", "composite"],
    },
    {
        "query": "What is the formula for integration by parts?",
        "truth": "∫ u dv = uv - ∫ v du",
        "keywords": ["integration by parts", "uv", "integral"],
    },
    {
        "query": "What is the ratio test for convergence of series?",
        "truth": "Let ρ = lim |a_{n+1}/a_n|. If ρ < 1 converges, ρ > 1 diverges, ρ = 1 inconclusive.",
        "keywords": ["ratio test", "convergence", "divergence"],
    },
    {
        "query": "State the divergence theorem (Gauss' theorem).",
        "truth": "The outward flux of F across closed surface S = triple integral of div F over enclosed region D.",
        "keywords": ["divergence theorem", "flux", "triple integral"],
    },
    {
        "query": "What is a Riemann sum and how is it used to define the definite integral?",
        "truth": "Riemann sum is Σ f(c_k)Δx_k. The definite integral is the limit of Riemann sums as max Δx_k → 0.",
        "keywords": ["riemann sum", "definite integral", "limit"],
    },
]

print("=" * 70)
print("FINAL EVALUATION — Optimized RAG Pipeline")
print(f"Model: {settings.llm_model} | Delay: {settings.request_delay}s")
print("=" * 70)

def grade_answer(answer: str, truth: str, keywords: list[str]) -> dict:
    a_low = answer.lower()
    t_low = truth.lower()

    kw_matches = sum(1 for kw in keywords if kw.lower() in a_low)
    kw_coverage = kw_matches / len(keywords)

    truth_key = set(re.findall(r'\b[a-zA-Z]\w+\b', t_low)) - {
        'the', 'a', 'an', 'is', 'of', 'in', 'to', 'and', 'for', 'if', 'on', 'at', 'by', 'be', 'with', 'that', 'this', 'are', 'was', 'as', 'or', 'but', 'not', 'from', 'it', 'its', 'we', 'can', 'who', 'what', 'which', 'when', 'where', 'how', 'all', 'each', 'every', 'both', 'neither', 'either', 'some', 'any', 'no', 'much', 'many', 'more', 'most', 'few', 'less', 'least'
    }
    term_matches = sum(1 for t in truth_key if t in a_low)
    precision = term_matches / max(len(truth_key), 1)

    has_citations = bool(re.search(r'\[\d+\]', answer))
    has_formulas = bool(re.search(r'[∫∑∏∂∇∞≈≠≤≥±×÷√]|\\[a-zA-Z]+|\\[\(\)\[\]]|\$\$|\$', answer))

    scored_pass = kw_coverage >= 0.5 and precision >= 0.25
    strict_pass = kw_coverage >= 0.6 and precision >= 0.35

    if strict_pass and has_citations:
        grade = "✅"
    elif scored_pass:
        grade = "⚠️"
    else:
        grade = "❌"
    
    return {
        "grade": grade,
        "keyword_coverage": kw_coverage,
        "precision": precision,
        "has_citations": has_citations,
        "has_formulas": has_formulas,
    }

correct_count = 0
partial_count = 0
wrong_count = 0

for i, item in enumerate(GROUND_TRUTHS):
    query = item["query"]
    truth = item["truth"]
    keywords = item["keywords"]

    print(f"\n  [{i+1}/{len(GROUND_TRUTHS)}] Processing: {query[:60]}...")
    sys.stdout.flush()

    # Optimized retrieval pipeline
    expanded = expand_query_text(query)
    docs = store.similarity_search(expanded, k=30)

    texts = [d.page_content for d in docs]
    reranked = reranker.rerank(expanded, texts, top_k=5)
    kept_indices = set()
    for t, _ in reranked:
        for idx, d in enumerate(docs):
            if d.page_content == t and idx not in kept_indices:
                kept_indices.add(idx)
                break
    docs = [docs[i] for i in sorted(kept_indices)]

    # Generate with rate limiting
    try:
        result = generator.generate(query, docs)
        answer = result["answer"]
        citations = result["citations"]
    except Exception as e:
        print(f"  ❌ GENERATION FAILED: {str(e)[:100]}")
        answer = f"ERROR: {e}"
        citations = []

    grade_info = grade_answer(answer, truth, keywords)
    
    if grade_info["grade"] == "✅":
        correct_count += 1
    elif grade_info["grade"] == "⚠️":
        partial_count += 1
    else:
        wrong_count += 1

    print(f"  {grade_info['grade']} Kw:{grade_info['keyword_coverage']:.0%} Pr:{grade_info['precision']:.0%} Cites:{grade_info['has_citations']} Fml:{grade_info['has_formulas']}")
    print(f"  A: {answer[:120]}...")
    print(f"  T: {truth[:100]}...")
    sys.stdout.flush()

    time.sleep(settings.request_delay)

total = len(GROUND_TRUTHS)
accuracy = (correct_count + partial_count) / total * 100
strict_accuracy = correct_count / total * 100

print(f"\n{'='*70}")
print(f"FINAL RESULTS")
print(f"{'='*70}")
print(f"  Strict correct (✅):  {correct_count}/{total} = {strict_accuracy:.1f}%")
print(f"  Partial (⚠️):         {partial_count}/{total}")
print(f"  Wrong (❌):           {wrong_count}/{total}")
print(f"  Effective accuracy:   {accuracy:.1f}%")
print(f"{'='*70}")

if strict_accuracy >= 95:
    print("✅🎯 STRICT ACCURACY >= 95% — TARGET ACHIEVED!")
elif accuracy >= 95:
    print("✅ EFFECTIVE ACCURACY >= 95% — TARGET ACHIEVED (with partials)")
else:
    print(f"❌ Below 95% ({accuracy:.1f}%). Further optimization needed.")
print(f"{'='*70}")
