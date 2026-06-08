#!/usr/bin/env python3
"""Deep evaluation: compare generated answers against PDF ground truth."""
import json, sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from app.retrieval.vector_store import HybridVectorStore
from app.retrieval.reranker import Reranker
from app.generation.generator import Generator

client = QdrantClient(path="/tmp/qdrant_calculus_db")
store = HybridVectorStore(client=client, bm25_path="/tmp/qdrant_calculus_bm25.pkl")
reranker = Reranker()
generator = Generator()

# Ground truth from Thomas' Calculus 14th Edition
GROUND_TRUTHS = [
    {
        "query": "What is the derivative of sin(x)?",
        "truth": "The derivative of sin(x) with respect to x is cos(x).",
        "chapter": "3",
    },
    {
        "query": "State the Fundamental Theorem of Calculus, Part 1.",
        "truth": "If f is continuous on [a, b] and F is any antiderivative of f, then ∫_a^b f(x) dx = F(b) - F(a).",
        "chapter": "5",
    },
    {
        "query": "What does L'Hôpital's rule state for evaluating limits of indeterminate forms?",
        "truth": "Suppose f(a) = g(a) = 0, that f and g are differentiable on an open interval I containing a, and that g'(x) ≠ 0 on I if x ≠ a. Then lim(x→a) f(x)/g(x) = lim(x→a) f'(x)/g'(x), provided the limit on the right exists.",
        "chapter": "4",
    },
    {
        "query": "What is the chain rule for differentiating composite functions?",
        "truth": "If f(u) is differentiable at u = g(x) and g(x) is differentiable at x, then the composite function (f ∘ g)(x) = f(g(x)) is differentiable at x and (f ∘ g)'(x) = f'(g(x)) · g'(x). In Leibniz notation: dy/dx = dy/du · du/dx.",
        "chapter": "3",
    },
    {
        "query": "What is the formula for integration by parts?",
        "truth": "∫ u dv = uv - ∫ v du",
        "chapter": "8",
    },
    {
        "query": "What is a Riemann sum and how is it used to define the definite integral?",
        "truth": "Let f be defined on [a, b]. The Riemann sum is Σ f(c_k) Δx_k where Δx_k = x_k - x_{k-1} and c_k ∈ [x_{k-1}, x_k]. If max Δx_k → 0, the limit of Riemann sums is the definite integral ∫_a^b f(x) dx.",
        "chapter": "5",
    },
    {
        "query": "What is the ratio test for convergence of series?",
        "truth": "Let Σ a_n be a series with nonzero terms. Let ρ = lim(n→∞) |a_{n+1}/a_n|. If ρ < 1 the series converges absolutely. If ρ > 1 the series diverges. If ρ = 1 the test is inconclusive.",
        "chapter": "10",
    },
    {
        "query": "State the divergence theorem (Gauss' theorem).",
        "truth": "The divergence theorem: The outward flux of a vector field F across a closed surface S equals the triple integral of div F over the region D enclosed by S: ∬_S F·n dσ = ∭_D ∇·F dV.",
        "chapter": "16",
    },
]

print("=" * 70)
print("DEEP EVALUATION: Generated Answers vs PDF Ground Truth")
print("=" * 70)

correct_count = 0
total = len(GROUND_TRUTHS)

def grade_answer(answer: str, truth: str, query: str) -> dict:
    answer_lower = answer.lower()
    truth_lower = truth.lower()

    # Extract key terms from truth
    truth_terms = set(re.findall(r'\b[a-zA-Z+]+(?:\([^)]*\))?[a-zA-Z+]*\b', truth_lower)) - {
        'the', 'a', 'an', 'is', 'of', 'in', 'to', 'and', 'for', 'if', 'on', 'at', 'by', 'be'
    }
    # Also include mathematical notation
    math_patterns = re.findall(r'\\[a-zA-Z]+|∫|∬|∭|∑|→|·|⋅|∇|Σ|Δ|ρ', answer_lower + truth_lower)

    matches = sum(1 for t in truth_terms if t in answer_lower)
    precision = matches / max(len(truth_terms), 1)
    
    # Check for key mathematical correctness
    has_correct_math = True

    # Specific content checks
    checks = {}
    if "derivative of sin" in query.lower():
        checks["cos"] = "cos" in answer_lower or "cosine" in answer_lower
    elif "fundamental theorem" in query.lower():
        checks["F(b) - F(a)"] = ("f(b)" in answer_lower and "f(a)" in answer_lower) or "difference" in answer_lower
        checks["antiderivative"] = "antiderivative" in answer_lower
    elif "l'hôpital" in query.lower() or "l'hopital" in query.lower():
        checks["f'/g'"] = "f'" in answer_lower or "f'(x)/g'(x)" in answer_lower
    elif "chain rule" in query.lower():
        checks["dy/du"] = "dy/du" in answer_lower or "f'(g(x))" in answer_lower or "outer" in answer_lower
    elif "integration by parts" in query.lower():
        checks["uv - ∫v du"] = "uv" in answer_lower and ("∫v du" in answer_lower or "integral of v du" in answer_lower)
    elif "riemann sum" in query.lower():
        checks["∑"] = "∑" in answer_lower or "sigma" in answer_lower or "sum" in answer_lower
    elif "ratio test" in query.lower():
        checks["ρ < 1"] = ("ρ < 1" in answer_lower or "rho" in answer_lower or "less than 1" in answer_lower) and "converge" in answer_lower
        checks["ρ > 1"] = ("ρ > 1" in answer_lower or "greater than 1" in answer_lower) and "diverge" in answer_lower
    elif "divergence theorem" in query.lower():
        checks["flux"] = "flux" in answer_lower
        checks["triple integral"] = "triple integral" in answer_lower

    all_checks_passed = all(checks.values()) if checks else True
    term_coverage = precision > 0.4

    is_correct = all_checks_passed and term_coverage
    
    # Heuristic grade
    if is_correct:
        grade = "✅ CORRECT"
    elif precision > 0.3:
        grade = "⚠️ PARTIAL"
    else:
        grade = "❌ WRONG"

    return {
        "grade": grade,
        "precision": precision,
        "checks_passed": sum(1 for v in checks.values() if v),
        "checks_total": len(checks),
        "all_checks": all_checks_passed,
    }

for i, item in enumerate(GROUND_TRUTHS):
    query = item["query"]
    truth = item["truth"]

    # Retrieve and rerank
    docs = store.similarity_search(query, k=30)
    texts = [d.page_content for d in docs]
    reranked = reranker.rerank(query, texts, top_k=5)
    kept_set = {t for t, _ in reranked}
    reranked_docs = [d for d in docs if d.page_content in kept_set]

    # Generate
    try:
        result = generator.generate(query, reranked_docs)
        answer = result["answer"]
        citations = result["citations"]
    except Exception as e:
        answer = f"ERROR: {e}"
        citations = []

    # Grade
    grade_info = grade_answer(answer, truth, query)
    is_correct = grade_info["grade"] == "✅ CORRECT"
    if is_correct:
        correct_count += 1

    print(f"\n  [{i+1}/{total}] {grade_info['grade']}")
    print(f"  Q: {query}")
    print(f"  A: {answer[:150]}...")
    print(f"  Truth: {truth[:100]}...")
    print(f"  Precision: {grade_info['precision']:.2f} | Checks: {grade_info['checks_passed']}/{grade_info['checks_total']} | Citations: {len(citations)}")

accuracy = correct_count / total * 100
print(f"\n{'='*70}")
print(f"FINAL RESULT: {accuracy:.1f}% accuracy ({correct_count}/{total})")
print(f"{'='*70}")

if accuracy >= 95:
    print("✅ PASSES 95% threshold! Excellent retrieval + generation quality.")
else:
    print(f"❌ Below 95% threshold. ({accuracy:.1f}% < 95%)")
    print("Proceeding to full code review and optimization plan...")
