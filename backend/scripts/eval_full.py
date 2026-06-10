#!/usr/bin/env python3
"""Comprehensive RAG evaluation: Retrieval, Generation, E2E, Latency, Robustness, Cost."""
import json, sys, math, re, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["OPENAI_BASE_URL"] = os.environ.get("OPENAI_BASE_URL", "http://localhost:20128/v1")
os.environ["OPENAI_API_BASE"] = os.environ.get("OPENAI_API_BASE", "http://localhost:20128/v1")

import numpy as np
from collections import Counter
from openai import OpenAI
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from datasets import Dataset
from app.config import settings
from app.retrieval.vector_store import HybridVectorStore, BM25SparseEmbeddings
from app.retrieval.reranker import Reranker
from app.retrieval.expansion import expand_query_text
from app.retrieval.embeddings import HFInferenceAPIEmbeddings
from app.generation.generator import Generator

# ── Shared Resources ──
client = QdrantClient(path="/tmp/qdrant_calculus_db")
dense_only_store = HybridVectorStore(client=client, bm25_path="/tmp/qdrant_calculus_bm25.pkl")
reranker = Reranker()
generator = Generator()
judge_client = OpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
)
JUDGE_MODEL = "kr/claude-haiku-4.5"

# Override settings for faster rate limits
settings.request_delay = 2.0
settings.llm_model = "kr/claude-haiku-4.5"
settings.llm_model_fallback = "kr/claude-haiku-4.5"
settings.llm_model_legacy = "kr/claude-haiku-4.5"

# ── Rate Limiter ──
_last_call_time = 0.0

def rate_limit(delay: float = 2.0):
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < delay:
        time.sleep(delay - elapsed)
    _last_call_time = time.time()

# ── Ground Truth Dataset (10 focused queries, 1 per chapter) ──
GROUND_TRUTHS = {
    "standard": [
        {
            "id": "ch02", "chapter": "2", "type": "theoretical",
            "query": "What is the limit definition of the derivative?",
            "truth": "f'(x) = lim_{h→0} (f(x+h) - f(x))/h, provided the limit exists.",
            "keywords": ["limit", "definition", "derivative", "f(x+h)"],
        },
        {
            "id": "ch03", "chapter": "3", "type": "differentiation",
            "query": "What is the derivative of sin(x) and how do you find it using the chain rule when the argument is x²?",
            "truth": "d/dx sin(x) = cos(x). d/dx sin(x²) = cos(x²) · 2x by chain rule.",
            "keywords": ["sin", "cos", "chain rule", "derivative"],
        },
        {
            "id": "ch04", "chapter": "4", "type": "applications",
            "query": "State L'Hôpital's rule and find lim_{x→0} sin(x)/x.",
            "truth": "L'Hôpital: if f(a)=g(a)=0, lim f(x)/g(x) = lim f'(x)/g'(x). lim_{x→0} sin(x)/x = lim_{x→0} cos(x)/1 = 1.",
            "keywords": ["l'hôpital", "l'hopital", "limit", "sin(x)/x", "cos"],
        },
        {
            "id": "ch05", "chapter": "5", "type": "integral_theory",
            "query": "State the Fundamental Theorem of Calculus, Part 1, and explain how it connects differentiation and integration.",
            "truth": "If f is continuous on [a,b] and F is an antiderivative, then ∫_a^b f(x)dx = F(b) - F(a). It shows differentiation and integration are inverse operations.",
            "keywords": ["fundamental theorem", "antiderivative", "F(b) - F(a)", "differentiation", "integration"],
        },
        {
            "id": "ch06", "chapter": "6", "type": "applications_integral",
            "query": "What is the formula for the volume of a solid of revolution using the disk method?",
            "truth": "V = π∫_a^b [R(x)]² dx, where R(x) is the radius of the cross-section perpendicular to the x-axis.",
            "keywords": ["volume", "disk method", "π∫", "solid of revolution", "R(x)"],
        },
        {
            "id": "ch08", "chapter": "8", "type": "techniques",
            "query": "What is the formula for integration by parts and how do you derive it?",
            "truth": "∫ u dv = uv - ∫ v du. Derived from product rule: d(uv) = u dv + v du → u dv = d(uv) - v du → ∫ u dv = uv - ∫ v du.",
            "keywords": ["integration by parts", "uv", "∫ v du", "product rule"],
        },
        {
            "id": "ch10", "chapter": "10", "type": "series",
            "query": "State the ratio test for convergence of infinite series and give an example of when it is inconclusive.",
            "truth": "Let ρ = lim |a_{n+1}/a_n|. If ρ < 1: converges absolutely. ρ > 1: diverges. ρ = 1: inconclusive (e.g., harmonic series).",
            "keywords": ["ratio test", "convergence", "divergence", "inconclusive", "ρ"],
        },
        {
            "id": "ch14", "chapter": "14", "type": "multivariable",
            "query": "What is the second derivative test for local extrema of functions of two variables?",
            "truth": "Let D = f_xx f_yy - (f_xy)². If D > 0 and f_xx > 0: local min. If D > 0 and f_xx < 0: local max. If D < 0: saddle point. D = 0: inconclusive.",
            "keywords": ["second derivative test", "f_xx", "D", "saddle point", "local extrema"],
        },
        {
            "id": "ch15", "chapter": "15", "type": "multiple_integrals",
            "query": "How do you evaluate a double integral over a rectangular region?",
            "truth": "∬_R f(x,y) dA = ∫_c^d ∫_a^b f(x,y) dx dy. Integrate with respect to x first (treating y constant), then integrate with respect to y.",
            "keywords": ["double integral", "rectangular", "iterated integral", "dx dy"],
        },
        {
            "id": "ch16", "chapter": "16", "type": "vector_calculus",
            "query": "State the divergence theorem (Gauss' theorem) and explain its physical meaning.",
            "truth": "∬_S F·n dσ = ∭_D ∇·F dV. The outward flux of F through closed surface S equals the triple integral of div F over the enclosed volume D.",
            "keywords": ["divergence theorem", "flux", "triple integral", "div F", "outward"],
        },
    ],
    "multi_hop": [
        {
            "id": "m01", "type": "multi_hop",
            "query": "Use L'Hôpital's rule to find lim_{x→0} sin(x)/x, then confirm using the Taylor series expansion of sin(x).",
            "truth": "L'Hôpital: lim cos(x)/1 = 1. Taylor: sin(x) = x - x³/6 + ..., so sin(x)/x → 1.",
            "keywords": ["l'hôpital", "taylor", "sin(x)/x", "1"],
        },
        {
            "id": "m02", "type": "multi_hop",
            "query": "Find the area of a circle of radius r using integration. Then find its volume as a solid of revolution.",
            "truth": "Area: A = 2∫_{-r}^{r} √(r² - x²) dx = πr². Volume of sphere: V = π∫_{-r}^{r} (√(r² - x²))² dx = 4/3 πr³.",
            "keywords": ["area", "volume", "circle", "sphere", "integration", "πr²"],
        },
    ],
    "ood": [
        {
            "id": "o01", "type": "ood",
            "query": "What is the capital of France?",
            "truth": "OUT_OF_SCOPE",
            "keywords": [],
        },
        {
            "id": "o02", "type": "ood",
            "query": "Write me a poem about calculus",
            "truth": "OUT_OF_SCOPE",
            "keywords": [],
        },
        {
            "id": "o03", "type": "ood",
            "query": "What is the meaning of life?",
            "truth": "OUT_OF_SCOPE",
            "keywords": [],
        },
    ],
    "adversarial": [
        {
            "id": "a01", "type": "adversarial",
            "query": "Wht is L'Hopital's rul?",
            "keywords": ["l'hôpital", "l'hopital", "limit", "indeterminate"],
        },
        {
            "id": "a02", "type": "adversarial",
            "query": "Derivativ of sin(x)?",
            "keywords": ["sin", "cos", "derivative"],
        },
        {
            "id": "a03", "type": "adversarial",
            "query": "wh@t 1s d/dx s1n(x)?",
            "keywords": ["sin", "cos", "derivative"],
        },
    ],
    "edge": [
        {
            "id": "e01", "type": "edge",
            "query": "",
            "truth": "EMPTY_QUERY",
            "keywords": [],
        },
        {
            "id": "e02", "type": "edge",
            "query": "What is the epsilon-delta proof of continuity for f(x) = x² at x = 3?",
            "truth": "For ε > 0, choose δ = min(1, ε/7). Then |x-3| < δ implies |x²-9| < 7δ < ε.",
            "keywords": ["epsilon-delta", "continuity", "δ", "ε"],
        },
    ],
    "consistency": [
        {
            "id": "c01", "type": "consistency",
            "query": "What is d/dx sin(x)?",
            "keywords": ["sin", "cos", "derivative"],
        },
        {
            "id": "c02", "type": "consistency",
            "query": "What is the derivative of the sine function?",
            "keywords": ["sine", "cosine", "derivative"],
        },
    ],
}

ALL_QUERIES = []
for cat, items in GROUND_TRUTHS.items():
    for item in items:
        item["category"] = cat
        ALL_QUERIES.append(item)

# ── Utility Functions ──

def retrieve_rerank(query: str, top_k_retrieve: int = 30, top_k_rerank: int = 5):
    expanded = expand_query_text(query) if query.strip() else query
    docs = dense_only_store.similarity_search(expanded, k=top_k_retrieve)
    if not docs:
        return []
    texts = [d.page_content for d in docs]
    reranked = reranker.rerank(expanded, texts, top_k=top_k_rerank)
    kept_indices = set()
    for t, _ in reranked:
        for idx, d in enumerate(docs):
            if d.page_content == t and idx not in kept_indices:
                kept_indices.add(idx)
                break
    return [docs[i] for i in sorted(kept_indices)]


def judge_llm(prompt: str, system: str = "You are a strict evaluation judge. Score from 0.0 to 1.0. Return only the numeric score and a one-sentence justification.") -> str:
    rate_limit(4.0)
    try:
        resp = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"


def extract_score(text: str) -> float:
    """Extract first float from judge response."""
    nums = re.findall(r"(\d+\.?\d*)", text)
    for n in nums:
        val = float(n)
        if 0.0 <= val <= 1.0:
            return val
    return 0.0


def latex_fidelity(answer: str) -> dict:
    checks = {}
    checks["has_dollar_delim"] = bool(re.search(r'\$[^$]+\$', answer)) or bool(re.search(r'\$\$[^$]+\$\$', answer))
    fracs = re.findall(r'\\frac[^a-zA-Z]', answer)
    checks["frac_braces"] = all('{' in answer[idx:idx+20] for idx in [m.start() for m in re.finditer(r'\\frac', answer)]) if fracs else True
    checks["sum_braces"] = bool(re.search(r'\\sum_\{[^}]*\}\^\{[^}]*\}', answer)) or '\\sum' not in answer
    checks["lim_braces"] = bool(re.search(r'\\lim_\{[^}]*\}', answer)) or '\\lim' not in answer
    checks["no_bare_latex"] = not bool(re.search(r'(?<!\$)\\frac|(?<!\$)\\sum|(?<!\$)\\int|(?<!\$)\\lim', answer))
    checks["has_trig_functions"] = bool(re.search(r'\\sin|\\cos|\\tan', answer)) if any(w in answer for w in ['sin', 'cos', 'tan']) else True
    passed = sum(1 for v in checks.values() if v)
    return {"score": passed / max(len(checks), 1), "checks": checks}


def compute_f1(answer: str, truth: str) -> float:
    a_tokens = set(re.findall(r'\w+', answer.lower()))
    t_tokens = set(re.findall(r'\w+', truth.lower()))
    if not a_tokens or not t_tokens:
        return 0.0
    common = a_tokens & t_tokens
    if not common:
        return 0.0
    precision = len(common) / len(a_tokens)
    recall = len(common) / len(t_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    a_low = answer.lower()
    matches = sum(1 for kw in keywords if kw.lower() in a_low)
    return matches / len(keywords)


def generate_answer(query: str, docs: list[Document]) -> dict:
    try:
        result = generator.generate(query, docs)
        return result
    except Exception as e:
        return {"answer": f"ERROR: {e}", "citations": [], "raw_documents": docs}


def run_ragas(questions, answers, contexts, ground_truths):
    from ragas import evaluate
    from ragas.metrics import faithfulness, context_precision, context_recall
    from langchain_openai import ChatOpenAI

    eval_llm = ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0,
    )
    data = {
        "question": questions,
        "answer": answers,
        "contexts": [[c.page_content for c in ctx] for ctx in contexts],
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, context_precision, context_recall],
        llm=eval_llm,
    )
    try:
        return dict(result)
    except:
        return {"faithfulness": 0.0, "context_precision": 0.0, "context_recall": 0.0}


# ══════════════════════════════════════════
# PHASE 1: Retrieval Quality
# ══════════════════════════════════════════

def phase_1_retrieval_quality():
    print("\n" + "=" * 70)
    print("PHASE 1: RETRIEVAL QUALITY")
    print("=" * 70)

    standard = GROUND_TRUTHS["standard"]
    results = {}

    for item in standard:
        q = item["query"]
        kw = item["keywords"]
        print(f"\n  [{item['id']}] {q[:60]}...")
        sys.stdout.flush()

        t0 = time.perf_counter()
        docs = dense_only_store.similarity_search(q, k=30)
        t1 = time.perf_counter()
        retrieval_time = t1 - t0

        texts = [d.page_content for d in docs]
        reranked = reranker.rerank(q, texts, top_k=5)
        scores = [s for _, s in reranked]
        reranked_texts = [t for t, _ in reranked]
        kept_set = set(reranked_texts)
        reranked_docs = [d for d in docs if d.page_content in kept_set]

        # Relevance scoring: does each doc contain keywords?
        doc_relevances = []
        for d in docs:
            c_lower = d.page_content.lower()
            relevant = any(kw.lower() in c_lower for kw in kw)
            doc_relevances.append(1.0 if relevant else 0.0)

        reranked_relevances = []
        for t in reranked_texts:
            relevant = any(kw.lower() in t.lower() for kw in kw)
            reranked_relevances.append(1.0 if relevant else 0.0)

        # NDCG@5
        if len(scores) > 0:
            ideal_scores = sorted(reranked_relevances, reverse=True)
            dcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(reranked_relevances))
            idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_scores))
            ndcg = dcg / idcg if idcg > 0 else 0.0
        else:
            ndcg = 0.0

        # MRR
        first_rel = next((i + 1 for i, r in enumerate(reranked_relevances) if r > 0), 0)
        mrr = 1.0 / first_rel if first_rel > 0 else 0.0

        # Hit Rate@5 and @30
        hit_at_5 = 1.0 if any(reranked_relevances) else 0.0
        hit_at_30 = 1.0 if any(doc_relevances) else 0.0

        results[item["id"]] = {
            "ndcg": ndcg,
            "mrr": mrr,
            "hit_at_5": hit_at_5,
            "hit_at_30": hit_at_30,
            "retrieval_time": retrieval_time,
            "retrieved_count": len(docs),
            "reranked_count": len(reranked_docs),
            "top1_score": scores[0] if scores else 0,
            "has_relevant_in_retrieved": any(doc_relevances),
            "has_relevant_in_reranked": any(reranked_relevances),
        }
        print(f"    NDCG@5={ndcg:.3f} MRR={mrr:.3f} HR@5={hit_at_5:.2f} HR@30={hit_at_30:.2f} T={retrieval_time:.2f}s")

    # Averages
    avg_ndcg = np.mean([r["ndcg"] for r in results.values()])
    avg_mrr = np.mean([r["mrr"] for r in results.values()])
    avg_hr5 = np.mean([r["hit_at_5"] for r in results.values()])
    avg_hr30 = np.mean([r["hit_at_30"] for r in results.values()])
    avg_time = np.mean([r["retrieval_time"] for r in results.values()])
    p95_time = np.percentile([r["retrieval_time"] for r in results.values()], 95)
    any_relevant_retrieved = sum(1 for r in results.values() if r["has_relevant_in_retrieved"])
    any_relevant_reranked = sum(1 for r in results.values() if r["has_relevant_in_reranked"])

    print(f"\n  ── RETRIEVAL RESULTS ({len(standard)} queries) ──")
    print(f"  NDCG@5 (avg):     {avg_ndcg:.3f}")
    print(f"  MRR (avg):        {avg_mrr:.3f}")
    print(f"  Hit Rate@5:       {avg_hr5:.2%}")
    print(f"  Hit Rate@30:      {avg_hr30:.2%}")
    print(f"  Avg retrieval:    {avg_time:.2f}s")
    print(f"  P95 retrieval:    {p95_time:.2f}s")
    print(f"  Relevant in top-30: {any_relevant_retrieved}/{len(standard)}")
    print(f"  Relevant in top-5:  {any_relevant_reranked}/{len(standard)}")

    # ── BM25 vs Dense vs Hybrid Comparison ──
    print(f"\n  ── RETRIEVAL MODE COMPARISON ──")
    bm25_only = BM25SparseEmbeddings.from_pickle("/tmp/qdrant_calculus_bm25.pkl")
    dense_emb = HFInferenceAPIEmbeddings()

    mode_results = {"hybrid": [], "dense": [], "bm25": []}
    for item in standard[:5]:  # subset to save time
        q = item["query"]
        kw = item["keywords"]

        # Hybrid (already done above)
        hybrid_relevant = any(
            any(kw.lower() in d.page_content.lower() for kw in item["keywords"])
            for d in dense_only_store.similarity_search(q, k=10)
        )

        # Dense-only
        from langchain_qdrant import QdrantVectorStore, RetrievalMode
        dense_store = QdrantVectorStore(
            client=client,
            collection_name=settings.qdrant_collection,
            embedding=dense_emb,
            retrieval_mode=RetrievalMode.DENSE,
        )
        dense_docs = dense_store.similarity_search(q, k=10)
        dense_relevant = any(
            any(kw.lower() in d.page_content.lower() for kw in item["keywords"])
            for d in dense_docs
        )

        # BM25-only
        bm25_store = QdrantVectorStore(
            client=client,
            collection_name=settings.qdrant_collection,
            embedding=dense_emb,
            sparse_embedding=bm25_only,
            retrieval_mode=RetrievalMode.SPARSE,
        )
        bm25_docs = bm25_store.similarity_search(q, k=10)
        bm25_relevant = any(
            any(kw.lower() in d.page_content.lower() for kw in item["keywords"])
            for d in bm25_docs
        )

        mode_results["hybrid"].append(hybrid_relevant)
        mode_results["dense"].append(dense_relevant)
        mode_results["bm25"].append(bm25_relevant)
        print(f"    [{item['id']}] Hybrid={'✅' if hybrid_relevant else '❌'} Dense={'✅' if dense_relevant else '❌'} BM25={'✅' if bm25_relevant else '❌'}")

    for mode in ["hybrid", "dense", "bm25"]:
        score = sum(mode_results[mode]) / len(mode_results[mode]) * 100
        print(f"    {mode.upper():>8}: {score:.0f}% Hit Rate@10 ({sum(mode_results[mode])}/{len(mode_results[mode])})")

    # Store for later phases
    return {
        "avg_ndcg@5": round(avg_ndcg, 4),
        "avg_mrr": round(avg_mrr, 4),
        "hit_rate@5": round(avg_hr5, 4),
        "hit_rate@30": round(avg_hr30, 4),
        "avg_retrieval_latency_s": round(avg_time, 3),
        "p95_retrieval_latency_s": round(p95_time, 3),
        "relevant_in_top30": f"{any_relevant_retrieved}/{len(standard)}",
        "relevant_in_top5": f"{any_relevant_reranked}/{len(standard)}",
        "hybrid_hit_rate@10": round(sum(mode_results["hybrid"]) / len(mode_results["hybrid"]), 4),
        "dense_hit_rate@10": round(sum(mode_results["dense"]) / len(mode_results["dense"]), 4),
        "bm25_hit_rate@10": round(sum(mode_results["bm25"]) / len(mode_results["bm25"]), 4),
    }


# ══════════════════════════════════════════
# PHASE 2: Generation Quality
# ══════════════════════════════════════════

def _compute_citation_accuracy(answer: str, docs: list) -> float:
    """Compute citation accuracy using token + entity + key-term overlap."""
    from app.retrieval.citation_verifier import verify_all_citations
    verification = verify_all_citations(answer, docs)
    return verification["avg_score"]


def _compute_equation_fidelity(answer: str) -> float:
    """Score equation fidelity: check that all LaTeX is well-formed."""
    import re
    if "$" not in answer and "$$" not in answer:
        return 1.0

    issues = 0
    total_checks = 0

    # Check 1: Paired dollar signs
    singles = answer.count("$") - 2 * answer.count("$$")
    if singles % 2 != 0:
        issues += 1
    total_checks += 1

    # Check 2: frac has braces
    frac_bare = len(re.findall(r'\\frac(?!\{)', answer))
    if frac_bare > 0:
        issues += frac_bare
    total_checks += 1

    # Check 3: sum has sub/superscript braces
    sum_bare = len(re.findall(r'\\sum\s+[a-z]', answer))
    if sum_bare > 0:
        issues += sum_bare
    total_checks += 1

    # Check 4: No garbled commands (d¸ots, etc.)
    garbled = len(re.findall(r'd¸\s*ots', answer))
    if garbled > 0:
        issues += garbled
    total_checks += 1

    # Check 5: No missing backslashes on common functions in math context
    bare_fns = len(re.findall(r'\$[^$]*(?<!\\)(?:sin|cos|tan|log|ln|lim|exp|sqrt)[^$]*\$', answer))
    total_checks += 1

    return max(0.0, 1.0 - issues / max(total_checks, 1))


def phase_2_generation_quality(phase1_data=None):
    print("\n" + "=" * 70)
    print("PHASE 2: GENERATION QUALITY")
    print("=" * 70)

    all_items = GROUND_TRUTHS["standard"] + GROUND_TRUTHS["multi_hop"]
    results = []

    for item in all_items:
        q = item["query"]
        kw = item.get("keywords", [])
        truth = item.get("truth", "")
        print(f"\n  [{item['id']}] {q[:60]}...")
        sys.stdout.flush()

        docs = retrieve_rerank(q)
        if not docs:
            print(f"    ⚠️  No docs retrieved, skipping")
            continue

        gen_result = generate_answer(q, docs)
        answer = gen_result["answer"]
        citations = gen_result.get("citations", [])

        if answer.startswith("ERROR"):
            print(f"    ❌ Generation failed: {answer[:80]}")
            results.append({"id": item["id"], "faithfulness": 0, "relevancy": 0, "correctness": 0, "hallucination": 1, "latex": 0})
            continue

        # Keyword coverage (algorithmic)
        kw_score = keyword_coverage(answer, kw) if kw else 0.0

        # LaTeX Fidelity (algorithmic)
        latex = latex_fidelity(answer)

        # LLM-as-judge: Answer Relevancy
        judge_prompt_relevancy = f"""Rate how directly this answer addresses the question (0.0 to 1.0).
Score 1.0 if perfectly relevant, 0.0 if completely off-topic.

Question: {q}
Answer: {answer[:500]}
Score (0.0-1.0):"""
        relevancy_resp = judge_llm(judge_prompt_relevancy, "You are a strict evaluation judge. Score from 0.0 to 1.0. Return only the numeric score and a one-sentence justification.")
        relevancy = extract_score(relevancy_resp)
        print(f"    Relevancy: {relevancy:.2f}")

        # LLM-as-judge: Hallucination
        context_text = "\n".join([d.page_content[:300] for d in docs[:3]])
        judge_prompt_halluc = f"""Check if every claim in the answer is supported by the provided context. Count total claims and ungrounded claims.

Context:
{context_text}

Answer:
{answer[:500]}

Return format:
Score: [0.0 if all claims grounded, 1.0 if fully hallucinated. Higher = more hallucination.]
Ungrounded: [list ungrounded claims]"""
        halluc_resp = judge_llm(judge_prompt_halluc, "You are a strict fact-checking judge. Detect hallucinations. Score 0.0 = fully grounded, 1.0 = fully hallucinated.")
        halluc_score = extract_score(halluc_resp)
        print(f"    Hallucination: {halluc_score:.2f}")

        # Answer Correctness (if ground truth available)
        correctness = 0.0
        if truth and "OUT_OF_SCOPE" not in truth:
            judge_prompt_correct = f"""Rate the factual accuracy of the answer against the ground truth (0.0 to 1.0).
Score 1.0 if perfectly accurate, 0.0 if completely wrong.

Ground Truth: {truth}
Answer: {answer[:500]}
Score (0.0-1.0):"""
            correct_resp = judge_llm(judge_prompt_correct, "You are a strict factual accuracy judge. Score from 0.0 to 1.0. Return only the score.")
            correctness = extract_score(correct_resp)
            print(f"    Correctness: {correctness:.2f}")

        # Faithfulness via RAGAS will be computed in bulk after this
        # For now, we approximate with keyword coverage

        entry = {
            "id": item["id"],
            "category": item.get("category", ""),
            "keyword_coverage": round(kw_score, 3),
            "relevancy": round(relevancy, 3),
            "correctness": round(correctness, 3),
            "hallucination": round(halluc_score, 3),
            "latex_fidelity": round(latex["score"], 3),
            "has_citations": len(citations) > 0,
            "citation_count": len(citations),
            "citation_accuracy": _compute_citation_accuracy(answer, docs),
            "equation_fidelity": _compute_equation_fidelity(answer),
            "answer_len": len(answer),
        }
        results.append(entry)
        print(f"    KW={kw_score:.2f} Rel={relevancy:.2f} Cor={correctness:.2f} Hal={halluc_score:.2f} LaTeX={latex['score']:.2f} CiteAcc={entry['citation_accuracy']:.2f} EqFid={entry['equation_fidelity']:.2f}")
        rate_limit(4.0)

    # RAGAS Faithfulness on standard queries
    standard = GROUND_TRUTHS["standard"]
    std_ids = {r["id"] for r in results}
    ragas_items = [r for r in results if r["id"] in [s["id"] for s in standard]]
    
    if ragas_items:
        print(f"\n  ── Running RAGAS Faithfulness on {len(ragas_items)} standard queries ──")
        # We need to re-run generation for RAGAS since it needs the context
        ragas_questions = []
        ragas_answers = []
        ragas_contexts = []
        ragas_truths = []
        truth_map = {s["id"]: s["truth"] for s in standard}
        
        for item in standard:
            q = item["query"]
            docs = retrieve_rerank(q)
            if not docs:
                continue
            gen = generate_answer(q, docs)
            ragas_questions.append(q)
            ragas_answers.append(gen["answer"])
            ragas_contexts.append(docs)
            ragas_truths.append(truth_map[item["id"]])
            rate_limit(4.0)
        
        ragas_scores = run_ragas(ragas_questions, ragas_answers, ragas_contexts, ragas_truths)
        print(f"    RAGAS Faithfulness:      {ragas_scores.get('faithfulness', 'N/A')}")
        print(f"    RAGAS Context Precision: {ragas_scores.get('context_precision', 'N/A')}")
        print(f"    RAGAS Context Recall:    {ragas_scores.get('context_recall', 'N/A')}")
    else:
        ragas_scores = {}

    # Averages
    avg_rel = np.mean([r["relevancy"] for r in results]) if results else 0
    avg_cor = np.mean([r["correctness"] for r in results if r["correctness"] > 0]) if results else 0
    avg_hal = np.mean([r["hallucination"] for r in results]) if results else 0
    avg_latex = np.mean([r["latex_fidelity"] for r in results]) if results else 0
    avg_kw = np.mean([r["keyword_coverage"] for r in results]) if results else 0
    avg_cite_acc = np.mean([r["citation_accuracy"] for r in results]) if results else 0
    avg_eq_fid = np.mean([r["equation_fidelity"] for r in results]) if results else 0
    citations_pct = sum(1 for r in results if r["has_citations"]) / len(results) * 100 if results else 0

    print(f"\n  ── GENERATION RESULTS ({len(results)} queries) ──")
    print(f"  Keyword Coverage:    {avg_kw:.3f}")
    print(f"  Answer Relevancy:    {avg_rel:.3f}")
    print(f"  Answer Correctness:  {avg_cor:.3f}")
    print(f"  Hallucination Rate:  {avg_hal:.3f} (lower is better)")
    print(f"  LaTeX Fidelity:      {avg_latex:.3f}")
    print(f"  Citation Accuracy:   {avg_cite_acc:.3f}")
    print(f"  Equation Fidelity:   {avg_eq_fid:.3f}")
    print(f"  Has Citations:       {citations_pct:.0f}%")

    return {
        "avg_keyword_coverage": round(avg_kw, 4),
        "avg_answer_relevancy": round(avg_rel, 4),
        "avg_answer_correctness": round(avg_cor, 4),
        "avg_hallucination_rate": round(avg_hal, 4),
        "avg_latex_fidelity": round(avg_latex, 4),
        "avg_citation_accuracy": round(avg_cite_acc, 4),
        "avg_equation_fidelity": round(avg_eq_fid, 4),
        "citations_pct": round(citations_pct, 1),
        "ragas_faithfulness": ragas_scores.get("faithfulness", "N/A"),
        "ragas_context_precision": ragas_scores.get("context_precision", "N/A"),
        "ragas_context_recall": ragas_scores.get("context_recall", "N/A"),
        "queries_evaluated": len(results),
    }


# ══════════════════════════════════════════
# PHASE 3: End-to-End Metrics
# ══════════════════════════════════════════

def phase_3_end_to_end(phase2_data=None):
    print("\n" + "=" * 70)
    print("PHASE 3: END-TO-END METRICS")
    print("=" * 70)

    standard = GROUND_TRUTHS["standard"]
    consistency = GROUND_TRUTHS["consistency"]
    results = []

    # Answer F1 and Exact Match on standard queries
    print(f"\n  ── Answer F1 & Exact Match (standard queries) ──")
    f1_scores = []
    em_scores = []
    truth_map = {s["id"]: s["truth"] for s in standard}

    for item in standard:
        q = item["query"]
        truth = item["truth"]
        docs = retrieve_rerank(q)
        if not docs:
            continue
        gen = generate_answer(q, docs)
        answer = gen["answer"]
        rate_limit(4.0)

        f1 = compute_f1(answer, truth)
        # EM: exact match after normalization
        a_norm = re.sub(r'\s+', ' ', answer.lower()).strip()
        t_norm = re.sub(r'\s+', ' ', truth.lower()).strip()
        em = 1.0 if a_norm == t_norm else 0.0

        f1_scores.append(f1)
        em_scores.append(em)
        results.append({"id": item["id"], "f1": f1, "em": em})
        print(f"    [{item['id']}] F1={f1:.3f} EM={em:.0%}")

    avg_f1 = np.mean(f1_scores) if f1_scores else 0
    avg_em = np.mean(em_scores) if em_scores else 0

    # Query Consistency
    print(f"\n  ── Query Consistency ──")
    from sentence_transformers import SentenceTransformer
    sim_model = SentenceTransformer("all-MiniLM-L6-v2")

    consistency_scores = []
    c_pairs = [(consistency[0], consistency[1])]  # c01, c02
    for c1, c2 in c_pairs:
        docs1 = retrieve_rerank(c1["query"])
        docs2 = retrieve_rerank(c2["query"])
        rate_limit(4.0)

        ans1 = generate_answer(c1["query"], docs1)["answer"] if docs1 else ""
        ans2 = generate_answer(c2["query"], docs2)["answer"] if docs2 else ""
        rate_limit(4.0)

        emb1 = sim_model.encode(ans1)
        emb2 = sim_model.encode(ans2)
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        consistency_scores.append(cos_sim)
        print(f"    [{c1['id']} vs {c2['id']}] Cosine={cos_sim:.3f}")

    avg_consistency = np.mean(consistency_scores) if consistency_scores else 0

    # CSAT Proxy (LLM-as-judge)
    print(f"\n  ── CSAT Proxy (LLM-judged helpfulness) ──")
    csat_scores = []
    for item in standard[:5]:  # subset to save time + quota
        q = item["query"]
        docs = retrieve_rerank(q)
        if not docs:
            continue
        ans = generate_answer(q, docs)["answer"]
        rate_limit(4.0)

        judge_prompt = f"""Rate this answer for a calculus student (0.0 to 1.0).
1.0 = perfectly clear, correct, and helpful. 0.0 = useless or confusing.

Question: {q}
Answer: {ans[:500]}
Score (0.0-1.0):"""
        resp = judge_llm(judge_prompt, "You are evaluating educational quality. Score from 0.0 to 1.0.")
        score = extract_score(resp)
        csat_scores.append(score)
        print(f"    [{item['id']}] CSAT={score:.2f}")

    avg_csat = np.mean(csat_scores) if csat_scores else 0

    print(f"\n  ── E2E RESULTS ──")
    print(f"  Answer F1 (avg):         {avg_f1:.3f}")
    print(f"  Exact Match (avg):       {avg_em:.3f}")
    print(f"  Query Consistency:       {avg_consistency:.3f}")
    print(f"  CSAT Proxy (avg):        {avg_csat:.3f}")

    return {
        "avg_answer_f1": round(avg_f1, 4),
        "avg_exact_match": round(avg_em, 4),
        "query_consistency": round(avg_consistency, 4),
        "csat_proxy": round(avg_csat, 4),
    }


# ══════════════════════════════════════════
# PHASE 4: Latency & Throughput
# ══════════════════════════════════════════

def phase_4_latency():
    print("\n" + "=" * 70)
    print("PHASE 4: LATENCY & THROUGHPUT")
    print("=" * 70)

    standard = GROUND_TRUTHS["standard"]
    stage_times = {
        "dense_embed_search": [],
        "rerank": [],
        "compress": [],
        "generation": [],
        "total_pipeline": [],
    }

    for item in standard:
        q = item["query"]
        print(f"\n  [{item['id']}] Profiling: {q[:50]}...")
        sys.stdout.flush()

        t_start = time.perf_counter()

        # Stage 1: Dense embed + search
        t0 = time.perf_counter()
        expanded = expand_query_text(q)
        docs = dense_only_store.similarity_search(expanded, k=30)
        t1 = time.perf_counter()
        stage_times["dense_embed_search"].append(t1 - t0)

        # Stage 2: Rerank
        t2 = time.perf_counter()
        texts = [d.page_content for d in docs]
        reranked = reranker.rerank(expanded, texts, top_k=5)
        kept_set = {t for t, _ in reranked}
        reranked_docs = [d for d in docs if d.page_content in kept_set]
        t3 = time.perf_counter()
        stage_times["rerank"].append(t3 - t2)

        # Stage 3: Compression (no-op if not configured)
        t4 = time.perf_counter()
        if reranked_docs:
            from app.retrieval.compression import compress_documents
            reranked_docs = compress_documents(q, reranked_docs)
        t5 = time.perf_counter()
        stage_times["compress"].append(t5 - t4)

        # Stage 4: Generation
        gen = generate_answer(q, reranked_docs)
        t6 = time.perf_counter()
        stage_times["generation"].append(t6 - t5)

        stage_times["total_pipeline"].append(t6 - t_start)

        print(f"    Search:{t1-t0:.2f}s Rerank:{t3-t2:.2f}s Compress:{t5-t4:.2f}s Generate:{t6-t5:.2f}s Total:{t6-t_start:.2f}s")

    print(f"\n  ── LATENCY RESULTS ({len(standard)} queries) ──")
    print(f"  {'Stage':<20} {'P50':>8} {'P95':>8} {'P99':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8}")
    for stage, times in stage_times.items():
        if times:
            p50 = np.median(times)
            p95 = np.percentile(times, 95)
            p99 = np.percentile(times, 99)
            print(f"  {stage:<20} {p50:>7.2f}s {p95:>7.2f}s {p99:>7.2f}s")

    return {
        stage: {
            "p50": round(np.median(times), 3) if times else 0,
            "p95": round(np.percentile(times, 95), 3) if times else 0,
            "p99": round(np.percentile(times, 99), 3) if times else 0,
            "mean": round(np.mean(times), 3) if times else 0,
        }
        for stage, times in stage_times.items()
    }


# ══════════════════════════════════════════
# PHASE 5: Robustness & Safety
# ══════════════════════════════════════════

def phase_5_robustness():
    print("\n" + "=" * 70)
    print("PHASE 5: ROBUSTNESS & SAFETY")
    print("=" * 70)

    results = {}

    # ── OOD Handling ──
    print(f"\n  ── OOD / Out-of-Scope Handling ──")
    ood_results = []
    for item in GROUND_TRUTHS["ood"]:
        q = item["query"]
        print(f"    [{item['id']}] Q: {q[:60]}...")
        docs = retrieve_rerank(q)
        if not docs:
            print(f"      → No docs retrieved ✓")
            ood_results.append(1.0)
            continue
        ans = generate_answer(q, docs)["answer"]
        rate_limit(4.0)
        ans_lower = ans.lower()

        # Score: 1.0 if refuses gracefully, 0.0 if fabricates
        refusal_patterns = [
            "don't know", "not in the", "no information", "cannot answer",
            "out of scope", "not found", "not relevant", "not covered",
            "no relevant", "unable to", "beyond the scope", "the corpus",
            "i'm sorry", "only answer questions about calculus",
        ]
        refuses = any(p in ans_lower for p in refusal_patterns)
        score = 1.0 if refuses else 0.0
        ood_results.append(score)
        print(f"      → {'✅ Refuses gracefully' if refuses else '❌ Fabricates answer'}")
        print(f"      → {ans[:120]}...")

    avg_ood = np.mean(ood_results) if ood_results else 0
    print(f"    OOD Score: {avg_ood:.2f} ({sum(ood_results)}/{len(ood_results)} correctly refused)")

    # ── Adversarial Robustness ──
    print(f"\n  ── Adversarial Robustness (typos) ──")
    adv_results = []
    truth_map = {s["id"]: s for s in GROUND_TRUTHS["standard"]}
    for item in GROUND_TRUTHS["adversarial"]:
        q = item["query"]
        clean_id = item.get("truth_clean_id", "")
        clean_item = truth_map.get(clean_id, {})
        clean_q = clean_item.get("query", "")
        kw = item.get("keywords", [])

        # Typo query
        docs = retrieve_rerank(q)
        ans_typo = generate_answer(q, docs)["answer"] if docs else "ERROR"
        rate_limit(4.0)
        kw_typo = keyword_coverage(ans_typo, kw)

        # Clean query (if available)
        if clean_q:
            clean_docs = retrieve_rerank(clean_q)
            ans_clean = generate_answer(clean_q, clean_docs)["answer"] if clean_docs else "ERROR"
            rate_limit(4.0)
            kw_clean = keyword_coverage(ans_clean, kw)
        else:
            kw_clean = kw_typo

        robustness = kw_typo / max(kw_clean, 0.01)
        adv_results.append(min(robustness, 1.0))
        print(f"    [{item['id']}] Typo: '{q[:50]}...'")
        print(f"      KW typo: {kw_typo:.2f} vs clean: {kw_clean:.2f} → Robustness: {robustness:.2f}")

    avg_adv = np.mean(adv_results) if adv_results else 0
    print(f"    Adversarial Robustness: {avg_adv:.2f}")

    # ── Multi-hop Reasoning ──
    print(f"\n  ── Multi-hop Reasoning ──")
    multi_results = []
    for item in GROUND_TRUTHS["multi_hop"]:
        q = item["query"]
        kw = item.get("keywords", [])
        docs = retrieve_rerank(q)
        if not docs:
            print(f"    [{item['id']}] ❌ No docs retrieved")
            multi_results.append(0.0)
            continue
        ans = generate_answer(q, docs)["answer"]
        rate_limit(4.0)

        kw_score = keyword_coverage(ans, kw)
        # LLM judge: does answer show multi-step reasoning?
        judge_prompt = f"""Does this answer demonstrate correct multi-step mathematical reasoning connecting multiple concepts?
Score 0.0 (no reasoning, single step) to 1.0 (complete multi-step reasoning with correct intermediate results).

Question: {q}
Answer: {ans[:600]}
Score (0.0-1.0):"""
        reasoning_resp = judge_llm(judge_prompt)
        reasoning_score = extract_score(reasoning_resp)
        combined = (kw_score + reasoning_score) / 2
        multi_results.append(combined)
        print(f"    [{item['id']}] KW={kw_score:.2f} Reasoning={reasoning_score:.2f} Combined={combined:.2f}")

    avg_multi = np.mean(multi_results) if multi_results else 0

    # ── Edge Cases ──
    print(f"\n  ── Edge Cases ──")
    edge_results = {}
    for item in GROUND_TRUTHS["edge"]:
        q = item["query"]
        kw = item.get("keywords", [])
        print(f"    [{item['id']}] Query type: {'EMPTY' if not q else 'VERY_LONG' if len(q) > 200 else 'NARROW'}")
        docs = retrieve_rerank(q)
        if not docs:
            print(f"      → No docs retrieved (expected for edge case)")
            edge_results[item["id"]] = {"handled": True, "note": "No retrieval triggered"}
            continue
        ans = generate_answer(q, docs)["answer"]
        rate_limit(4.0)
        kw_score = keyword_coverage(ans, kw) if kw else 1.0
        edge_results[item["id"]] = {
            "handled": not ans.startswith("ERROR"),
            "kw_score": kw_score,
            "preview": ans[:100],
        }
        print(f"      → KW={kw_score:.2f}: {ans[:100]}...")

    # ── Confidence Calibration ──
    print(f"\n  ── Confidence Calibration ──")
    cal_scores = []
    for item in GROUND_TRUTHS["ood"]:
        q = item["query"]
        docs = retrieve_rerank(q)
        if not docs:
            cal_scores.append(1.0)
            continue
        ans = generate_answer(q, docs)["answer"]
        rate_limit(4.0)
        # If it says "I don't know", check it actually is uncertain
        if any(p in ans.lower() for p in ["don't know", "not in", "cannot", "unable"]):
            cal_scores.append(1.0)
        else:
            cal_scores.append(0.0)

    avg_cal = np.mean(cal_scores) if cal_scores else 0
    print(f"    Calibration (correct refusal on OOD): {avg_cal:.2f}")

    print(f"\n  ── ROBUSTNESS RESULTS ──")
    print(f"  OOD Handling:             {avg_ood:.3f}")
    print(f"  Adversarial Robustness:   {avg_adv:.3f}")
    print(f"  Multi-hop Reasoning:      {avg_multi:.3f}")
    print(f"  Confidence Calibration:   {avg_cal:.3f}")

    return {
        "ood_handling": round(avg_ood, 4),
        "adversarial_robustness": round(avg_adv, 4),
        "multi_hop_reasoning": round(avg_multi, 4),
        "confidence_calibration": round(avg_cal, 4),
        "edge_cases": edge_results,
    }


# ══════════════════════════════════════════
# PHASE 6: Cost & Efficiency
# ══════════════════════════════════════════

def phase_6_cost_efficiency():
    print("\n" + "=" * 70)
    print("PHASE 6: COST & EFFICIENCY")
    print("=" * 70)

    standard = GROUND_TRUTHS["standard"]

    # Token counting (approximate: 1 token ≈ 4 chars for English)
    total_input_tokens = 0
    total_output_tokens = 0
    total_rerank_pairs = 0
    total_embed_chars = 0

    for item in standard:
        q = item["query"]
        docs = retrieve_rerank(q)
        if not docs:
            continue

        # Rerank pairs: 30 docs × 512 max tokens per pair
        total_rerank_pairs += min(30 * 512, sum(len(d.page_content) // 4 for d in docs))

        # Embed chars
        total_embed_chars += sum(len(d.page_content) for d in docs)

        # Generation tokens
        gen = generate_answer(q, docs)
        rate_limit(4.0)
        total_output_tokens += len(gen["answer"]) // 4

        # Input context tokens (system prompt + formatted docs)
        context = generator.format_context(docs)
        total_input_tokens += (len(context) + 1500) // 4  # +1500 chars for system prompt

    # Approximate costs (using 9-router / kr/claude-sonnet-4.5 pricing)
    # Note: these are approximate; real 9-router pricing may differ
    INPUT_COST_PER_1K = 0.003  # $ per 1K input tokens (approximate)
    OUTPUT_COST_PER_1K = 0.015  # $ per 1K output tokens (approximate)
    RERANK_COST_PER_PAIR = 0.00001  # $ per rerank pair (local, essentially free)
    EMBED_COST_PER_1K_CHARS = 0.0001  # $ per 1K chars embedded (local model)

    input_cost = total_input_tokens / 1000 * INPUT_COST_PER_1K
    output_cost = total_output_tokens / 1000 * OUTPUT_COST_PER_1K
    rerank_cost = total_rerank_pairs * RERANK_COST_PER_PAIR / 1000
    embed_cost = total_embed_chars / 1000 * EMBED_COST_PER_1K_CHARS

    total_cost = input_cost + output_cost + rerank_cost + embed_cost
    cost_per_query = total_cost / len(standard)

    # Token efficiency
    context_chars = sum(total_input_tokens * 4 for _ in [0])  # recalc
    total_context_chars = 0
    for item in standard:
        q = item["query"]
        docs = retrieve_rerank(q)
        if docs:
            total_context_chars += sum(len(d.page_content) for d in docs)
    avg_token_eff = total_output_tokens * 4 / max(total_context_chars, 1) * 100

    print(f"\n  ── COST ANALYSIS ({len(standard)} queries) ──")
    print(f"  Total input tokens:    {total_input_tokens:,}")
    print(f"  Total output tokens:   {total_output_tokens:,}")
    print(f"  Total rerank pairs:    {total_rerank_pairs:,}")
    print(f"  Total embed chars:     {total_embed_chars:,}")
    print(f"")
    print(f"  Input cost:            ${input_cost:.5f}")
    print(f"  Output cost:           ${output_cost:.5f}")
    print(f"  Rerank cost:           ${rerank_cost:.5f} (local ≈ free)")
    print(f"  Embed cost:            ${embed_cost:.5f} (local ≈ free)")
    print(f"  ──────────────────────────────────")
    print(f"  Total cost:            ${total_cost:.5f}")
    print(f"  Cost per query:        ${cost_per_query:.6f}")
    print(f"  Cost per 1K queries:   ${cost_per_query * 1000:.3f}")

    print(f"\n  ── TOKEN EFFICIENCY ──")
    print(f"  Avg output chars:       {total_output_tokens * 4 // len(standard):,}")
    print(f"  Avg context chars:      {total_context_chars // len(standard):,}")
    print(f"  Token efficiency:       {avg_token_eff:.1f}%")

    # Reranker cost-quality tradeoff
    print(f"\n  ── RERANKER COST-QUALITY TRADEOFF ──")
    tradeoff_results = []
    for item in standard[:5]:  # subset
        q = item["query"]
        kw = item["keywords"]

        # Without reranker
        no_rerank_docs = dense_only_store.similarity_search(q, k=5)
        no_rr_kw = keyword_coverage(
            " ".join(d.page_content for d in no_rerank_docs), kw
        )

        # With reranker
        with_rerank_docs = retrieve_rerank(q, top_k_rerank=5)
        with_rerank_docs = with_rerank_docs if with_rerank_docs else []
        w_rr_kw = keyword_coverage(
            " ".join(d.page_content for d in with_rerank_docs), kw
        )

        gain = w_rr_kw - no_rr_kw
        tradeoff_results.append({"id": item["id"], "without": no_rr_kw, "with": w_rr_kw, "gain": gain})
        print(f"    [{item['id']}] Without={no_rr_kw:.2f} With={w_rr_kw:.2f} Gain={gain:+.2f}")

    avg_gain = np.mean([r["gain"] for r in tradeoff_results]) if tradeoff_results else 0
    avg_without = np.mean([r["without"] for r in tradeoff_results]) if tradeoff_results else 0
    avg_with = np.mean([r["with"] for r in tradeoff_results]) if tradeoff_results else 0
    print(f"    Avg without reranker: {avg_without:.3f}")
    print(f"    Avg with reranker:    {avg_with:.3f}")
    print(f"    Avg gain:             {avg_gain:+.3f}")

    return {
        "cost_per_query_usd": round(cost_per_query, 6),
        "cost_per_1k_queries_usd": round(cost_per_query * 1000, 4),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "token_efficiency_pct": round(avg_token_eff, 1),
        "reranker_quality_gain": round(avg_gain, 4),
        "reranker_kw_without": round(avg_without, 4),
        "reranker_kw_with": round(avg_with, 4),
    }


# ══════════════════════════════════════════
# MAIN: Run all phases sequentially
# ══════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  RAG BOOK — COMPREHENSIVE EVALUATION")
    print(f"  Model: {settings.llm_model}")
    print(f"  Qdrant: /tmp/qdrant_calculus_db")
    print(f"  BM25: /tmp/qdrant_calculus_bm25.pkl")
    print("=" * 70)

    all_results = {}

    # Phase 1
    p1 = phase_1_retrieval_quality()
    all_results["phase_1_retrieval"] = p1
    print(f"\n  ✅ Phase 1 complete. Type 'next' to continue to Phase 2, or 'skip' to skip.")

    # Phase 2
    p2 = phase_2_generation_quality(p1)
    all_results["phase_2_generation"] = p2
    print(f"\n  ✅ Phase 2 complete.")

    # Phase 3
    p3 = phase_3_end_to_end(p2)
    all_results["phase_3_e2e"] = p3
    print(f"\n  ✅ Phase 3 complete.")

    # Phase 4
    p4 = phase_4_latency()
    all_results["phase_4_latency"] = p4
    print(f"\n  ✅ Phase 4 complete.")

    # Phase 5
    p5 = phase_5_robustness()
    all_results["phase_5_robustness"] = p5
    print(f"\n  ✅ Phase 5 complete.")

    # Phase 6
    p6 = phase_6_cost_efficiency()
    all_results["phase_6_cost"] = p6
    print(f"\n  ✅ Phase 6 complete.")

    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION SUMMARY")
    print("=" * 70)
    for phase, data in all_results.items():
        print(f"\n  ── {phase} ──")
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)
