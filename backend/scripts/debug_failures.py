#!/usr/bin/env python3
"""Re-check failing queries with different models."""
import sys, time, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from app.config import settings
from app.retrieval.vector_store import HybridVectorStore
from app.retrieval.reranker import Reranker
from app.generation.generator import Generator
from app.retrieval.expansion import expand_query_text
from qdrant_client import QdrantClient

client = QdrantClient(path="/tmp/qdrant_calculus_db")
store = HybridVectorStore(client=client, bm25_path="/tmp/qdrant_calculus_bm25.pkl")
reranker = Reranker()
settings.request_delay = 2.0

FAILING = [
    ("ch04", "State L'Hôpital's rule and find limit of sin(x)/x as x→0.",
     "If f(a)=g(a)=0 or ±∞, lim f(x)/g(x) = lim f'(x)/g'(x). For sin(x)/x: lim cos(x)/1 = 1."),
    ("ch14", "What is the second derivative test for local extrema of functions of two variables?",
     "D = f_xx f_yy - (f_xy)². D>0, f_xx>0: local min. D>0, f_xx<0: local max. D<0: saddle. D=0: inconclusive."),
    ("ch16", "State the divergence theorem (Gauss' theorem) and explain its physical meaning.",
     "∬_S F·n dσ = ∭_D ∇·F dV. Flux through closed surface = triple integral of divergence over enclosed volume."),
]

def retrieve_rerank(query):
    expanded = expand_query_text(query)
    docs = store.similarity_search(expanded, k=30)
    texts = [d.page_content for d in docs]
    reranked = reranker.rerank(expanded, texts, top_k=5)
    kept = {t for t, _ in reranked}
    return [d for d in docs if d.page_content in kept]

judge = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
JUDGE_MODEL = "kr/claude-haiku-4.5"

def judge_llm(prompt):
    time.sleep(2)
    resp = judge.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "system", "content": "Score 0.0-1.0. Return only the numeric score and one sentence."},
                  {"role": "user", "content": prompt}],
        temperature=0, max_tokens=100)
    return resp.choices[0].message.content.strip()

for model_name, label in [("kr/deepseek-3.2", "DeepSeek-3.2"), ("kr/claude-haiku-4.5", "Claude Haiku-4.5")]:
    print()
    print("=" * 70)
    print(f"  MODEL: {label}")
    print("=" * 70)

    gen = Generator()
    gen.primary_model = model_name
    gen.fallback_model = model_name
    gen.legacy_model = model_name
    gen.last_request_time = 0.0

    for qid, query, truth in FAILING:
        print(f"\n  --- [{qid}] {query[:60]}... ---")
        docs = retrieve_rerank(query)
        if not docs:
            print("  No docs retrieved")
            continue

        result = gen.generate(query, docs)
        answer = result["answer"]
        citations = result.get("citations", [])

        # Judge: correctness
        corr_prompt = (
            f"Rate correctness (0-1) of the answer vs ground truth.\n"
            f"Question: {query}\n"
            f"Answer: {answer[:400]}\n"
            f"Truth: {truth}\n"
            f"Score:"
        )
        corr = judge_llm(corr_prompt)

        # Judge: hallucination
        ctx = "\n".join([d.page_content[:200] for d in docs[:2]])
        hal_prompt = (
            f"Does the answer hallucinate? Score 0=fully grounded, 1=fully hallucinated.\n"
            f"Context:\n{ctx}\n"
            f"Answer:\n{answer[:400]}\n"
            f"Score:"
        )
        hal = judge_llm(hal_prompt)

        # Extract scores
        c_score = float(re.findall(r"(\d+\.?\d*)", corr)[0]) if re.findall(r"(\d+\.?\d*)", corr) else 0
        h_score = float(re.findall(r"(\d+\.?\d*)", hal)[0]) if re.findall(r"(\d+\.?\d*)", hal) else 0

        latex_ok = bool(re.search(r'\$[^$]+\$|\$\$', answer))

        print(f"  Correctness: {corr[:100]}")
        print(f"  Hallucination: {hal[:100]}")
        print(f"  LaTeX: {'✅' if latex_ok else '❌'} | Cites: {len(citations)}")
        print(f"  Answer preview: {answer[:150]}...")
