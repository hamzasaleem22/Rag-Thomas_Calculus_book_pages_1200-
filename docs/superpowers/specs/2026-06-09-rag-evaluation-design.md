# RAG Evaluation Design — Thomas Calculus RAG System

## Goal
Comprehensively evaluate the RAG pipeline across 6 dimensions: Retrieval Quality, Generation Quality, End-to-End Metrics, Latency & Throughput, Robustness & Safety, and Cost & Efficiency.

## Constraints
- No new model downloads. All evaluation uses existing resources: 9-router at `localhost:20128/v1` (`kr/claude-sonnet-4.5` as judge LLM), local ONNX/transformers reranker, local BM25 + Qdrant.
- Open-source tools only. RAGAS for compatible metrics, custom Python for the rest.
- Rate limit: 3-5s delay between queries + existing `settings.request_delay` (10s).
- Standalone script (no live server needed) — imports modules directly for per-stage profiling.

## Ground Truth Dataset

25 queries across 6 categories:

| Category | Count | Examples |
|----------|-------|---------|
| Standard (ch 2-16) | 12 | Derivative of sin(x), FTC Part 1, Chain rule, Integration by parts, Ratio test, Divergence theorem, Riemann sum, L'Hôpital, Second derivative test, Separable DE, Arc length, Taylor series |
| Multi-hop | 3 | "Use chain rule to find derivative of sin(x²) and verify", "Volume of revolution using disks vs washers", "Find limit using L'Hôpital then verify with series expansion" |
| OOD / out-of-scope | 3 | "What is the capital of France?", "Explain quantum computing", "Write a poem about calculus" |
| Adversarial typos | 3 | "Wht is L'Hopital's rul?", "Derivativ of sin(x)?", "Fnd the aera of circl using intgration" |
| Edge cases | 3 | Empty query "", 300-word extremely specific query, ultra-narrow topic "What is the epsilon-delta proof of continuity for f(x)=x² at x=3?" |
| Consistency pair | 2 | Two queries with identical meaning: "d/dx sin(x)?" vs "What is the derivative of sine?" |

## Scoring Architecture

Three methods for computing metrics:

### 1. RAGAS (standard queries only)
- Metrics: Faithfulness, Context Precision, Context Recall
- Uses `datasets.Dataset` + `ragas.evaluate()` with `kr/claude-sonnet-4.5` as eval LLM via OpenAI-compatible 9-router
- Runs on the 12 standard queries only (RAGAS requires ground truth)

### 2. Custom Python computation (algorithmic, no LLM)
- Metrics: NDCG@K, MRR, Hit Rate@K, Answer F1, Exact Match, Latency quantiles
- Pure `numpy`/`scipy` computations on retrieval scores and token overlap

### 3. Custom LLM-as-judge (all query types)
- Metrics: Answer Relevancy, Answer Correctness, Hallucination Rate, LaTeX Fidelity, CSAT proxy, OOD handling score
- Prompts `kr/claude-sonnet-4.5` via 9-router with structured scoring requests
- Each returns a score 0.0-1.0 with brief justification

## Metric Specifications

### Phase 1: Retrieval Quality
- **Context Precision**: RAGAS metric. Proportion of retrieved chunks relevant to query.
- **Context Recall**: RAGAS metric. Proportion of ground truth covered by retrieved chunks.
- **NDCG@K**: Custom. Uses reranker scores as graded relevance. DCG = Σ (2^rel_i - 1) / log2(i+1). IDCG from ideal ranking.
- **MRR**: Custom. 1/rank of first chunk containing ground-truth keywords. Averaged.
- **Hit Rate@K**: Custom. Binary: does any relevant chunk appear in top-5 or top-30?
- **BM25 vs Dense vs Hybrid comparison**: Three retrieval modes compared on the same 12 standard queries.

### Phase 2: Generation Quality
- **Faithfulness**: RAGAS metric (on standard queries). Every claim in answer must be grounded in context.
- **Answer Relevancy**: LLM-as-judge. Prompt: "On a scale 0.0-1.0, how directly does this answer address the question? 1.0 = perfectly relevant."
- **Answer Correctness**: LLM-as-judge with ground truth reference. Score 0.0-1.0 for factual accuracy.
- **Hallucination Rate**: LLM-as-judge. "List any claims NOT supported by the provided context. Count ungrounded_claims / total_claims."
- **LaTeX Fidelity**: Regex-based. Checks: `$...$` or `$$...$$` wrapping, `\frac{}{}` with both braces, `\sum_{}^{}`, `\lim_{}`, correct `\sin`, `\cos`. Score = passed_checks / total_checks.

### Phase 3: End-to-End Metrics
- **RAGAS Score**: Composite of faithfulness + context_precision + context_recall averaged.
- **Answer F1**: Token-level precision/recall between generated answer and ground truth. Uses `sklearn.metrics.f1_score` on tokenized sets.
- **Exact Match (EM)**: Strict binary. 1 if generated answer matches ground truth exactly (after normalization — lowercase, strip whitespace, normalize unicode).
- **Query Consistency**: Same question phrased two ways must yield answers that score >= 0.8 cosine similarity (using `all-MiniLM-L6-v2` embeddings).
- **CSAT Proxy**: LLM-as-judge. "Rate this answer for a calculus student: 0.0 (useless) to 1.0 (perfectly helpful)."

### Phase 4: Latency & Throughput
- Per-query instrumentation using `time.perf_counter()` around each stage:
  - sparse_embed_time: BM25 embed_query + Qdrant sparse search
  - dense_embed_time: HF model embed + Qdrant dense search
  - retrieval_fuse_time: hybrid fusion in Qdrant
  - rerank_time: Cross-encoder reranker forward pass
  - compress_time: LLM compression calls
  - gen_ttft_time: time to first token (simulated non-streaming)
  - gen_total_time: total generation
- Aggregated: P50, P95, P99 for each stage across all queries.

### Phase 5: Robustness & Safety
- **OOD handling**: For 3 out-of-scope queries, score = 1.0 if system answers "I don't know" or "Not in the corpus", 0.0 if it fabricates.
- **Adversarial robustness**: For 3 typo-ridden queries, compare answer quality vs the clean version of same query. Score = quality(typo) / quality(clean).
- **Multi-hop score**: For 3 multi-hop queries, LLM-as-judge checks intermediate reasoning steps. Score = proportion of correct intermediate steps.
- **Edge case score**: For 3 edge case queries (empty, very long, very specific), qualitative assessment of handling.
- **Confidence calibration**: When the answer contains "I don't know" or "not specified", verify it's actually uncertain (not falsely modest).

### Phase 6: Cost & Efficiency
- **Cost per query**: Token counts (input + output) from each stage:
  - Embedding: 384-dim dense, ~10K tokens/chunk, 30 chunks = 300K tokens embedded per query
  - Re-rank: 30 pairs × 512 tokens = ~15K tokens
  - LLM generation: measured from response usage
  - $/1k queries projection using approximate prices
- **Token efficiency**: (answer_tokens / context_tokens) — what fraction of sent context appears in answer.
- **Re-ranker cost-quality tradeoff**: Compare metrics with reranker ON vs OFF on the same 12 standard queries.

## Phases Execution Order

1. Phase 1: Retrieval Quality (standard queries only, fastest)
2. Phase 2: Generation Quality (all query types, needs retrieval output)
3. Phase 3: End-to-End Metrics (standard + consistency queries)
4. Phase 4: Latency & Throughput (same run as Phase 1, instrumented)
5. Phase 5: Robustness & Safety (OOD + adversarial + edge queries)
6. Phase 6: Cost & Efficiency (computed from Phase 1-5 token counts)

Each phase outputs results to console. Phase starts only after the previous phase results are reported and acknowledged.

## Implementation Plan

The implementation plan file at `plan/evaluate-rag-pipeline-1.md` will contain:
- Single script: `backend/scripts/eval_full.py`
- Each phase as a function that can run independently
- Shared ground truth dataset and judge client
- Shared rate limiter
- Report collation at end

## Files Affected
- `backend/scripts/eval_full.py` — new file, the evaluation orchestrator
- `AGENTS.md` — updated with evaluation results after each phase

## Risks & Assumptions
- **RISK-001**: 9-router may rate-limit. Mitigated by 3-5s delays + existing 10s request_delay.
- **RISK-002**: RAGAS answer_relevancy may fail with non-OpenAI API (known issue). Fallback to custom LLM-as-judge.
- **ASSUMPTION-001**: Qdrant at `/tmp/qdrant_calculus_db` is populated and BM25 is fitted.
- **ASSUMPTION-002**: `kr/claude-sonnet-4.5` model is available on the 9-router.
