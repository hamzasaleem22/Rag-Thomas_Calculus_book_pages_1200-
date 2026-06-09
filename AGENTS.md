# Rag_Book - Agent Memory

## Project
RAG pipeline for "Thomas' Calculus Early Transcendentals, 14th Edition" (1262-page PDF). Hybrid search (dense + BM25 sparse) with reranking, LLM generation with citations, FastAPI backend, React frontend.

## Key Files

| File | Purpose |
|------|---------|
| `backend/app/config.py` | Settings (env: HF_TOKEN_1, HF_TOKEN_2, OPENAI_API_KEY) |
| `backend/app/retrieval/embeddings.py` | `HFInferenceAPIEmbeddings` — local sentence-transformers (all-MiniLM-L6-v2) with model caching |
| `backend/app/retrieval/vector_store.py` | `BM25SparseEmbeddings` + `HybridVectorStore` (Qdrant wrapper). BM25 fits on doc ingest, persists to pickle |
| `backend/app/retrieval/reranker.py` | `Reranker` with `cross-encoder/ms-marco-MiniLM-L-6-v2` (local transformers). Supports MMR diversity |
| `backend/app/retrieval/expansion.py` | Query expansion with math synonym map (L'Hôpital → l'hopital, etc.) |
| `backend/app/retrieval/compression.py` | LLM-based contextual compression — extracts relevant sentences from chunks (latency-heavy) |
| `backend/app/retrieval/hyde.py` | HyDE: generates hypothetical textbook passage to improve retrieval (disabled by default) |
| `backend/app/retrieval/latex_sanitizer.py` | Post-processes LLM answer — fixes bare `\frac{}{}` missing braces, wraps bare LaTeX in `$$`, removes Unicode dupes |
| `backend/app/retrieval/math_cleaner.py` | Pre-retrieval: cleans PDF garbled math (`/uniXXXX` glyphs, spaced func names, `q`→`∞`, `p`→`π`) |
| `backend/app/generation/generator.py` | `Generator` — model cascade (primary→fallback→legacy), retry logic, streaming, self-consistency, citation verification, answer validation with regen |
| `backend/app/ingestion/chunker.py` | Markdown header splitter + RecursiveCharacterTextSplitter (token-estimated). Injects `## Section` headers, enforces theorem boundaries, merges broken LaTeX |
| `backend/app/ingestion/cleaner.py` | Unicode artifact fix (`/uni2032`→`'`), LaTeX normalization (`\[`→`$$`), footer/garbled line removal |
| `backend/app/ingestion/indexer.py` | Batch embed (dense + sparse) → `PointStruct` upsert to Qdrant. Returns BM25 for persistence |
| `backend/app/ingestion/pdf_loader.py` | `MarkerDocumentLoader` — `PdfReader` extraction with Unicode math → LaTeX conversion, page-split documents |
| `backend/app/api/endpoints.py` | FastAPI: `/query`, `/query/stream` (SSE), `/ingest`, `/health`, `/metrics` |
| `backend/app/api/schemas.py` | Pydantic models: `QueryRequest`, `QueryResponse`, `Citation`, `IngestResponse` |
| `backend/app/_compat.py` | RAGAS VertexAI compatibility shim (likely dead code) |
| `backend/run_api.py` | Uvicorn entry point, reload enabled |
| `backend/run_index.py` | Quick index runner (but does NOT persist BM25 — use `scripts/index_full.py` instead) |
| `backend/scripts/ingest.py` | CLI ingestion pipeline (PDF→markdown→clean→chunk→index) |
| `backend/scripts/index_full.py` | Full-book index with BM25 persistence to `/tmp/qdrant_calculus_bm25.pkl` |
| `backend/scripts/evaluate.py` | RAGAS evaluation (old — uses pickled client, not persistent Qdrant) |
| `backend/scripts/evaluate_deep.py` | Deep eval: 8 ground-truth queries with term-matching grader |
| `backend/scripts/final_eval.py` | Final eval with rate-limited calls, query expansion, keyword grading |
| `backend/scripts/diagnose_pipeline.py` | 20-query diagnostic: BM25 health, retrieval accuracy, chunk quality, citations |
| `backend/scripts/eval_full.py` | Comprehensive 6-phase evaluation script (retrieval, generation, E2E, latency, robustness, cost) |
| `backend/scripts/debug_failures.py` | Model comparison debug script for failing queries |
| `backend/tests/test_chunking.py` | 16 manual unit tests (not pytest) for chunking, cleaning, BM25, query expansion |
| `frontend/` | Vite + React + Tailwind chat UI with citations |
| `frontend/vite.config.ts` | Vite config — proxies `/api` → backend with path rewrite (strips `/api` prefix) |

## Data
- Chunked text: `backend/data/markdown/parsed_docs.jsonl` (1316 chunks)
- Qdrant persistent: `/tmp/qdrant_calculus_db` (1316 vectors indexed)
- BM25 state: `/tmp/qdrant_calculus_bm25.pkl` (14254 vocab, 1316 docs, fitted)
- Evaluation ground truth: `backend/scripts/eval_full.py` (10 standard + 2 multi-hop + 1 OOD + 2 adversarial + 2 edge + 2 consistency queries)

## API & Models
- **LLM primary**: `kr/claude-sonnet-4.5` via 9-router at `http://localhost:20128/v1`
- **LLM fallback**: `kr/deepseek-3.2`
- **LLM legacy**: `cx/gpt-5.5`
- **Eval model (Generation + Judge)**: `kr/claude-haiku-4.5` (used in evaluation script)
- **Embedding**: `all-MiniLM-L6-v2` (384-dim) — local sentence-transformers
- **Sparse**: Custom BM25 (local, unoptimized k1=1.5, b=0.75)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` — local transformers
- **HF tokens**: Both in `backend/.env` (`HF_TOKEN_1`, `HF_TOKEN_2`), fallback logic in `settings.get_hf_token()`
- **Chunk config**: chunk_size=1024 (tokens, estimated as len/4), overlap=128

## RAGAS Results (latest)
- Faithfulness: 0.8889 (previous run)
- Context Precision: 0.7289
- Context Recall: 1.0000
- Answer Relevancy: N/A (9-router doesn't support openai provider for RAGAS sub-jobs)
- Note: RAGAS returned 0.0 for all metrics with kr/claude-haiku-4.5 (known incompatibility). Custom LLM-as-judge used instead.

## Evaluation Results (2026-06-09) — All Phases Complete
Using `kr/claude-haiku-4.5` for generation + judge via 9-router. 10 standard queries (ch2-16) + 2 multi-hop + diversity set.

### Phase 1: Retrieval Quality
| Metric | Score |
|--------|-------|
| NDCG@5 | 1.000 |
| MRR | 1.000 |
| Hit Rate@5 | 100% (10/10) |
| Hit Rate@30 | 100% (10/10) |
| Hybrid/Dense/BM25@10 | 100% all modes |
| Avg retrieval latency | 1.10s (P95: 4.96s) |

### Phase 2: Generation Quality
| Metric | Score |
|--------|-------|
| Answer Correctness | 0.935 |
| Answer Relevancy | 0.829 |
| Keyword Coverage | 0.756 |
| Hallucination Rate | 0.367 (lower=better) |
| LaTeX Fidelity | 0.764 |
| Citations in answers | 100% |

**Failing queries improved with Haiku vs DeepSeek:**
- ch04 L'Hôpital: Cor 0.00→0.95, Hal 1.00→0.20 ✅
- ch14 2nd deriv test: Cor 0.00→0.95, Hal 0.50→0.20 ✅
- ch16 Divergence thm: Cor 0.90→0.95, Hal 0.80→0.80 (still hallucinating)

**Remaining issues:** ch16 divergence theorem hallucinates heavily (~0.80). Multi-hop queries also show elevated hallucination (~0.60-0.80). LaTeX formatting needs improvement (0.76).

### Phase 3: End-to-End Metrics
| Metric | Score | Notes |
|--------|:-----:|-------|
| Answer F1 | 0.314 | Low (generated answers longer than short GTs) |
| Exact Match | 0.000 | Expected (structured output ≠ GT format) |
| Query Consistency | 0.825 | ✅ Same Q, different phrasing → 82.5% semantic similarity |
| CSAT Proxy | 0.822 | ✅ LLM-judged helpfulness for students |

### Phase 4: Latency & Throughput
| Stage | P50 | P95 | P99 |
|-------|:---:|:---:|:---:|
| Dense embed + search | 0.36s | 3.91s | 6.21s |
| Reranker | 4.22s | 6.20s | 7.29s |
| Compression (5 LLM calls) | 12.42s | 15.00s | 15.83s |
| Generation | 10.71s | 13.15s | 13.40s |
| **Total pipeline** | **29.62s** | **34.07s** | **34.56s** |

**Hidden bottlenecks:** Compression (12.4s) and reranker (4.2s) dominate — not the LLM generation itself.

### Phase 5: Robustness & Safety
| Metric | Score | Assessment |
|--------|:-----:|------------|
| OOD Handling | 0.000 | ❌ System fabricates for out-of-scope queries |
| Adversarial Robustness | 1.000 | ✅ Typos have zero effect |
| Multi-hop Reasoning | 0.779 | ⚠️ Decent but moderate hallucination |
| Edge Cases | Mixed | Empty query → hallucinated; narrow query handled well |
| Confidence Calibration | 0.000 | ❌ Never says "I don't know" |

**Critical:** System prompt lacks OOD/refusal instructions. Must add for production.

### Phase 6: Cost & Efficiency
| Metric | Value |
|--------|:-----:|
| Cost per query | $0.012 |
| Cost per 1K queries | $12.21 |
| Token efficiency | 9.0% (output is 9% of context length) |
| Reranker quality gain | -0.05 (adds latency without keyword coverage improvement over dense-only top-5) |

### Key Recommendations
1. **Add OOD/refusal instructions** to system prompt — critical for production safety
2. **Compression is the #1 latency bottleneck** — cache or skip when context is small
3. **Reranker cost-quality questionable** — 4.2s added but no keyword coverage gain on this domain
4. **LaTeX formatting needs improvement** (0.76) — consider post-processing or better exemplars
5. **ch16 divergence theorem and multi-hop queries** need focused improvement

## Running
- Backend: `source .venv/bin/activate && cd backend && python3 -m uvicorn app.api.endpoints:app --host 0.0.0.0 --port 8000`
- Frontend: `cd frontend && npm run dev` (proxies `/api` → backend on port 8000)
- Tests: `cd backend && python tests/test_chunking.py`

## Directory structure
- All backend code under `backend/` (app/, scripts/, data/, tests/, run_api.py, run_index.py, .env)
- Frontend under `frontend/`
- Shared venv at root `.venv/`

## Critical Fixes
- **Vite proxy**: Frontend sends to `/api/query`, backend serves `/query`. The proxy in `vite.config.ts` uses `path.replace(/^\/api/, "")` to strip the prefix.
- **Server CWD**: Must run `uvicorn` from `backend/` directory, not root. Otherwise `from app.api.endpoints import app` fails because `app/` lives inside `backend/`.

## Status (all complete)
- Phase 1: PDF extraction ✅
- Phase 2: Chunking ✅
- Phase 3: Qdrant index (hybrid dense + BM25) ✅
- Phase 4: Reranking ✅
- Phase 5: Generation with citations ✅
- Phase 6: RAGAS evaluation ✅
- Phase 7: FastAPI backend ✅
- Phase 8: React frontend ✅
- Phase 9: Streaming + multi-turn ✅

## Known Issues (All 17 Fixed — 2026-06-09)

All issues from the code review have been resolved:

### 🔴 Critical (Fixed)
1. **Hardcoded API key removed** — `scripts/evaluate.py` now uses `os.environ` defaults and `settings.openai_api_key`.
2. **`/metrics` chunk_count is dynamic** — queries Qdrant `collection.count()` instead of hardcoded 1316.
3. **`run_index.py` persists BM25 state** — now uses `index_documents_and_return_bm25` and saves to `/tmp/qdrant_calculus_bm25.pkl`.
4. **Streaming sanitizes before yield** — `generate_stream` maintains a `last_yielded_len` cursor and yields sanitized diffs.

### 🟠 High (Fixed)
5. **HyDE doc cached** — `hyde_doc` generated once, stored in variable, reused for second retrieval.
6. **Duplicate schema removed** — `StreamQuery` deleted from `endpoints.py`; `history` field added to `QueryRequest` in `schemas.py`.
7. **Sync blocking fixed** — `retrieve_and_rerank` wrapped with `loop.run_in_executor()` in both `/query` and `/query/stream`.
8. **Compression latency reduced** — per-chunk delay lowered from `request_delay * 0.5` to `min(request_delay * 0.1, 0.5s)`.
9. **Math LaTeX round-trip fixed** — `math_cleaner.py` no longer replaces `\infty`, `\le`, `\ge`, `\ne` with Unicode.

### 🟡 Medium (Fixed)
10. **`GITHUB_ACESS_TOKEN` typo fixed** — renamed to `GITHUB_ACCESS_TOKEN` in `.env`.
11. **Hardcoded path fixed** — `run_index.py` uses `Path(__file__).resolve().parent` instead of absolute path.
12. **pytest framework added** — `test_chunking.py` converted to pytest (uses `pytest>=8.0`).
13. **`evaluate.py` uses persistent Qdrant** — changed from pickled `QdrantClient` to `QdrantClient(path="/tmp/qdrant_calculus_db")`.
14. **Chunk count is live** — `/metrics` counts from Qdrant via `client.count()`.

### 🟢 Low (Fixed)
15. **`_compat.py` removed** — dead VertexAI patching code deleted (not imported anywhere).
16. **Imports moved to top** — `clean_chunk` and `sanitize_answer` now imported at module level in `generator.py`.
17. **Overlap documented** — cross-reference comments added to both `cleaner.py` and `math_cleaner.py`.

## Dependencies
- Python >= 3.11
- Key packages: langchain, langchain-qdrant, qdrant-client, fastapi, sentence-transformers, transformers, torch, pypdf, ragas
- See `backend/requirements.txt` and `backend/pyproject.toml`
