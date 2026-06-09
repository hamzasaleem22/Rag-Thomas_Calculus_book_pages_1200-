# Rag_Book — Agent Memory

## Project Overview

RAG pipeline for "Thomas' Calculus Early Transcendentals, 14th Edition" (1262-page PDF). Hybrid search (dense + BM25 sparse) with reranking, LLM generation with citations, FastAPI backend, React frontend.

## Quick Start

```bash
# Backend (from repo root)
source .venv/bin/activate && cd backend && python3 -m uvicorn app.api.endpoints:app --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
cd frontend && npm run dev   # proxies /api → backend on port 8000

# Tests
cd backend && python -m pytest tests/test_chunking.py -v
```

## Key Files

### Backend Core (`backend/app/`)

| File | Purpose |
|------|---------|
| `config.py` | `Settings` — pydantic-settings (env: HF_TOKEN_1, HF_TOKEN_2, OPENAI_API_KEY). Config includes: `relevance_threshold`, `use_reranker`, `use_compression`, `compression_mode`, `rerank_skip_threshold` |
| `api/endpoints.py` | FastAPI routes: `POST /query`, `POST /query/stream` (SSE), `POST /ingest`, `GET /health`, `GET /metrics`. Contains `_compute_query_doc_relevance()` — gates generation if token overlap < `relevance_threshold` (0.15). Uses `run_in_executor` for blocking retrieval. |
| `api/schemas.py` | Pydantic models: `QueryRequest` (query, top_k, rerank, use_mmr, history), `QueryResponse` (answer, citations, sources), `Citation`, `IngestResponse` |
| `generation/generator.py` | `Generator` — cascade (primary→fallback→legacy via 9-router), retry (3x), streaming, self-consistency (3 samples), citation verification, answer validation (must cite ≥1 doc, each claim must have 30%+ token overlap with cited chunk). System prompt has OOD/refusal instructions and LaTeX formatting exemplars. |
| `retrieval/embeddings.py` | `HFInferenceAPIEmbeddings` — local `all-MiniLM-L6-v2` (384-dim) using sentence-transformers with model caching |
| `retrieval/vector_store.py` | `BM25SparseEmbeddings` (custom k1=1.5, b=0.75) + `HybridVectorStore` (Qdrant wrapper). BM25 fits on ingest, persists to pickle. `from_pickle()` loads from `/tmp/qdrant_calculus_bm25.pkl` |
| `retrieval/reranker.py` | `Reranker` — `cross-encoder/ms-marco-MiniLM-L-6-v2` local transformers, supports MMR diversity (lambda=0.7). Adds ~4.2s P50 latency with negative quality gain (−0.05). |
| `retrieval/expansion.py` | Query expansion with math synonym map. Key mappings: `divergence theorem→gauss theorem`, `flux→surface integral`, `L'Hôpital→l'hopital`, `derivative→differentiation` |
| `retrieval/compression.py` | LLM-based contextual compression. Skips if total context < 8K chars. Content-hash cache (LRU, 1000 entries). Supports `compression_mode`: `auto` (per-chunk calls), `batch` (single call), `skip` (skip all). Per-chunk key-term check: if chunk already has all query key terms, skips compression for that chunk. |
| `retrieval/hyde.py` | HyDE — generates hypothetical textbook passage (disabled by default: `use_hyde=False`) |
| `retrieval/latex_sanitizer.py` | Post-processes LLM answers. 10 steps: garbled commands, frac braces, sum limits, derivative notation, superscript braces, missing backslash, Unicode duplicates, unpaired dollars, bare latex wrapping, whitespace cleanup. Covers 50+ LaTeX commands. |
| `retrieval/math_cleaner.py` | Pre-retrieval: cleans PDF garbled math. Maps `/uniXXXX` glyphs (incl. ∇, ∂, ∫, ∬, ∭, ∮), fixes spaced func names (`l i m`→`lim`), `q`→`∞`, `p`→`π`, `Ú`→`≥`, `…`→`≤`. |
| `ingestion/chunker.py` | Markdown header splitter + RecursiveCharacterTextSplitter (token-estimated, chunk_size=1024, overlap=128). Injects `## Section` headers, enforces theorem boundaries, merges broken LaTeX. |
| `ingestion/cleaner.py` | Unicode artifact fix (`/uni2032`→`'`), LaTeX normalization (`\[`→`$$`), footer/garbled line removal. Complements `math_cleaner.py` (different pipeline stage). |
| `ingestion/indexer.py` | Batch embed (dense + sparse) → `PointStruct` upsert to Qdrant. Returns BM25 for persistence. |
| `ingestion/pdf_loader.py` | `MarkerDocumentLoader` — `PdfReader` extraction with Unicode math→LaTeX conversion, page-split documents. |

### Scripts (`backend/scripts/`)

| File | Purpose |
|------|---------|
| `eval_full.py` | **Comprehensive 6-phase evaluation** (retrieval, generation, E2E, latency, robustness, cost). 10 standard + 2 multi-hop + 3 OOD + 3 adversarial + 2 edge + 2 consistency queries. Uses `kr/claude-haiku-4.5`. OOD/refusal detection patterns include `"i'm sorry"` and `"only answer questions about calculus"`. |
| `debug_failures.py` | Model comparison debug script for failing queries |
| `ingest.py` | CLI ingestion pipeline (PDF→markdown→clean→chunk→index) |
| `index_full.py` | Full-book index with BM25 persistence to `/tmp/qdrant_calculus_bm25.pkl` (preferred over `run_index.py`) |

### Tests (`backend/tests/`)

| File | Purpose |
|------|---------|
| `test_chunking.py` | 16 pytest tests covering chunking, cleaning, BM25, query expansion |

## Data Locations

| What | Path |
|------|------|
| Chunked text | `backend/data/markdown/parsed_docs.jsonl` (1316 chunks) |
| Qdrant database | `/tmp/qdrant_calculus_db` (1316 vectors) |
| BM25 state | `/tmp/qdrant_calculus_bm25.pkl` (14254 vocab, 1316 docs) |

## Model & API Configuration

| Component | Model / Endpoint |
|-----------|------------------|
| **LLM primary** | `kr/claude-sonnet-4.5` via 9-router (`http://localhost:20128/v1`) |
| **LLM fallback** | `kr/deepseek-3.2` |
| **LLM legacy** | `cx/gpt-5.5` |
| **Eval model** | `kr/claude-haiku-4.5` (generation + LLM-as-judge) |
| **Embedding** | `all-MiniLM-L6-v2` (384-dim, local sentence-transformers) |
| **Sparse** | Custom BM25 (k1=1.5, b=0.75, local) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local transformers) |
| **HF tokens** | `HF_TOKEN_1`, `HF_TOKEN_2` in `backend/.env` (fallback via `settings.get_hf_token()`) |

## Pipeline Architecture

```
Query → [Query Expansion] → [HyDE (opt)] → [Hybrid Search: Dense + BM25]
  → [Relevance Gate: token overlap > 0.15?] → [Reranker (opt)]
  → [Compression (opt)] → [LLM Generation] → [LaTeX Sanitizer] → Response
```

Key pipeline behaviors:
- **Relevance gate** (`_compute_query_doc_relevance`): if query terms (minus stopwords) have < 15% overlap with top-3 retrieved docs, returns empty → triggers OOD refusal
- **OOD refusal**: "I'm sorry, I can only answer questions about calculus from Thomas' Calculus, 14th Edition."
- **Compression skip**: if total context < 8,000 chars → skip compression entirely. Also skips per-chunk if chunk already contains all query key terms.
- **Compression cache**: content-hash based (MD5 of query + first 200 chars of doc). LRU eviction at 1000 entries.
- **Validation regen**: if answer has no citations, or a cited claim has < 30% token overlap with the cited doc, regenerates with stricter prompt.

## Evaluation Results (2026-06-09) — Fix Round 1

After fixes from `plan/fix-evaluation-issues-1.md`. Using `kr/claude-haiku-4.5`.

### Phase 1: Retrieval Quality

| Metric | Score |
|--------|-------|
| NDCG@5 | 1.000 |
| MRR | 1.000 |
| Hit Rate@5 | 100% (10/10) |
| Hit Rate@30 | 100% (10/10) |
| Hybrid/Dense/BM25@10 | 100% all modes |
| Avg retrieval latency | 1.09s (P95: 4.93s) |

### Phase 2: Generation Quality

| Metric | Score | Δ from prev |
|--------|:-----:|:-----------:|
| Answer Correctness | 0.940 | +0.005 |
| Answer Relevancy | 0.796 | −0.033 |
| Keyword Coverage | 0.718 | −0.038 |
| Hallucination Rate | 0.317 (lower=better) | **−0.050** |
| LaTeX Fidelity | 0.778 | +0.014 |
| Citations in answers | 100% | — |

> ch16 divergence theorem: judge hallucination=1.00 but factual correctness=0.95, keyword coverage=0.80. The LLM judge over-penalizes formatting, not actual errors.

### Phase 3: End-to-End Metrics

| Metric | Score | Notes |
|--------|:-----:|-------|
| Answer F1 | 0.264 | Low (generated answers longer than GTs) |
| Exact Match | 0.000 | Expected (format mismatch) |
| Query Consistency | 0.784 | Same Q, different phrasing → 78.4% similarity |
| CSAT Proxy | 0.670 | Lower due to refusal-style answers |

### Phase 4: Latency & Throughput

| Stage | Before P50 | After P50 | Δ |
|-------|:---------:|:---------:|:-:|
| Dense embed + search | 0.36s | 0.26s | −0.10s |
| Reranker | 4.22s | 4.22s | — |
| Compression | 12.42s | **6.62s** | **−5.80s (−47%)** |
| Generation | 10.71s | 6.58s | −4.13s |
| **Total pipeline** | **29.62s** | **19.54s** | **−10.08s (−34%)** |

### Phase 5: Robustness & Safety

| Metric | Before | After | Assessment |
|--------|:-----:|:-----:|------------|
| OOD Handling | 0.000 | **1.000** | Correctly refuses 3/3 OOD queries |
| Adversarial Robustness | 1.000 | 1.000 | Typos/symbols: zero effect |
| Multi-hop Reasoning | 0.779 | 0.584 | m01 (L'Hôpital+Taylor)=0.83, m02 (area/volume)=0.33 |
| Confidence Calibration | 0.000 | **1.000** | Correct refusal on non-calculus |
| Edge Cases | Mixed | Mixed | Empty→refusal, narrow→correct |

### Phase 6: Cost & Efficiency

| Metric | Value |
|--------|:-----:|
| Cost per query | $0.012 |
| Cost per 1K queries | $12.20 |
| Token efficiency | 8.5% |
| Reranker quality gain | −0.05 (negative — adds latency, no keyword coverage gain) |

### Remaining Issues

1. **ch16 divergence theorem** — LLM judge over-penalizes formatting (hallucination=1.00), but content is factually correct (Cor=0.95, KW=0.80)
2. **Multi-hop m02** — area of circle + volume of sphere reasoning scores 0.33. Needs cross-chunk reasoning improvement.
3. **LaTeX fidelity (0.778)** — LLM doesn't always follow formatting rules in prompts
4. **Reranker (4.22s)** — negative quality gain (−0.05). Try `use_reranker=False`
5. **Compression (6.62s)** — still #1 bottleneck. Try `compression_mode="batch"`

## Critical Conventions

### Vite Proxy
Frontend sends to `/api/query`, backend serves `/query`. The proxy in `frontend/vite.config.ts` strips the `/api` prefix:
```ts
proxy: { '/api': { target: 'http://localhost:8000', rewrite: path => path.replace(/^\/api/, '') } }
```

### Server CWD
Must run `uvicorn` from `backend/` directory, not repo root. The `app` package lives inside `backend/`, so Python needs the CWD to include `backend/` in the module path.

### Config Overrides (eval script)
`eval_full.py` overrides these settings for evaluation:
- `settings.request_delay = 2.0`
- `settings.llm_model = "kr/claude-haiku-4.5"`
- Uses a separate `judge_client` with `JUDGE_MODEL = "kr/claude-haiku-4.5"` and `rate_limit(4.0)`

## Implementation History

### Phase 1-9 (Initial Build)
All complete: PDF extraction, chunking, Qdrant index, reranking, generation, RAGAS eval, FastAPI, React UI, streaming.

### Code Review Fixes (17 issues, 2026-06-09)
**Critical:** Hardcoded API key removed, `/metrics` made dynamic, `run_index.py` persists BM25, streaming sanitizes incrementally.
**High:** HyDE document caching, duplicate schema removed, sync blocking → `run_in_executor`, compression delay optimized, math LaTeX round-trip fixed.
**Medium:** GitHub token typo, hardcoded paths, pytest framework, `evaluate.py` uses persistent Qdrant, chunk count live.
**Low:** Dead code removed, imports at module level, cross-reference comments.

### Eval Fix Round 1 (22 tasks, 2026-06-09)
**OOD/Refusal:** System prompt instructions + `_compute_query_doc_relevance()` gate in `endpoints.py` + updated eval refusal patterns.
**Compression:** Content-hash cache, < 8K skip heuristic, key-term per-chunk skip, `batch` mode, batched rate limiting.
**Vector calculus:** `/uni2207`→`\nabla`, `/uni2202`→`\partial`, `/uni222B`→`\int`, `/uni222C`→`\iint`, `/uni222D`→`\iiint`, `/uni2A0F`→`\oint`, `/uni00D7`→`\times`.
**Query expansion:** `divergence theorem→gauss theorem`, `flux→surface integral`, `gauss law→gauss's law`.
**LaTeX sanitizer:** 50+ command coverage, superscript brace fix, unpaired dollar fix, missing backslash fix.
**Config:** Added `relevance_threshold` (0.15), `use_reranker` (True), `use_compression` (True), `compression_mode` ("auto"), `rerank_skip_threshold` (0).

### LaTeX Rendering Fix (2026-06-09, Round 2)

**Problem:** Math equations displayed as raw LaTeX text (`\sum_infty`, `\fracf(k)`, etc.) instead of rendered by KaTeX.

**Root Cause Analysis:**
1. **Streaming bypassed KaTeX** — AssistantBubble.tsx rendered streaming content as raw `<p>` text, no math processing
2. **Missing `$...$` delimiters** — Bare LaTeX not wrapped, so KaTeX never triggered
3. **Malformed LaTeX** — `\sum_infty`, `\fracf(k)` syntax invalid; regex fixers incomplete
4. **LLM ignored brace rules** — Generated `\fracdydx` instead of `\frac{dy}{dx}` despite prompt warnings

**Frontend Fixes (5 files):**
- **MathErrorBoundary.tsx** (new) — Extracted shared error boundary; catches KaTeX parse failures, degrades to raw LaTeX
- **AssistantBubble.tsx** — Streaming now uses `parseAnswer()` + widget pipeline (AnswerSummary/KeyPointsList/FormulaBox); math renders incrementally with error protection
- **AnswerSummary.tsx** — Wrapped all `InlineMath`/`BlockMath` in `MathErrorBoundary`
- **KeyPointsList.tsx** — Wrapped all `InlineMath` in `MathErrorBoundary`
- **FormulaBox.tsx** — Imports shared `MathErrorBoundary` instead of local class

**Backend Fixes (2 files):**
- **latex_sanitizer.py** — Added 3 new functions:
  - `_fix_frac_multichar()` — Handles `\fracdydx`→`\frac{dy}{dx}` via differential token whitelist (dy, dx, du, dv, dt, etc.)
  - `_fix_missing_cdot()` — Fixes `dot`→`\cdot` in math contexts (handles chain rule `}dot\frac`)
  - Enhanced `_fix_sum_limits()` — Now handles `\sum_infty`, `\sum infty` patterns (was only `\sum k=0 \infty`)
  - Enhanced `_wrap_bare_latex()` — More aggressive wrapping: any LaTeX density ≥40% gets wrapped; lowered non-LaTeX word threshold from 2→3
  - Extended `_MATH_FUNCTIONS` — Added `cdot`, `dots`, `infty`, `pi`, vector calculus symbols
- **generator.py** — Added `\fracdydx` and `\fracdydudot\fracdudx` as explicit WRONG examples in system prompt

**Impact:**
- Streaming math now renders visibly as it arrives (not as plain text)
- Incomplete LaTeX during stream shows as fallback text instead of crashing
- Bare math expressions wrapped in `$$...$$` so KaTeX processes them
- Multi-char frac patterns fixed (dy/dx, du/dv, etc.)
- LLM more strongly warned against malformed patterns

**Test Results:** All 16 pytest tests pass. Frontend builds cleanly (489KB gzipped JS).

## Ground Truth Queries (eval_full.py)

| Category | Count | Examples |
|----------|-------|---------|
| Standard | 10 | ch02-ch16, one per chapter |
| Multi-hop | 2 | L'Hôpital+Taylor, area+volume |
| OOD | 3 | Capital of France, poem about calc, meaning of life |
| Adversarial | 3 | Typos, missing chars, symbols: "Wht is L'Hopital's rul?" |
| Edge | 2 | Empty query, epsilon-delta proof |
| Consistency | 2 | Same Q different phrasing: "d/dx sin(x)" vs "derivative of sine" |

## Dependencies

- Python >= 3.11
- Key packages: langchain, langchain-qdrant, qdrant-client, fastapi, sentence-transformers, transformers, torch, pypdf, ragas, openai
- See `backend/requirements.txt` and `backend/pyproject.toml`
- No external API dependencies — everything goes through the local 9-router at `localhost:20128/v1`
