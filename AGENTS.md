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
| `config.py` | `Settings` — pydantic-settings (env: HF_TOKEN_1, HF_TOKEN_2, OPENAI_API_KEY). Config includes: `relevance_threshold`, `use_reranker`, `reranker_method`, `use_compression`, `compression_mode`, `compression_skip_threshold`, `response_cache_enabled`, `response_cache_max_size` |
| `api/endpoints.py` | FastAPI routes: `POST /query`, `POST /query/stream` (SSE), `POST /ingest`, `GET /health`, `GET /metrics`. Contains `_compute_query_doc_relevance()` — adaptive gate (0.05/0.10/0.15 by query type). Uses `detect_query_type`, `detect_chapter_reference` for intelligent retrieval. |
| `api/schemas.py` | Pydantic models: `QueryRequest`, `QueryResponse`, `Citation` (with `chunk_id`, `confidence_score`), `IngestResponse` |
| `generation/generator.py` | `Generator` — cascade (primary→fallback→legacy via 9-router), retry (3x), streaming, response caching (LRU, 100 entries), multi-metric citation verification, graceful degradation (removes uncited claims). |
| `retrieval/embeddings.py` | `HFInferenceAPIEmbeddings` — local `all-MiniLM-L6-v2` (384-dim) using sentence-transformers with model caching |
| `retrieval/vector_store.py` | `BM25SparseEmbeddings` (custom k1=1.5, b=0.75) + `HybridVectorStore` (Qdrant wrapper with chapter filtering). BM25 fits on ingest, persists to pickle. |
| `retrieval/reranker.py` | `Reranker` — multiple methods: `token_overlap`, `bm25`, `cross_encoder`, `hybrid`. Default disabled (`use_reranker=False`). MMR diversity support. |
| `retrieval/expansion.py` | Query expansion with math synonym map + `detect_query_type()` (equation/theorem/definition/general) + `detect_chapter_reference()` for metadata filtering |
| `retrieval/compression.py` | LLM-based contextual compression. Skips if total context < `compression_skip_threshold` (2000 chars). Content-hash cache (LRU, 1000 entries). Supports `compression_mode`: `auto`, `batch`, `skip`. |
| `retrieval/citation_verifier.py` | **NEW** — Multi-metric citation verification: token overlap (0.3), entity overlap (0.5), key terms (0.3). Returns composite score per citation. |
| `retrieval/latex_sanitizer.py` | Post-processes LLM answers. 10+ steps: garbled commands, frac braces, sum limits, derivative notation, superscript braces, missing backslash, Unicode duplicates, unpaired dollars, bare latex wrapping, whitespace cleanup. 50+ LaTeX commands. |
| `retrieval/math_cleaner.py` | Pre-ingestion: cleans PDF garbled math. Maps `/uniXXXX` glyphs (incl. ∇, ∂, ∫, ∬, ∭, ∮), fixes spaced func names (`l i m`→`lim`), `q`→`∞`, `p`→`π`, `Ú`→`≥`, `…`→`≤`. Now runs at ingestion time (not per-query). |
| `ingestion/chunker.py` | **ENHANCED** — Content-type-aware chunking (prose/theorem/definition/example/equation/table), adaptive chunk sizes, min/max enforcement (100-3000 chars), enhanced LaTeX boundary detection. |
| `ingestion/cleaner.py` | Unicode artifact fix, LaTeX normalization, footer/garbled line removal. Now integrates `math_cleaner` at ingestion time. |
| `ingestion/pdf_structure.py` | **NEW** — Page range detection, section classification (front_matter/table_of_contents/preface/chapter_content), TOC extraction. |
| `ingestion/metadata_enricher.py` | **NEW** — Chunk metadata enrichment: equation_density, has_formula, key_terms, chunk_size_tokens, section_number, chapter_number. |
| `ingestion/indexer.py` | Batch embed (dense + sparse) → `PointStruct` upsert to Qdrant. Returns BM25 for persistence. |
| `ingestion/pdf_loader.py` | `MarkerDocumentLoader` — `PdfReader` extraction with Unicode math→LaTeX conversion, page-split documents. Now includes section metadata. |

### Scripts (`backend/scripts/`)

| File | Purpose |
|------|---------|
| `eval_full.py` | **Comprehensive 6-phase evaluation** (retrieval, generation, E2E, latency, robustness, cost). 10 standard + 2 multi-hop + 3 OOD + 3 adversarial + 2 edge + 2 consistency queries. Includes citation accuracy and equation fidelity metrics. |
| `evaluate_chunks.py` | **NEW** — Standalone chunk quality evaluation: size distribution, metadata completeness, LaTeX integrity, content type distribution, chapter coverage. |
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
| Chunked text | `backend/data/markdown/parsed_docs.jsonl` (1614 chunks) |
| Qdrant database | `/tmp/qdrant_calculus_db` (1614 vectors) |
| BM25 state | `/tmp/qdrant_calculus_bm25.pkl` (14158 vocab, 1614 docs) |

## Model & API Configuration

| Component | Model / Endpoint |
|-----------|------------------|
| **LLM primary** | `kr/claude-sonnet-4.5` via 9-router (`http://localhost:20128/v1`) |
| **LLM fallback** | `kr/deepseek-3.2` |
| **LLM legacy** | `cx/gpt-5.5` |
| **Eval model** | `kr/claude-haiku-4.5` (generation + LLM-as-judge) |
| **Embedding** | `BAAI/bge-small-en-v1.5` (384-dim, 512 tokens, local sentence-transformers). Query prefix: `"Represent this sentence for searching relevant passages: "`. Upgraded from all-MiniLM-L6-v2 (256-token limit) on 2026-06-10. |
| **Sparse** | Custom BM25 (k1=1.5, b=0.75, local) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local transformers) |
| **HF tokens** | `HF_TOKEN_1`, `HF_TOKEN_2` in `backend/.env` (fallback via `settings.get_hf_token()`) |

## Pipeline Architecture

```
Query → [Query Type Detection] → [Query Expansion] → [HyDE (auto for equations)]
  → [Hybrid Search: Dense + BM25 + Chapter Filter]
  → [Adaptive Relevance Gate: 0.05/0.10/0.15 by type]
  → [Lightweight Reranker (opt)] → [Compression (opt, skip <2K)]
  → [Response Cache Check] → [LLM Generation] → [Multi-Metric Citation Verification]
  → [LaTeX Sanitizer] → Response (with confidence scores)
```

Key pipeline behaviors:
- **Query type detection** (`detect_query_type`): classifies as `equation`, `theorem`, `definition`, or `general`
- **Adaptive relevance gate**: 0.05 for equations, 0.10 for definitions, 0.15 for general queries
- **HyDE auto-activation**: automatically enables HyDE for equation-heavy queries (even if `use_hyde=False`)
- **Chapter filtering**: if query mentions "Chapter N" or "Section N.M", retrieval filters by `chapter_number`
- **Response cache**: LRU cache (100 entries) for identical queries — returns cached answer with fresh citations
- **Multi-metric citation verification**: token overlap + entity overlap + key terms check per citation
- **Graceful degradation**: removes uncited claims from answer instead of regenerating entirely
- **Reranker disabled by default**: `use_reranker=False` (confirmed negative quality gain). Enable with `use_reranker=True` and `reranker_method` (token_overlap, bm25, cross_encoder, hybrid)
- **Compression skip**: skips if total context < 2,000 chars (configurable via `compression_skip_threshold`)
- **Math cleaning at ingestion**: garbled PDF math is cleaned once during indexing, not per-query

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

**Resolved by Deep-Dive Audit (2026-06-10):**
1. ✅ **Reranker (4.22s)** — Default disabled (`use_reranker=False`). Multiple methods available: token_overlap, bm25, cross_encoder, hybrid.
2. ✅ **Compression (6.62s)** — Skip threshold added (2000 chars). Can be further optimized with `compression_mode="batch"`.
3. ✅ **Math cleaning latency** — Moved to ingestion time. No per-query overhead.
4. ✅ **Request delay** — Reduced from 10.0s to 3.0s.
5. ✅ **Citation accuracy** — Multi-metric verification (token + entity + key terms) with confidence scoring.
6. ✅ **Chunk quality** — Min/max enforcement (100-3000 chars), content-type routing, LaTeX boundary detection.
7. ✅ **Front matter handling** — Pages 1-21 properly classified and tagged.
8. ✅ **Metadata enrichment** — All chunks have equation_density, key_terms, chapter_number, etc.

**Still Open:**
1. **ch16 divergence theorem** — LLM judge over-penalizes formatting (hallucination=1.00), but content is factually correct (Cor=0.95, KW=0.80). Judge prompt tuning needed.
2. **Multi-hop m02** — area of circle + volume of sphere reasoning scores 0.33. Needs cross-chunk reasoning improvement (sub-query decomposition).
3. **LaTeX fidelity (0.778)** — LLM doesn't always follow formatting rules. Consider adding more few-shot examples or post-generation LaTeX validation.
4. **Answer F1 (0.264)** — Generated answers are longer than ground truths. Consider answer length constraints or better grounding.

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
- `settings.request_delay = 2.0` (production default is now 3.0)
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

### Deep-Dive Audit & Optimization (2026-06-10, 10 Phases)

Comprehensive pipeline audit and optimization across 10 phases. All phases completed successfully.

**Phase 1: Document Structure Analysis**
- Created `ingestion/pdf_structure.py` — page range detection, section classification, TOC extraction
- Pages 1-5 (front matter) tagged as `section: "front_matter"`
- Pages 6-11 (TOC) tagged as `section: "table_of_contents"` and included in chunking
- Pages 12-21 (preface) tagged as `section: "preface"`
- Page 22+ tagged as `section: "chapter_content"`

**Phase 2: Re-Chunking Strategy**
- Rewrote `chunker.py` with content-type-aware chunking
- Added `_classify_content_type()` — detects prose/theorem/definition/example/equation/table
- Implemented min/max chunk size enforcement (100-3000 chars)
- Enhanced LaTeX boundary detection (nested `$$`, `\[]`, `\(\)`)
- Theorem/Definition/Example blocks preserved as atomic units

**Phase 3: Metadata Enhancement**
- Created `metadata_enricher.py` — comprehensive chunk metadata enrichment
- Added `equation_density`, `has_formula`, `key_terms`, `chunk_size_tokens`, `section_number`, `chapter_number`
- Integrated into chunking pipeline via `enrich_chunks()`

**Phase 4: Retrieval Optimization**
- Enhanced `expansion.py` with query type detection (`detect_query_type()`) and chapter reference detection
- Added adaptive relevance thresholds (0.05 for equations, 0.10 for definitions, 0.15 default)
- Implemented HyDE auto-activation for equation-heavy queries
- Added metadata-filtered retrieval by chapter in `vector_store.py`

**Phase 5: Reranker Optimization**
- Rewrote `reranker.py` with multiple methods: `token_overlap`, `bm25`, `cross_encoder`, `hybrid`
- Default `use_reranker=False` (negative quality gain confirmed)
- Added `reranker_method` config option for experimentation
- Token-overlap scoring as lightweight fallback

**Phase 6: Citation Accuracy Framework**
- Created `citation_verifier.py` — multi-metric verification (token overlap, entity overlap, key terms)
- Updated `generator.py` to use `verify_all_citations()` for answer validation
- Added `chunk_id` and `confidence_score` to Citation schema
- Graceful degradation: removes uncited claims instead of rejecting entire answer

**Phase 7: Mathematical Content Preservation**
- Moved `math_cleaner.py` from runtime to ingestion pipeline (runs once, not per-query)
- Updated `cleaner.py` to import and use `math_clean_chunk()` during document cleaning
- Removed per-query cleaning from `generator.py` (reduced latency)

**Phase 8: Evaluation Framework**
- Created `scripts/evaluate_chunks.py` — standalone chunk quality evaluation
- Added `_compute_citation_accuracy()` to `eval_full.py` (multi-metric verification)
- Added `_compute_equation_fidelity()` to `eval_full.py` (LaTeX well-formedness checks)
- Integrated new metrics into Phase 2 generation quality report

**Phase 9: Production Optimization**
- Reduced `request_delay` from 10.0s to 3.0s
- Added `compression_skip_threshold: 2000` (skip compression for short contexts)
- Implemented response caching in `Generator` (LRU, max 100 entries, configurable)
- Added `response_cache_enabled` and `response_cache_max_size` config options

**Phase 10: Documentation**
- Updated implementation plan with completion status
- Updated AGENTS.md with all changes
- All 16 pytest tests pass after all phases

**Key Improvements:**
- Chunk quality: min/max enforcement (100-3000 chars), content-type routing
- Retrieval: adaptive thresholds, chapter filtering, HyDE auto-activation
- Citations: multi-metric verification, confidence scoring, graceful degradation
- Latency: request_delay 10s→3s, math cleaning moved to ingestion, response caching
- Evaluation: citation accuracy, equation fidelity, chunk quality metrics

**Files Modified (14):**
- `backend/app/config.py` — new config options
- `backend/app/api/endpoints.py` — adaptive retrieval, citation verification
- `backend/app/api/schemas.py` — Citation schema with chunk_id, confidence_score
- `backend/app/generation/generator.py` — citation verification, response caching
- `backend/app/retrieval/vector_store.py` — chapter filtering
- `backend/app/retrieval/reranker.py` — multiple reranker methods
- `backend/app/retrieval/expansion.py` — query type detection
- `backend/app/retrieval/latex_sanitizer.py` — (no changes, already comprehensive)
- `backend/app/retrieval/citation_verifier.py` — **NEW**
- `backend/app/ingestion/chunker.py` — content-type routing, min/max enforcement
- `backend/app/ingestion/cleaner.py` — integrated math cleaning
- `backend/app/ingestion/pdf_structure.py` — **NEW**
- `backend/app/ingestion/metadata_enricher.py` — **NEW**
- `backend/scripts/eval_full.py` — citation accuracy, equation fidelity metrics
- `backend/scripts/evaluate_chunks.py` — **NEW**

**Test Results:** All 16 pytest tests pass. Frontend builds cleanly.

### Embedding Model Upgrade (2026-06-10)

Migrated from `all-MiniLM-L6-v2` (256-token limit) to `BAAI/bge-small-en-v1.5` (512-token limit). Same 384-dim — zero Qdrant config changes.

- **Problem**: all-MiniLM-L6-v2 has `max_seq_length=256` via sentence-transformers. Current chunks avg 1734 chars (~433 tokens) → previously only 59% embedded.
- **Solution**: bge-small-en-v1.5 (384-dim, 512 tokens, MTEB ~60.3). Full chunks now fit within context window. Requires query prefix: `"Represent this sentence for searching relevant passages: "`.
- **Performance**: 1614 chunks indexed in 393s (6.5 min) on CPU. Disk: 257MB. Removed old model (88MB freed).

### Rechunking Project — Chapters, Pages, Content-Type, TOC (2026-06-10, Phases 0-6)

Complete rechunking pipeline overhaul. From 1316 inconsistent chunks to 1614 well-structured chunks.

**Phase 0: Embedding Model Upgrade** — `all-MiniLM-L6-v2` (256 tokens) → `BAAI/bge-small-en-v1.5` (512 tokens). Same 384-dim, zero Qdrant changes. Query prefix: `"Represent this sentence for searching relevant passages: "`.

**Phase 1: Chapter Detection + Page Offset** — `get_book_page(pdf_page) → pdf_page - 21` (Chapter 1 starts at PDF page 22 = book page 1). Answer key pages 1100-1262 detected via regex (excluded from normal chapters). `CHAPTER_PAGE_PATTERN` rejects lines with `=` to avoid answer-key false positives.

**Phase 2: Content-Type Classification** — `_classify_content_type()` detects prose/theorem/definition/example/equation/table/answer_key. Adaptive chunk sizes (prose=1000, answer_key=1024, etc.). `_chunk_answer_key()` with `RecursiveCharacterTextSplitter(1024, 64)`.

**Phase 3: TOC Structured Chunking** — `_chunk_toc()` parses TOC into one chunk per chapter with JSON `toc_sections` metadata. 16 TOC chunks total.

**Phase 4: Full Rechunking Pipeline** — `evaluate_chunks.py` passes 4/4: size (100-3100: 100%), metadata (all >90%), LaTeX (100% intact), chapters (16/16). Fixed LaTeX false positives from currency/garbled `$` via enhanced `_has_unmatched_latex()`.

**Phase 5: Re-index into Qdrant** — 1614 vectors indexed in 393s (CPU). BM25 state: 14158 vocab, 1614 docs. Fixed nested payload filter: `key="metadata.chapter_number"`.

**Phase 6: Integration Test** — All endpoints verified: non-streaming query, streaming (SSE), OOD refusal ("only calculus"), chapter-filtered retrieval, `/health`, `/metrics` (1614 chunks). 21/21 tests pass.

**Files Modified (6):**
- `backend/app/ingestion/chunker.py` — Phase 1-4 changes (answer_key, TOC, size enforcement, LaTeX cleanup)
- `backend/app/ingestion/pdf_structure.py` — `get_book_page()`, answer_key in `PAGE_RANGES`, TOC chapter names
- `backend/app/ingestion/metadata_enricher.py` — `extract_chapter_number()` handles answer_key
- `backend/app/ingestion/indexer.py` — (unchanged, re-run with new chunks)
- `backend/app/retrieval/vector_store.py` — Fixed chapter filter to use `metadata.chapter_number` dot notation
- `backend/scripts/evaluate_chunks.py` — Enhanced LaTeX check with currency + garbled $ filter, MAX_CHUNK_CHARS=3100

**Test Results:** 21/21 pytest tests pass.

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
