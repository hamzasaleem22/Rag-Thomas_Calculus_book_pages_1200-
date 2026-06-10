# Rag_Book - Agent Memory

## Project
RAG pipeline for "Thomas' Calculus Early Transcendentals, 14th Edition" (1262-page PDF). Hybrid search (dense + BM25 sparse) with reranking, LLM generation with citations, FastAPI backend, React frontend with glassmorphism UI and KaTeX math rendering.

## Key Files

### Backend
| File | Purpose |
|------|---------|
| `backend/app/config.py` | Settings (env: HF_TOKEN_1, HF_TOKEN_2, OPENAI_API_KEY). New: `response_cache_enabled`, `compression_skip_threshold` |
| `backend/app/retrieval/embeddings.py` | `HFInferenceAPIEmbeddings` — local sentence-transformers (all-MiniLM-L6-v2) |
| `backend/app/retrieval/vector_store.py` | `BM25SparseEmbeddings` + `HybridVectorStore` (Qdrant wrapper with chapter filtering) |
| `backend/app/retrieval/reranker.py` | Multiple methods: token_overlap, bm25, cross_encoder, hybrid. Default disabled. |
| `backend/app/retrieval/citation_verifier.py` | **NEW** — Multi-metric citation verification (token overlap, entity overlap, key terms) |
| `backend/app/retrieval/expansion.py` | Query expansion + `detect_query_type()` + `detect_chapter_reference()` |
| `backend/app/retrieval/math_cleaner.py` | Cleans garbled PDF math. Now runs at ingestion time, not per-query. |
| `backend/app/retrieval/latex_sanitizer.py` | Post-processes LLM output to fix malformed LaTeX |
| `backend/app/generation/generator.py` | Response caching (LRU 100), multi-metric citation verification, graceful degradation |
| `backend/app/ingestion/chunker.py` | **ENHANCED** — Content-type-aware chunking, min/max enforcement (100-3000 chars), LaTeX boundary detection |
| `backend/app/ingestion/cleaner.py` | Unicode fixes + LaTeX normalization + math cleaning (ingestion-time) |
| `backend/app/ingestion/pdf_structure.py` | **NEW** — Page range detection, section classification, TOC extraction |
| `backend/app/ingestion/metadata_enricher.py` | **NEW** — Chunk metadata enrichment (equation_density, key_terms, etc.) |
| `backend/app/ingestion/indexer.py` | Batch embed + PointStruct upsert to Qdrant (persistent mode) |
| `backend/app/api/endpoints.py` | FastAPI: `/query`, `/query/stream` (SSE), `/ingest`, `/health`. Adaptive relevance gate. |
| `backend/app/api/schemas.py` | Pydantic models. Citation now includes `chunk_id` and `confidence_score` |
| `backend/run_api.py` | Uvicorn entry point |

### Frontend (29 source files)
| File | Purpose |
|------|---------|
| `frontend/src/App.tsx` | Root orchestration — wires useChat + useChatHistory, session switching |
| `frontend/src/index.css` | Design system: Tailwind v4 + glass utilities + 12 keyframe animations + KaTeX overrides |
| `frontend/src/types/index.ts` | All shared TypeScript interfaces |
| `frontend/src/services/api.ts` | API client: SSE streaming via ReadableStream, sync fallback, health check |
| `frontend/src/hooks/useChat.ts` | Chat state: messages, streaming, send, regenerate, delete, cancel, loadMessages |
| `frontend/src/hooks/useChatHistory.ts` | Session management: localStorage persistence, 20 session max, CRUD |
| `frontend/src/utils/storage.ts` | localStorage CRUD helpers, session create/update/delete, formatDate |
| `frontend/src/utils/answerParser.ts` | **UPDATED** — Parses LLM output into Summary/KeyPoints/Formulas. Integrates latexSanitizer. Only extracts formulas from **Formula:** section |
| `frontend/src/utils/latexSanitizer.ts` | **NEW** — Frontend LaTeX sanitizer: fixes \fracf, f (k)→f^{(k)}, d¸ots, bare LaTeX, duplicate formulas |
| `frontend/src/components/ui/GlassCard.tsx` | Frosted-glass container (glass/glass-strong/glass-hover) |
| `frontend/src/components/ui/GradientButton.tsx` | Blue→cyan gradient button with hover glow |
| `frontend/src/components/ui/IconButton.tsx` | Small icon button with tooltip |
| `frontend/src/components/ui/Badge.tsx` | Pill badge (blue/cyan/violet/default) |
| `frontend/src/components/layout/Header.tsx` | Glass header: brand, New Chat button, History dropdown |
| `frontend/src/components/layout/ChatContainer.tsx` | Scrollable chat area with auto-scroll, message routing |
| `frontend/src/components/input/ChatInput.tsx` | Glass input bar: text input + Send/Stop toggle |
| `frontend/src/components/chat/EmptyState.tsx` | Hero welcome: floating diamond, 3 clickable example prompts |
| `frontend/src/components/chat/UserBubble.tsx` | Right-aligned gradient user message |
| `frontend/src/components/chat/AssistantBubble.tsx` | **UPDATED** — Left-aligned glass card: streaming cursor → parsed widgets. Passes citations to all widgets |
| `frontend/src/components/chat/MessageActions.tsx` | Hover toolbar: Copy, Regenerate, Delete |
| `frontend/src/components/chat/StreamingIndicator.tsx` | 3-dot pulse animation during loading |
| `frontend/src/components/widgets/AnswerSummary.tsx` | **UPDATED** — Summary card: renders $$..$$ as BlockMath, $...$ as InlineMath, [N] as CitationPopup. Sanitizes LaTeX before KaTeX |
| `frontend/src/components/widgets/KeyPointsList.tsx` | **UPDATED** — Numbered bullet list with gradient circles, inline KaTeX, CitationPopup |
| `frontend/src/components/widgets/FormulaBox.tsx` | **UPDATED** — Compact formula display: max 3 formulas, InlineMath for short, BlockMath for complex. Sanitizes LaTeX. Fixed error boundary (removed break-all) |
| `frontend/src/components/widgets/CitationCard.tsx` | Individual citation display with page/chapter badges |
| `frontend/src/components/widgets/SourcesWidget.tsx` | Collapsible citations container |
| `frontend/src/components/widgets/CitationPopup.tsx` | **NEW** — Interactive citation hover popover with page, chapter, excerpt |

### Plans & Docs
| File | Purpose |
|------|---------|
| `UI-Plan.md` | Comprehensive UI upgrade plan (design system, architecture, phases) |
| `opencode.md` | This file — agent memory |

## Data
- Chunked text: `backend/data/markdown/parsed_docs.jsonl` (1,316 chunks)
- Qdrant storage: `/tmp/qdrant_calculus_db` (persistent local QdrantClient)
- BM25 state: `/tmp/qdrant_calculus_bm25.pkl` (pickled vocab + IDF stats)
- **NOTE:** PDF extraction has 92 unique /uniXXXX artifact types, spaced-out function names (l i m), ƒ instead of f, S instead of →, p instead of π. Math is NOT in LaTeX format in source chunks.

## API & Models
- **Primary LLM**: `kr/claude-sonnet-4.5` via 9-router at `http://localhost:20128/v1`
- **Fallback LLM**: `kr/deepseek-3.2` (auto-retry on rate limit)
- **Legacy LLM**: `cx/gpt-5.5`
- **Embedding**: `all-MiniLM-L6-v2` (384-dim) — local sentence-transformers
- **Sparse**: Custom BM25 (persistent state in `/tmp/qdrant_calculus_bm25.pkl`)
- **Reranker**: Multiple methods available (token_overlap, bm25, cross_encoder, hybrid). **Default disabled** (`use_reranker=False`) due to negative quality gain.
- **Rate limiting**: 3-second delay between LLM requests (reduced from 10s for production optimization)
- **Response caching**: LRU cache (100 entries) for identical queries

## Frontend Architecture
- **Framework**: React 19 + TypeScript + Vite 8 + Tailwind CSS v4
- **Math rendering**: react-katex (KaTeX) for LaTeX → beautiful formulas
- **Theme**: Light glassmorphism — white/frosted-glass cards, blue→cyan gradient accents
- **Streaming**: SSE via `POST /api/query/stream` with ReadableStream parsing
- **Persistence**: localStorage — 20 chat sessions max, auto-save, session switcher
- **Animations**: 12+ CSS keyframe animations (no external library), stagger children
- **Build output**: ~148KB gzipped JS, ~15KB gzipped CSS

## Math Rendering Pipeline (Critical)
```
PDF chunks (garbled) → math_cleaner.py (INGESTION TIME) → clean chunks stored
    ↓
Query → retrieve clean chunks → LLM context
    ↓
LLM generates answer (with $...$ LaTeX) → latex_sanitizer.py (backend)
    ↓
SSE tokens → frontend → sanitizeLatex() (frontend) → KaTeX render
```

### Known PDF Artifact Patterns (92 total /uni codes)
- `/uni2206` → Δ, `/uni220A` → ∈, `/uni2218` → ∘, `/uni00B0` → °
- `l i m` → lim, `s i n` → sin, `c o s` → cos (127/244/213 chunks affected)
- `ƒ` → f (650 chunks), `S` → → (206 chunks)
- `d¸ots` → \dots (cedilla corruption)
- `\fracf` → `\frac{f}` (LLM sometimes omits braces)

## Running
```bash
# Backend
cd backend && source ../.venv/bin/activate && python run_api.py

# Frontend  
cd frontend && npm run dev
# Build: npm run build (output in dist/)
```

## RAGAS Results
- Faithfulness: 0.8889
- Context Precision: 0.7289
- Context Recall: 1.0000

## Final Evaluation
- **Effective accuracy: 100%** (8/8 test queries correct)
- **Strict accuracy: 87.5%** (7/8 fully precise)
- **0 wrong answers**

## Session Log

### Session: UI Upgrade + Math Rendering Fixes
**Changes made:**

1. **Full frontend rewrite** — Monolithic App.tsx (141 lines) → 29 modular files
   - Component tree: ui/, layout/, chat/, widgets/, input/
   - Custom hooks: useChat (streaming + message ops), useChatHistory (localStorage sessions)
   - Services: api.ts (SSE streaming via ReadableStream)
   - Design system: Light glassmorphism, Inter font, 12+ CSS animations

2. **Math rendering pipeline** (3 layers of cleanup):
   - `backend/app/retrieval/math_cleaner.py` — Cleans garbled PDF chunks BEFORE LLM
   - `backend/app/retrieval/latex_sanitizer.py` — Post-processes LLM output for malformed LaTeX
   - `frontend/src/utils/latexSanitizer.ts` — Frontend sanitizer for streaming text

3. **System prompt overhaul** — ChatGPT-style concise output with explicit LaTeX syntax rules:
   - Must use `\frac{num}{den}` with braces
   - Must use `\sum_{k=0}^{\infty}` with braces
   - Must wrap ALL math in `$...$` or `$$...$$`
   - Max 300 words, no filler

4. **Citation hover popups** — `CitationPopup.tsx` shows page/chapter/excerpt on hover
5. **FormulaBox compacted** — Only shows Formula-section formulas (max 3), not duplicates from summary/keypoints
6. **KaTeX error boundary fixed** — Removed `break-all` (was causing character-by-character wrapping)

### Known Issues
- Some complex LaTeX (nested fractions, multi-line equations) may still fail KaTeX rendering if the LLM produces malformed output despite the sanitizer
- The `normalizeMathText()` function in answerParser.ts can be overly aggressive — the `wrapOrphanedLatex()` may wrap prose lines that happen to contain LaTeX commands
- PDF source chunks have severe math garbling — a better PDF extractor (Nougat, Mathpix) would fundamentally improve output quality

### Session: Deep-Dive RAG Audit & Optimization (2026-06-10)

**10-Phase comprehensive pipeline audit and optimization. All phases completed.**

**New Files (4):**
- `backend/app/ingestion/pdf_structure.py` — Page range detection, section classification (front_matter/TOC/preface/chapter_content)
- `backend/app/ingestion/metadata_enricher.py` — Chunk metadata enrichment (equation_density, key_terms, chapter_number, etc.)
- `backend/app/retrieval/citation_verifier.py` — Multi-metric citation verification (token overlap + entity overlap + key terms)
- `backend/scripts/evaluate_chunks.py` — Standalone chunk quality evaluation script

**Key Changes:**

1. **Chunking (Phase 2)** — Content-type-aware chunking (prose/theorem/definition/example/equation/table) with adaptive chunk sizes, min/max enforcement (100-3000 chars), enhanced LaTeX boundary detection
2. **Metadata (Phase 3)** — Every chunk now has: `equation_density`, `has_formula`, `key_terms`, `chunk_size_tokens`, `section_number`, `chapter_number`, `content_type`
3. **Retrieval (Phase 4)** — Query type detection (equation/theorem/definition/general), adaptive relevance thresholds (0.05/0.10/0.15), HyDE auto-activation for equations, chapter-filtered retrieval
4. **Reranker (Phase 5)** — Multiple methods (token_overlap, bm25, cross_encoder, hybrid). Default disabled (`use_reranker=False`).
5. **Citations (Phase 6)** — Multi-metric verification with confidence scores. Citation schema now includes `chunk_id` and `confidence_score`. Graceful degradation removes uncited claims.
6. **Math (Phase 7)** — Math cleaning moved from runtime to ingestion time (runs once, not per-query). Reduced per-query latency.
7. **Evaluation (Phase 8)** — Added `_compute_citation_accuracy()` and `_compute_equation_fidelity()` metrics to `eval_full.py`. Created `evaluate_chunks.py` for chunk quality assessment.
8. **Production (Phase 9)** — Request delay reduced 10s→3s, response caching (LRU 100 entries), `compression_skip_threshold` config.

**Pipeline Architecture (Updated):**
```
Query → [Query Type Detection] → [Expansion] → [HyDE (auto for equations)]
  → [Hybrid Search: Dense + BM25 + Chapter Filter]
  → [Adaptive Relevance Gate: 0.05/0.10/0.15 by type]
  → [Lightweight Reranker (opt)] → [Compression (opt, skip <2K)]
  → [Response Cache Check] → [LLM Generation]
  → [Multi-Metric Citation Verification] → [LaTeX Sanitizer] → Response
```

**All 16 pytest tests pass after all 10 phases.**
