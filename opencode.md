# Rag_Book - Agent Memory

## Project
RAG pipeline for "Thomas' Calculus Early Transcendentals, 14th Edition" (1262-page PDF). Hybrid search (dense + BM25 sparse) with reranking, LLM generation with citations, FastAPI backend, React frontend with glassmorphism UI and KaTeX math rendering.

## Key Files

### Backend
| File | Purpose |
|------|---------|
| `backend/app/config.py` | Settings (env: HF_TOKEN_1, HF_TOKEN_2, OPENAI_API_KEY) |
| `backend/app/retrieval/embeddings.py` | `HFInferenceAPIEmbeddings` — local sentence-transformers (all-MiniLM-L6-v2) |
| `backend/app/retrieval/vector_store.py` | `BM25SparseEmbeddings` + `HybridVectorStore` (Qdrant wrapper) |
| `backend/app/retrieval/reranker.py` | Reranker with `cross-encoder/ms-marco-MiniLM-L-6-v2` (local transformers) |
| `backend/app/retrieval/math_cleaner.py` | **NEW** — Cleans garbled PDF math in chunks before sending to LLM (fixes /uniXXXX, spaced functions, ƒ→f, S→→) |
| `backend/app/retrieval/latex_sanitizer.py` | **NEW** — Post-processes LLM output to fix malformed LaTeX (\fracf→\frac{f}, missing $...$, d¸ots→\dots) |
| `backend/app/generation/generator.py` | **UPDATED** — System prompt enforces ChatGPT-style concise LaTeX with proper \frac{}{} braces. Integrates math_cleaner + latex_sanitizer |
| `backend/app/ingestion/chunker.py` | **OPTIMIZED** — 1024-token chunks with 128 overlap, theorem boundary detection |
| `backend/app/ingestion/cleaner.py` | **OPTIMIZED** — Unicode math fixes, LaTeX normalization, footer removal |
| `backend/app/ingestion/indexer.py` | Batch embed + PointStruct upsert to Qdrant (persistent mode) |
| `backend/app/api/endpoints.py` | FastAPI: `/query`, `/query/stream` (SSE), `/ingest`, `/health` |
| `backend/app/api/schemas.py` | Pydantic models |
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
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` — local transformers with MMR diversity
- **Rate limiting**: 10-second delay between LLM requests

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
PDF chunks (garbled) → math_cleaner.py → clean chunks → LLM context
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
