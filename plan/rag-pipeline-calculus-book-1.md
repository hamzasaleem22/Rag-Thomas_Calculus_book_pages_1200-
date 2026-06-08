---
goal: Production-Grade RAG Pipeline for Thomas' Calculus Textbook
version: 1.0
date_created: 2026-06-07
owner: Developer
status: 'Planned'
tags: rag, langchain, calculus, pdf-processing, production, evaluation
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

Implementation plan for a production-grade **GraphRAG pipeline** built on LangChain for **Thomas' Calculus Early Transcendentals, 14th Edition** — **CPU-only environment**. The pipeline ingests a 1200+ page calculus PDF with heavy mathematical notation, indexes it with dense embeddings (`BAAI/bge-small-en-v1.5` — 384d) and hybrid search (Qdrant), reranks results (`BAAI/bge-reranker-v2-mini` — lightweight cross-encoder), and generates answers with inline citations via GPT-4o-mini. All Hugging Face models are accessed via the **Inference API** (no local download) using a Hugging Face access token. Evaluated with RAGAS metrics. UI: FastAPI backend + React frontend with inline citation popups.

## 1. Requirements & Constraints

- **REQ-001**: PDF ingestion must preserve mathematical equations (LaTeX extraction required)
- **REQ-002**: Must use Hugging Face models for embeddings and reranking via **Inference API** (no local download, uses `HF_TOKEN`)
- **REQ-003**: LLM generation via OpenAI GPT-4o-mini API
- **REQ-004**: Inline citation system — citations appear inside the answer text as clickable markers; clicking opens a popup showing the source page snippet
- **REQ-005**: Hybrid search (dense + sparse/BM25) for better math formula lookups
- **REQ-006**: RAGAS evaluation pipeline for faithfulness, answer relevancy, context precision
- **REQ-007**: FastAPI backend + React frontend
- **REQ-008**: Local-first deployment, containerization later
- **REQ-009**: Chunking must respect document hierarchy (chapters, sections, subsections)
- **CON-001**: Hugging Face models accessed via **Inference API** (no local download, requires `HF_TOKEN`)
- **CON-002**: **CPU-only environment** — no GPU available. Inference API avoids local compute constraints entirely
- **CON-003**: Embedding dimension = 384 (BGE-small-en-v1.5)
- **PAT-001**: Follow LangChain Expression Language (LCEL) for composable chains
- **PAT-002**: Use LangGraph for multi-step retrieval workflow
- **PAT-003**: Use Hugging Face Inference API for HF models (no local download)

## 2. Implementation Steps

---

## Phase Report: Status & Checklist

At the end of each phase implementation, a **Phase Report** will be generated and appended to `opencode.md`. The report follows this template:

```
## Phase N Report: [Phase Name]

**Status**: ✅ SUCCESS / ❌ FAILED / ⏳ PARTIAL
**Date**: YYYY-MM-DD

### Checklist

| # | Criterion | Pass/Fail | Notes |
|---|-----------|-----------|-------|
| 1 | [criterion 1] | ✅/❌ | ... |
| 2 | [criterion 2] | ✅/❌ | ... |

### Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| ...    | ...    | ...    | ✅/❌  |

### Issues Encountered

- [issue 1] → [resolution]
- [issue 2] → [resolution]

### Next Phase Ready? ✅ / ❌
```

---

### Implementation Phase 1: Project Scaffolding & PDF Ingestion

- GOAL-001: Set up project structure, dependencies, and PDF extraction pipeline

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Initialize project with Poetry/pyproject.toml: `langchain`, `langchain-community`, `langchain-qdrant`, `langchain-openai`, `langchain-huggingface`, `sentence-transformers`, `marker-pdf`, `pypdf`, `fastapi`, `uvicorn`, `ragas`, `qdrant-client`, `semantic-text-splitter` | | |
| TASK-002 | Create directory structure: `app/`, `app/ingestion/`, `app/retrieval/`, `app/generation/`, `app/api/`, `app/evaluation/`, `frontend/`, `data/`, `config/`, `tests/`, `scripts/` | | |
| TASK-003 | Implement PDF ingestion script using Marker CLI to convert PDF → Markdown (preserves LaTeX equations, section headers, page numbers) | | |
| TASK-004 | Write custom LangChain document loader that reads Marker output Markdown files with page-number metadata | | |
| TASK-005 | Implement cleaning pipeline: remove extraneous artifacts, normalize LaTeX notation, handle multi-column layouts | | |

**Phase 1 Checklist:**

| # | Criterion | Pass/Fail | Notes |
|---|-----------|-----------|-------|
| 1 | All dependencies installed without conflicts | ❌ | |
| 2 | Directory structure matches plan exactly | ❌ | |
| 3 | Marker CLI successfully converts PDF → Markdown with equations intact | ❌ | |
| 4 | Custom document loader reads Markdown and preserves page numbers | ❌ | |
| 5 | Cleaning pipeline removes artifacts without losing content | ❌ | |
| 6 | Sample output verified: 3 random pages checked for LaTeX integrity | ❌ | |

### Implementation Phase 2: Chunking Strategy

- GOAL-002: Implement intelligent chunking that preserves document hierarchy and equation integrity

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-006 | Implement `MarkdownHeaderTextSplitter` to split by chapter (level 1), section (level 2), subsection (level 3) — preserve hierarchy as metadata | | |
| TASK-007 | Apply `RecursiveCharacterTextSplitter` as secondary splitter: chunk_size=512, chunk_overlap=50, separators=["\n\n", "\n", " ", ""] | | |
| TASK-008 | Add post-processing: merge chunks that break equations mid-LaTeX (check for unmatched `$$` or `$` delimiters) | | |
| TASK-009 | Store metadata per chunk: `chapter`, `section`, `subsection`, `page_number`, `chunk_index`, `is_equation` flag | | |

**Phase 2 Checklist:**

| # | Criterion | Pass/Fail | Notes |
|---|-----------|-----------|-------|
| 1 | Chunks respect chapter/section boundaries (no cross-section chunks) | ❌ | |
| 2 | No LaTeX equations broken across chunks (unmatched $$/$ = 0) | ❌ | |
| 3 | Average chunk size within 450-550 token range | ❌ | |
| 4 | Metadata correctly populated for every chunk | ❌ | |
| 5 | Total chunk count reasonable for 1200-page book (~8k-15k chunks) | ❌ | |

### Implementation Phase 3: Embeddings & Vector Store

- GOAL-003: Set up embedding pipeline with **BAAI/bge-small-en-v1.5** (CPU-optimized, 384d) and Qdrant vector store with hybrid search

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-010 | Set up BGE-small-en-v1.5 via Hugging Face Inference API (`huggingface_hub.InferenceClient`). Embeddings called as API with `HF_TOKEN`. No local model download | | |
| TASK-011 | Spin up Qdrant locally via Docker (or in-memory for dev) with 384-dim vector config | | |
| TASK-012 | Create Qdrant collection with named vectors: `dense` (384d, cosine) + enable payload-based sparse indexing for hybrid search | | |
| TASK-013 | Implement indexing pipeline: batch embed chunks (batch_size=64 — larger batch due to smaller model) → upsert to Qdrant with full payload (text + metadata) | | |
| TASK-014 | Implement hybrid search query builder: combine dense vector similarity + BM25 sparse score with `alpha=0.5` | | |

**Phase 3 Checklist:**

| # | Criterion | Pass/Fail | Notes |
|---|-----------|-----------|-------|
| 1 | BGE-small-en-v1.5 Inference API responds < 200ms per embedding call | ❌ | |
| 2 | Qdrant instance running and reachable | ❌ | |
| 3 | Collection created with correct 384-dim config + hybrid support | ❌ | |
| 4 | All chunks indexed successfully (100% indexed / total chunks) | ❌ | |
| 5 | Hybrid search returns results for 3 test queries | ❌ | |
| 6 | Indexing time benchmarked (target: < 30 min for 1200-page book, batch_size=64) | ❌ | |

### Implementation Phase 4: Reranking

- GOAL-004: Implement reranking layer with **BAAI/bge-reranker-v2-mini** (lightweight cross-encoder for CPU)

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-015 | Set up BGE-reranker-v2-mini via Hugging Face Inference API. Load via `huggingface_hub.InferenceClient`. No local model download | | |
| TASK-016 | Implement `Reranker` wrapper compatible with LangChain's document compressor interface | | |
| TASK-017 | Configure reranking: retrieve top 30 from Qdrant → rerank → keep top 5 (retrieve more since mini reranker is fast enough) | | |
| TASK-018 | Add caching layer for reranking results (LRU cache with TTL) to avoid redundant computation | | |

**Phase 4 Checklist:**

| # | Criterion | Pass/Fail | Notes |
|---|-----------|-----------|-------|
| 1 | BGE-reranker-v2-mini Inference API responds < 1s for 30 items | ❌ | |
| 2 | Reranker correctly re-orders results (top-5 differ from top-5 raw retrieval) | ❌ | |
| 3 | Reranking latency optimized: retrieve 30 → rerank → keep 5 < 2s total | ❌ | |
| 4 | Cache hit reduces latency by ≥ 50% on repeated queries | ❌ | |
| 5 | Reranker integrates seamlessly as LangChain document compressor | ❌ | |

### Implementation Phase 5: Generation with Inline Citations

- GOAL-005: Build generation pipeline with GPT-4o-mini and inline citation system

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-019 | Build `CitationFormatter` utility: assign citation IDs [1], [2], etc. to each retrieved chunk; inject into prompt context with source mapping | | |
| TASK-020 | Create generation prompt template: system instructs GPT-4o-mini to answer using only provided context, and append citation IDs inline after each factual statement like `...Fundamental Theorem of Calculus states[...](citation-1)` | | |
| TASK-021 | Implement post-processing parser: extract citation markers from LLM output → build `{answer_text, citations: [{id, chunk_text, page, section, chapter}]}` | | |
| TASK-022 | Build LangChain `Runnable` chain: retriever → reranker → prompt → LLM → output parser | | |
| TASK-023 | Implement LangGraph workflow for multi-step retrieval: decompose complex calculus questions, retrieve per-subquery, aggregate, generate | | |

**Phase 5 Checklist:**

| # | Criterion | Pass/Fail | Notes |
|---|-----------|-----------|-------|
| 1 | Citation IDs correctly assigned to retrieved chunks before LLM call | ❌ | |
| 2 | Prompt instructs LLM to use `[citation-N]` markers — verified with test query | ❌ | |
| 3 | Post-processor correctly parses LLM output into structured answer + citations list | ❌ | |
| 4 | End-to-end chain runs: query → retrieve → rerank → generate → output (measured < 10s) | ❌ | |
| 5 | LangGraph correctly decomposes a multi-step query (e.g., "Prove FTC Part 1") | ❌ | |
| 6 | Citation mapping resolves to correct page/chapter/section for clicked popup | ❌ | |

### Implementation Phase 6: RAGAS Evaluation

- GOAL-006: Set up offline evaluation pipeline with RAGAS metrics

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-024 | Create test dataset of 50+ calculus Q&A pairs (question, ground_truth_answer, expected_chunks) covering: definitions, theorems, proofs, examples, problem solutions | | |
| TASK-025 | Implement RAGAS evaluation script: run pipeline on test set → compute `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall` | | |
| TASK-026 | Add embedding-distance-based evaluation for equation accuracy (compare ground-truth LaTeX vs output LaTeX) | | |
| TASK-027 | Log evaluation results to local file + console dashboard with drift comparison | | |
| TASK-028 | Integrate evaluation as CI step (optional, post-MVP) | | |

**Phase 6 Checklist:**

| # | Criterion | Pass/Fail | Notes |
|---|-----------|-----------|-------|
| 1 | Test dataset created with ≥50 diverse Q&A pairs across the book | ❌ | |
| 2 | RAGAS script runs without errors on 5-sample subset | ❌ | |
| 3 | Faithfulness score ≥ 0.85 | ❌ | |
| 4 | Answer relevancy score ≥ 0.90 | ❌ | |
| 5 | Context precision score ≥ 0.80 | ❌ | |
| 6 | Context recall score ≥ 0.75 | ❌ | |
| 7 | Equation accuracy metric implemented (LaTeX similarity) | ❌ | |
| 8 | Evaluation results logged to file with timestamp | ❌ | |

### Implementation Phase 7: FastAPI Backend API

- GOAL-007: Build production FastAPI backend with streaming, caching, and observability

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-029 | Create FastAPI app with CORS, middleware (logging, timing), and health endpoint | | |
| TASK-030 | Implement `POST /api/query` endpoint: accepts `{query: string, top_k?: number}`, returns `{answer: string, citations: [{id, text_snippet, page, chapter, section}], latency_ms: number}` | | |
| TASK-031 | Implement `POST /api/query/stream` endpoint: SSE streaming of generated tokens with citation metadata | | |
| TASK-032 | Implement `GET /api/citation/{id}` endpoint: returns full chunk text + surrounding context for popup display | | |
| TASK-033 | Add request-level caching (semantic cache): check if similar query was answered before via embedding similarity | | |
| TASK-034 | Add conversation history support: optional `session_id` parameter, store recent context for follow-up questions | | |

**Phase 7 Checklist:**

| # | Criterion | Pass/Fail | Notes |
|---|-----------|-----------|-------|
| 1 | FastAPI app starts, health endpoint returns 200 | ❌ | |
| 2 | POST /api/query returns correct response structure with citations | ❌ | |
| 3 | POST /api/query/stream sends SSE events correctly | ❌ | |
| 4 | GET /api/citation/{id} returns full chunk with page context | ❌ | |
| 5 | Semantic cache returns cached response for highly similar query (cosine > 0.95) | ❌ | |
| 6 | Conversation history stores and retrieves last 5 exchanges per session | ❌ | |
| 7 | All endpoints handle errors gracefully (500 → structured JSON) | ❌ | |

### Implementation Phase 8: React Frontend

- GOAL-008: Build React frontend with inline citation popups and polished UX

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-035 | Scaffold React (Vite + TypeScript + Tailwind CSS) project in `frontend/` | | |
| TASK-036 | Build main chat interface: message list, input box, send button, loading indicator | | |
| TASK-037 | Implement inline citation rendering: parse `[citation-N]` markers in answer text → render as clickable superscript links styled as badges | | |
| TASK-038 | Build citation popover component: on click, fetch `/api/citation/{id}` → show small popup/card with: source page number, section title, excerpt text (scrollable) | | |
| TASK-039 | Add book-page preview: if page number available, show small PDF page thumbnail (optional — requires PDF.js or server-side page render) | | |
| TASK-040 | Implement streaming response: use EventSource/fetch with ReadableStream to stream tokens in real-time | | |
| TASK-041 | Add conversation sidebar: session history, ability to start new chat | | |
| TASK-042 | Add query suggestions: show 3-5 suggested questions on initial load (e.g., "Explain the Fundamental Theorem of Calculus", "What is a limit?") | | |

**Phase 8 Checklist:**

| # | Criterion | Pass/Fail | Notes |
|---|-----------|-----------|-------|
| 1 | React app builds without errors (`npm run build` passes) | ❌ | |
| 2 | Chat interface: messages display, input works, send triggers API call | ❌ | |
| 3 | Inline citations render as clickable superscript elements | ❌ | |
| 4 | Citation popover opens on click and displays page/section/excerpt correctly | ❌ | |
| 5 | Streaming response: tokens appear progressively in chat | ❌ | |
| 6 | Conversation sidebar shows sessions, switching works | ❌ | |
| 7 | Query suggestions displayed on empty state | ❌ | |
| 8 | Responsive: works on desktop and tablet | ❌ | |
| 9 | Loading states and error states handled gracefully | ❌ | |

### Implementation Phase 9: Advanced Features

- GOAL-009: Add production-grade enhancements — observability, security, performance

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-043 | Add LangSmith tracing for full observability of retrieval → rerank → generation pipeline | | |
| TASK-044 | Implement rate limiting and request validation (Pydantic models) | | |
| TASK-045 | Add Docker Compose setup: fastapi app + qdrant + optional redis for caching | | |
| TASK-046 | Write comprehensive tests: unit tests for chunking/citation parser, integration tests for API endpoints, eval test for RAGAS | | |
| TASK-047 | Add indexing CLI command: `rag index` — full re-index or incremental update | | |

**Phase 9 Checklist:**

| # | Criterion | Pass/Fail | Notes |
|---|-----------|-----------|-------|
| 1 | LangSmith trace captured for a sample query (full span: retrieve → rerank → generate) | ❌ | |
| 2 | Rate limiting: >10 requests/sec returns 429 | ❌ | |
| 3 | Docker Compose starts all services: app + qdrant | ❌ | |
| 4 | Unit tests pass (≥ 90% coverage on core modules) | ❌ | |
| 5 | Integration tests pass for all API endpoints | ❌ | |
| 6 | CLI `rag index` command re-indexes successfully | ❌ | |

## 3. Alternatives

- **ALT-001 (Marker vs PyMuPDF)**: PyMuPDF is faster but loses equation structure. Marker is slower but produces clean LaTeX, essential for calculus. Chosen: **Marker**.
- **ALT-002 (BGE-reranker-v2-m3 vs BGE-reranker-v2-mini)**: Full v2-m3 (2.7B params) is heavier on Inference API quota. Mini is cheaper and faster. Chosen: **BGE-reranker-v2-mini**.
- **ALT-003 (BGE-large-en-v1.5 vs BGE-small-en-v1.5 vs all-MiniLM)**: BGE-large (1.5B, 1024d) costs more API credits per call. all-MiniLM-L6-v2 (80M, 384d) is cheaper but lower quality. BGE-small-en-v1.5 (200M, 384d) is the best quality/cost compromise via Inference API. Chosen: **BGE-small-en-v1.5**.
- **ALT-003 (Streamlit vs FastAPI+React)**: Streamlit is faster to prototype but limited for custom citation popups. Chosen: **FastAPI+React** for production quality.
- **ALT-004 (Qdrant vs Chroma vs Pinecone)**: Chroma lacks hybrid search (critical for math). Pinecone is managed (cost). Qdrant offers free local hybrid search. Chosen: **Qdrant**.
- **ALT-005 (OpenAI vs local LLM)**: Local LLMs like Llama 3.1 8B are free but require GPU for acceptable speed. On CPU, 8B models run at <1 token/sec — unusable. GPT-4o-mini is cheap (~$0.15/M tokens) and far more accurate. Chosen: **GPT-4o-mini**.

## 4. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | ≥0.3 | Core framework |
| `langchain-community` | ≥0.3 | Community integrations |
| `langchain-qdrant` | ≥0.2 | Qdrant vector store integration |
| `langchain-huggingface` | ≥0.1 | HuggingFace embeddings via LangChain |
| `langchain-openai` | ≥0.3 | OpenAI/GPT-4o-mini integration |
| `huggingface_hub` | ≥0.27 | Hugging Face Inference API client (no local model download) |
| `marker-pdf` | latest | PDF → Markdown with LaTeX preservation |
| `pypdf` | ≥4.0 | Fallback PDF text extraction |
| `qdrant-client` | ≥1.12 | Qdrant vector DB client |
| `fastapi` | ≥0.115 | REST API framework |
| `uvicorn[standard]` | ≥0.34 | ASGI server |
| `ragas` | ≥0.2 | RAG evaluation metrics |
| `langgraph` | ≥0.3 | Multi-step retrieval workflows |
| `langsmith` | ≥0.3 | Tracing & observability |
| `semantic-text-splitter` | ≥0.2 | Better text chunking |
| `react` | ≥18 | Frontend UI |
| `tailwindcss` | ≥3 | Frontend styling |
| `vite` | ≥5 | Frontend build tool |

## 5. Files

- **FILE-001**: `pyproject.toml` — Python dependencies and project config
- **FILE-002**: `app/__init__.py` — App package init
- **FILE-003**: `app/config.py` — Centralized configuration (Pydantic Settings)
- **FILE-004**: `app/ingestion/pdf_loader.py` — Marker-based PDF → Markdown loader
- **FILE-005**: `app/ingestion/chunker.py` — Hierarchical chunking pipeline
- **FILE-006**: `app/ingestion/indexer.py` — Embed + index to Qdrant
- **FILE-007**: `app/retrieval/embeddings.py` — BGE-large-en-v1.5 embedding setup
- **FILE-008**: `app/retrieval/vector_store.py` — Qdrant hybrid search client
- **FILE-009**: `app/retrieval/reranker.py` — BGE-reranker-v2-m3 wrapper
- **FILE-010**: `app/retrieval/retriever.py` — Composed retriever (hybrid search + rerank)
- **FILE-011**: `app/generation/prompts.py` — Prompt templates (system, human)
- **FILE-012**: `app/generation/chain.py` — LCEL chain + LangGraph workflow
- **FILE-013**: `app/generation/citation_formatter.py` — Citation ID assignment + post-processing parser
- **FILE-014**: `app/generation/schemas.py` — Pydantic models for request/response
- **FILE-015**: `app/api/main.py` — FastAPI app instance
- **FILE-016**: `app/api/routes.py` — API endpoint definitions
- **FILE-017**: `app/api/dependencies.py` — DI for chain, retriever, etc.
- **FILE-018**: `app/evaluation/dataset.py` — Test dataset builder
- **FILE-019**: `app/evaluation/runner.py` — RAGAS evaluation runner
- **FILE-020**: `frontend/src/App.tsx` — Main React app
- **FILE-021**: `frontend/src/components/ChatInterface.tsx` — Chat UI
- **FILE-022**: `frontend/src/components/CitationPopover.tsx` — Citation popup component
- **FILE-023**: `frontend/src/components/MessageList.tsx` — Message display with inline citations
- **FILE-024**: `frontend/src/api/client.ts` — API client
- **FILE-025**: `data/` — Raw PDF + extracted Markdown + Qdrant storage
- **FILE-026**: `docker-compose.yml` — Container orchestration
- **FILE-027**: `Dockerfile` — API server container

## 6. Testing

- **TEST-001**: Unit test for chunker — verify hierarchy preservation, no broken LaTeX
- **TEST-002**: Unit test for citation parser — verify LLM output → structured citation extraction
- **TEST-003**: Integration test for `POST /api/query` — full pipeline smoke test
- **TEST-004**: Integration test for streaming endpoint — verify SSE format
- **TEST-005**: RAGAS evaluation — faithfulness ≥0.85, context precision ≥0.80, answer relevancy ≥0.90
- **TEST-006**: Performance test — P95 latency < 5s for end-to-end query
- **TEST-007**: Citation accuracy test — spot-check 20 queries that citations actually support the claim

## 7. Risks & Assumptions

- **RISK-001**: Marker may fail on certain complex multi-column pages or unusual equation layouts → fallback to PyMuPDF raw text extraction
- **RISK-002**: BGE-reranker-v2-mini via Inference API adds network latency (~200-500ms) per rerank → acceptable
- **RISK-003**: BGE-small-en-v1.5 via Inference API processes embeddings server-side → no CPU load, but depends on API rate limits and network
- **RISK-004**: GPT-4o-mini may hallucinate in citations (attribute wrong section) → mitigation: strict system prompt + manual eval checks
- **RISK-005**: 1200+ page PDF → large number of chunks (~10k-20k) → Qdrant may need memory tuning
- **ASSUMPTION-001**: Book PDF is text-layer PDF (not scanned) — if scanned, need OCR layer (Marker handles this partially)
- **ASSUMPTION-002**: User has at least 8GB RAM (CPU-only, no GPU required). No local model storage needed — all HF models via Inference API
- **ASSUMPTION-003**: Valid `HF_TOKEN` with Inference API access for BAAI/bge-small-en-v1.5 and BAAI/bge-reranker-v2-mini
- **ASSUMPTION-003**: OpenAI API key available for GPT-4o-mini

## 8. Related Specifications / Further Reading

- [Marker PDF → Markdown](https://github.com/VikParuchuri/marker)
- [BGE-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [BGE-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Qdrant Hybrid Search](https://qdrant.tech/documentation/concepts/hybrid-queries/)
- [LangChain Expression Language](https://python.langchain.com/docs/expression_language/)
- [RAGAS Evaluation](https://docs.ragas.io/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
