# Rag_Book — Intelligent Math Textbook Q&A

A **Retrieval-Augmented Generation (RAG)** pipeline for querying *Thomas' Calculus Early Transcendentals, 14th Edition* (1,262 pages). Ask calculus questions in natural language and get concise answers with citations, rendered LaTeX formulas, and context-aware responses.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   FRONTEND                       │
│     React 19 + Vite + Tailwind v4 + KaTeX       │
│              (localhost:5173)                    │
└──────────────┬──────────────────────────────────┘
               │ /api/query, /api/query/stream
               ▼
┌──────────────────────────────────────────────────┐
│                  BACKEND                          │
│     FastAPI + Uvicorn (localhost:8000)            │
│                                                    │
│  ┌─────────┐  ┌──────┐  ┌─────────┐  ┌────────┐ │
│  │Ingestion│  │Query │  │Generation│  │Citation│ │
│  │ Pipeline│→ │Pipeline│→│   LLM    │  │Verify  │ │
│  └────┬────┘  └──┬───┘  └────┬────┘  └────────┘ │
│       │          │           │                    │
│       ▼          ▼           ▼                    │
│  ┌──────────────────────────────────────┐        │
│  │        VECTOR DATABASE (Qdrant)       │        │
│  │  Dense: bge-small-en-v1.5 (384-dim)  │        │
│  │  Sparse: BM25 (k1=1.5, b=0.75)       │        │
│  │  1,614 chunks from 1,262 PDF pages   │        │
│  └──────────────────────────────────────┘        │
│                                                    │
│  ┌──────────────────────────────────────┐        │
│  │          LLM ROUTER (9 models)        │        │
│  │  Primary: claude-sonnet-4.5           │        │
│  │  Fallback: deepseek-3.2, gpt-5.5     │        │
│  └──────────────────────────────────────┘        │
└──────────────────────────────────────────────────┘
```

## Pipeline Flow

```
User Query
    │
    ├─→ Query Type Detection (equation / theorem / definition / general)
    ├─→ Query Expansion (synonyms + math notation mapping)
    ├─→ HyDE (auto-activates for equation-heavy queries)
    ├─→ Hybrid Search (dense + BM25 + chapter filter)
    ├─→ Adaptive Relevance Gate (0.05–0.15 by query type)
    ├─→ Lightweight Reranker (optional, default off)
    ├─→ Contextual Compression (optional, skip <2K chars)
    ├─→ Response Cache (LRU, 100 entries)
    ├─→ LLM Generation (3× retry cascade)
    ├─→ Multi-Metric Citation Verification
    └─→ LaTeX Sanitizer → Response
```

## Features

### 🔍 Hybrid Retrieval
- **Dense search**: `BAAI/bge-small-en-v1.5` (384-dim, 512-token context)
- **Sparse search**: Custom BM25 (k1=1.5, b=0.75) fitted on 1,614 chunks
- **Chapter filtering**: automatically restricts search scope when query mentions a chapter/section
- **Adaptive relevance gate**: thresholds adjust per query type (0.05 equations, 0.10 definitions, 0.15 general)

### 🧠 Intelligent Query Understanding
- **Query type detection**: classifies queries as equation/theorem/definition/general
- **Query expansion**: math synonym map (e.g., "divergence theorem" → "gauss theorem")
- **HyDE auto-activation**: automatically generates hypothetical documents for equation-heavy queries

### 🤖 Reliable LLM Generation
- **9-model router**: primary → fallback → legacy cascade with 3× retry
- **System prompt**: enforces proper LaTeX formatting with explicit rules against truncated commands
- **Response caching**: LRU cache (100 entries) for identical queries
- **Streaming**: SSE-based token streaming with real-time LaTeX sanitization

### ✅ Citation Verification
- **Multi-metric scoring**: token overlap + entity overlap + key terms
- **Confidence scores**: per-citation composite score
- **Graceful degradation**: removes uncited claims instead of regenerating

### 📐 LaTeX Rendering
- **KaTeX rendering**: fast math rendering via `react-katex`
- **Comprehensive sanitizer**: fixes `\f`→`\frac`, Unicode dots→`\cdot`, missing braces, double backslashes, and 20+ other common LLM LaTeX errors
- **Math error boundary**: gracefully falls back to raw LaTeX on parse failures
- **Tailwind v4 compatible**: CSS overrides protect fraction bars from border resets

### 📊 Evaluation & Monitoring
- **6-phase evaluation**: retrieval, generation, E2E, latency, robustness, cost
- **Standalone chunk evaluation**: size distribution, metadata completeness, LaTeX integrity
- **Live metrics endpoint**: query latency, chunk count, system health

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 19, TypeScript 6, Vite 8, Tailwind CSS v4 |
| **Math Rendering** | KaTeX 0.17, react-katex |
| **Backend** | Python 3.12+, FastAPI, Uvicorn |
| **LLM Router** | 9-model local router at `localhost:20128/v1` |
| **Primary LLM** | claude-sonnet-4.5 |
| **Fallback LLM** | deepseek-3.2, gpt-5.5 |
| **Vector DB** | Qdrant (local, `/tmp/qdrant_calculus_db`) |
| **Dense Embeddings** | `BAAI/bge-small-en-v1.5` (sentence-transformers) |
| **Sparse Embeddings** | Custom BM25 (scikit-learn) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (optional) |
| **PDF Processing** | marker-pdf, pypdf |
| **Frameworks** | LangChain, LangGraph |
| **Evaluation** | RAGAS |
| **Ingestion** | 1,262 PDF pages → 1,614 chunks |

## Getting Started

### Prerequisites
- Python ≥ 3.11
- Node.js ≥ 20
- Local LLM router at `localhost:20128/v1` (or configure in `.env`)

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd rag_book

# Backend
python -m venv .venv
source .venv/bin/activate
cd backend
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

### Configuration

Create `backend/.env`:
```env
HF_TOKEN_1=your_huggingface_token
HF_TOKEN_2=your_fallback_token
OPENAI_API_KEY=your_openai_key
```

### Run the App

```bash
# Terminal 1 — Backend (from backend/)
source ../.venv/bin/activate
python3 -m uvicorn app.api.endpoints:app --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend (from frontend/)
npm run dev
```

Open **http://localhost:5173** in your browser.

### Ingest the Textbook

```bash
cd backend
source ../.venv/bin/activate
python scripts/ingest.py --pdf path/to/thomas_calculus.pdf
```

Or run the full indexing pipeline:
```bash
python scripts/index_full.py
```

## Project Structure

```
rag_book/
├── backend/
│   ├── app/
│   │   ├── api/           # FastAPI routes & schemas
│   │   ├── generation/    # LLM prompt building & response handling
│   │   ├── ingestion/     # PDF parsing, chunking, cleaning, indexing
│   │   ├── retrieval/     # Embeddings, vector store, reranking, compression
│   │   ├── config.py      # Pydantic settings (env-based)
│   │   └── evaluation/    # Metrics & evaluation utilities
│   ├── scripts/           # Ingestion, evaluation, debug scripts
│   └── tests/             # Pytest suite (21 tests)
├── frontend/
│   ├── src/
│   │   ├── components/    # UI components (chat, widgets, citations)
│   │   ├── hooks/         # useChat, SSE streaming
│   │   ├── utils/         # LaTeX sanitizer, rendering utilities
│   │   └── types/         # TypeScript interfaces
│   └── package.json
├── AGENTS.md              # Agent memory / project documentation
└── README.md
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Non-streaming RAG query with citations |
| `POST` | `/query/stream` | SSE-streamed query response |
| `POST` | `/ingest` | Trigger PDF ingestion pipeline |
| `GET` | `/health` | Server health check |
| `GET` | `/metrics` | Live chunk count & system stats |

### Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the chain rule?", "top_k": 5, "rerank": false}'
```

## Evaluation

```bash
cd backend && source ../.venv/bin/activate

# Full 6-phase evaluation
python scripts/eval_full.py

# Chunk quality evaluation only
python scripts/evaluate_chunks.py

# Run tests
python -m pytest -v
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **No external API** | All LLM calls go through local 9-router at `localhost:20128/v1` |
| **Math cleaning at ingestion** | PDF garbled Unicode cleaned once, not per-query (saved ~6s latency) |
| **Reranker disabled by default** | Negative quality gain (−0.05) for 4.2s latency cost |
| **Compression skip threshold** | 2,000 chars — short contexts don't need compression |
| **Separate katex-fix.css** | Loaded last to prevent Tailwind v4 border reset from hiding fraction bars |
| **Streaming avoids sanitizer** | Raw tokens streamed; full sanitization deferred to frontend |
| **Graceful degradation** | Unverified claims removed from answer instead of full regeneration |
