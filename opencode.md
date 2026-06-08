# Rag_Book - Agent Memory

## Project
RAG pipeline for "Thomas' Calculus Early Transcendentals, 14th Edition" (1262-page PDF). Hybrid search (dense + BM25 sparse) with reranking, LLM generation with citations, FastAPI backend, React frontend.

## Key Files

| File | Purpose |
|------|---------|
| `backend/app/config.py` | Settings (env: HF_TOKEN_1, HF_TOKEN_2, OPENAI_API_KEY) |
| `backend/app/retrieval/embeddings.py` | `HFInferenceAPIEmbeddings` — local sentence-transformers (all-MiniLM-L6-v2) |
| `backend/app/retrieval/vector_store.py` | `BM25SparseEmbeddings` + `HybridVectorStore` (Qdrant wrapper) |
| `backend/app/retrieval/reranker.py` | Reranker with `cross-encoder/ms-marco-MiniLM-L-6-v2` (local transformers) |
| `backend/app/generation/generator.py` | `Generator` — GPT-5.5 via 9-router with citation extraction + streaming |
| `backend/app/ingestion/chunker.py` | 512-token chunks with 50 overlap, section/chapter metadata |
| `backend/app/ingestion/indexer.py` | Batch embed + PointStruct upsert to Qdrant |
| `backend/app/api/endpoints.py` | FastAPI: `/query`, `/query/stream` (SSE), `/ingest`, `/health` |
| `backend/app/api/schemas.py` | Pydantic models |
| `backend/scripts/ingest.py` | CLI ingestion pipeline |
| `backend/scripts/evaluate.py` | RAGAS evaluation |
| `backend/scripts/index_full.py` | Full-book index runner |
| `backend/run_api.py` | Uvicorn entry point |
| `frontend/` | Vite + React + Tailwind chat UI with citations |
| `frontend/vite.config.ts` | Vite config — proxies `/api` → backend with path rewrite (strips `/api` prefix) |
| `backend/app/api/schemas.py` | Pydantic models |

## Data
- Chunked text: `backend/data/markdown/parsed_docs.jsonl` (8,082 chunks)
- Qdrant cache: `/tmp/qdrant_calculus.pkl` (pickled QdrantClient in-memory)

## API & Models
- **LLM**: `cx/gpt-5.5` via 9-router at `http://localhost:20128/v1`
- **Embedding**: `all-MiniLM-L6-v2` (384-dim) — local sentence-transformers
- **Sparse**: Custom BM25 (local)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` — local transformers
- **HF tokens**: Both in `backend/.env` (`HF_TOKEN_1`, `HF_TOKEN_2`), fallback logic in `settings.get_hf_token()`

## RAGAS Results (latest)
- Faithfulness: 0.8889
- Context Precision: 0.7289
- Context Recall: 1.0000
- Answer Relevancy: N/A (9-router doesn't support openai provider for RAGAS sub-jobs)

## Running
- Backend: `source .venv/bin/activate && cd backend && python3 -m uvicorn app.api.endpoints:app --host 0.0.0.0 --port 8000`
- Frontend: `cd frontend && npm run dev` (proxies `/api` → backend on port 8000)

## Directory structure
- All backend code under `backend/` (app/, scripts/, data/, tests/, run_api.py, .env)
- Frontend under `frontend/`
- Shared venv at root `.venv/`

## Critical Fixes
- **Vite proxy**: Frontend sends to `/api/query`, backend serves `/query`. The proxy in `vite.config.ts` uses `path.replace(/^\/api/, "")` to strip the prefix.
- **Server CWD**: Must run `uvicorn` from `backend/` directory, not root. Otherwise `from app.api.endpoints import app` fails because `app/` lives inside `backend/`.

## Status (all complete)
- Phase 1: PDF extraction ✅
- Phase 2: Chunking (8082 chunks) ✅
- Phase 3: Qdrant index (hybrid dense + BM25) ✅
- Phase 4: Reranking ✅
- Phase 5: Generation with citations ✅
- Phase 6: RAGAS evaluation ✅
- Phase 7: FastAPI backend ✅
- Phase 8: React frontend ✅
- Phase 9: Streaming + multi-turn ✅
