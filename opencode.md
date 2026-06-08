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
| `backend/app/generation/generator.py` | `Generator` — Multi-model (Claude/DeepSeek/GPT) with rate limiting, validation, self-consistency |
| `backend/app/ingestion/chunker.py` | **OPTIMIZED** — 1024-token chunks with 128 overlap, theorem boundary detection, noise filtering |
| `backend/app/ingestion/cleaner.py` | **OPTIMIZED** — Unicode math fixes, LaTeX normalization, footer removal |
| `backend/app/ingestion/indexer.py` | Batch embed + PointStruct upsert to Qdrant (persistent mode) |
| `backend/app/api/endpoints.py` | FastAPI: `/query`, `/query/stream` (SSE), `/ingest`, `/health` |
| `backend/app/api/schemas.py` | Pydantic models |
| `backend/scripts/ingest.py` | CLI ingestion pipeline |
| `backend/scripts/evaluate.py` | RAGAS evaluation |
| `backend/scripts/index_full.py` | Full-book index runner |
| `backend/scripts/migrate_chunks.py` | **NEW** — Re-chunks with optimized settings and re-indexes Qdrant + BM25 |
| `backend/scripts/diagnose_pipeline.py` | **NEW** — Comprehensive pipeline diagnostics (20 test queries) |
| `backend/scripts/final_eval.py` | **NEW** — Deep evaluation with rate-limited Claude Sonnet 4.5 |
| `backend/tests/test_chunking.py` | **NEW** — 16 unit tests for chunking, BM25, query expansion |
| `backend/app/retrieval/expansion.py` | **NEW** — Query expansion with math synonyms |
| `backend/app/retrieval/hyde.py` | **NEW** — Hypothetical Document Embeddings generation |
| `backend/app/retrieval/compression.py` | **NEW** — Contextual compression of retrieved chunks |
| `backend/run_api.py` | Uvicorn entry point |
| `frontend/` | Vite + React + Tailwind chat UI with citations |
| `frontend/vite.config.ts` | Vite config — proxies `/api` → backend with path rewrite (strips `/api` prefix) |
| `backend/app/api/schemas.py` | Pydantic models |

## Data
- Chunked text: `backend/data/markdown/parsed_docs.jsonl` (1,316 chunks — reduced from 8,082 via optimized chunking)
- Qdrant storage: `/tmp/qdrant_calculus_db` (persistent local QdrantClient)
- BM25 state: `/tmp/qdrant_calculus_bm25.pkl` (pickled vocab + IDF stats for sparse retrieval)

## API & Models
- **Primary LLM**: `kr/claude-sonnet-4.5` via 9-router at `http://localhost:20128/v1`
- **Fallback LLM**: `kr/deepseek-3.2` (auto-retry on rate limit)
- **Embedding**: `all-MiniLM-L6-v2` (384-dim) — local sentence-transformers
- **Sparse**: Custom BM25 (local, with persistent state in `/tmp/qdrant_calculus_bm25.pkl`)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` — local transformers with MMR diversity
- **HF tokens**: Both in `backend/.env` (`HF_TOKEN_1`, `HF_TOKEN_2`), fallback logic in `settings.get_hf_token()`
- **Rate limiting**: 10-second delay between LLM requests to avoid 429 errors

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
- **BM25 persistence**: BM25 vocab/IDF stats are now saved to `/tmp/qdrant_calculus_bm25.pkl` — auto-loaded on vector store init.
- **Rerank dedup**: Fixed collision bug — now uses index-based tracking instead of content-key dedup to preserve ordering.
- **Unicode math artifacts**: Cleaner now handles `/uni2032` (prime), `/uni2260.alt10` (not equal), `/uni2212.boldH` (minus), etc.

## Next Steps (optional)

1. **Upgrade embedding model** — change `embedding_model` in `.env` to `sentence-transformers/all-mpnet-base-v2` and re-run `scripts/migrate_chunks.py`
2. **Enable HyDE/self-consistency** — set `USE_HYDE=true` or `USE_SELF_CONSISTENCY=true` in `.env` for higher quality (slower) answers
3. **Tune retrieval parameters** — adjust `top_k_retrieve`, `top_k_rerank`, `mmr_lambda` in `app/config.py`
4. **Add frontend integration** — the `/query/stream` endpoint supports SSE streaming for real-time answers
5. **Run RAGAS re-evaluation** — `scripts/evaluate.py` to get updated faithfulness/precision scores

## Status (all complete)
- Phase 1: PDF extraction ✅
- Phase 2: Chunking (1,316 optimized chunks — reduced from 8,082) ✅
- Phase 3: Qdrant index (persistent local storage + BM25 state) ✅
- Phase 4: Reranking with MMR diversity ✅
- Phase 5: Generation with citations (Claude Sonnet 4.5 + validation) ✅
- Phase 6: RAGAS evaluation ✅
- Phase 7: FastAPI backend with `/metrics` endpoint ✅
- Phase 8: React frontend ✅
- Phase 9: Streaming + multi-turn ✅

## Optimization Phases (completed)

### Phase 0: Infrastructure Fixes
- Persistent Qdrant storage at `/tmp/qdrant_calculus_db`
- BM25 state persisted to `/tmp/qdrant_calculus_bm25.pkl`
- Switched to `kr/claude-sonnet-4.5` (better for math than rate-limited `cx/gpt-5.5`)
- Rate limiting: 10-second delays between LLM requests
- Multi-model fallback: Claude → DeepSeek → GPT-5.5

### Phase 1: Data Quality & Chunking
- **Unicode math cleanup** — fixed `/uni2032`→`'`, `/uni2260.alt10`→`≠`, `/uni2212.boldH`→`−`
- **Token-based chunking** — switched from character count to estimated tokens (1024 tokens, 128 overlap)
- **Theorem boundary detection** — splits at `THEOREM`, `Definition`, `EXAMPLE` markers
- **Noise filtering** — removed chunks < 50 chars (TOC entries, page numbers)
- **Chunk reduction** — 8,082 → 1,316 chunks (6x fewer, much larger and more coherent)
- **Migration script** — `scripts/migrate_chunks.py` re-chunks and re-indexes

### Phase 2: Retrieval Optimization
- **MMR diversity** — added `Reranker.rerank_with_mmr()` with λ=0.7
- **Query expansion** — `retrieval/expansion.py` adds math synonyms (L'Hôpital ↔ l'Hopital, derivative ↔ differentiation, etc.)
- **Index-based rerank dedup** — fixed collision bug where identical content lost ordering

### Phase 3: Generation Enhancement
- **Multi-model fallback** — tries `kr/claude-sonnet-4.5` → `kr/deepseek-3.2` → `cx/gpt-5.5`
- **Strengthened system prompt** — structured output (Answer/Key Points/Formula), citation format enforced
- **Answer validation** — checks that cited chunks actually contain claimed content, re-prompts if mismatch
- **Self-consistency mode** — generates 3 samples at temp=0.3, picks most consistent

### Phase 4: Infrastructure & Testing
- **16 unit tests** — chunking, BM25, query expansion, LaTeX normalization, theorem boundaries
- **`/metrics` endpoint** — exposes config (model, top_k, delays, features)
- **Logging** — tracks validation failures, reranker issues, compression attempts

### Phase 5: Advanced Features
- **HyDE (Hypothetical Document Embeddings)** — generates ideal chunk from query, uses it for retrieval
- **Contextual compression** — extracts relevant sentences from chunks via LLM
- **Embedding model ready to upgrade** — can switch to `sentence-transformers/all-mpnet-base-v2` (768-dim)

## Final Evaluation Results
- **Effective accuracy: 100%** (all 8 test queries answered correctly)
- **Strict accuracy: 87.5%** (7/8 fully precise)
- **0 wrong answers**
- One partial answer (L'Hôpital's rule) — correctly describes when/how to use it but lacks exact formal conditions (split across PDF pages)

## Usage

```bash
# Start backend
cd backend && source ../.venv/bin/activate && python run_api.py

# Query with optimizations enabled (MMR, compression)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the derivative of sin(x)?", "use_mmr": true}'

# Check metrics
curl http://localhost:8000/metrics

# Re-chunk with optimized settings
cd backend && python scripts/migrate_chunks.py
```
