# RAG Pipeline Optimization Plan

## Executive Summary

**Current state**: Retrieval accuracy is 100% (keyword-match), but generation quality has issues with theorem boundary splits, noisy chunks, math notation garbling, and unicode artifacts. Overall answer accuracy is below 95%.

**Root causes identified**:
1. ★ **BM25 not persisted** → query sparse vectors empty (FIXED)
2. ★ **512-char chunks too small** → theorems split across chunks
3. ★ **Character-based (not token-based) chunking** → inefficient context usage
4. ★ **Math Unicode artifacts** (`/uni2032`, `/uni2260.alt10`) → garbled retrieval
5. ★ **No semantic boundary detection** → logical content split mid-theorem
6. ★ **Rerank dedup uses page_content as key** → ordering lost, collisions
7. ★ **18 tiny/noise chunks (< 50 chars)** → pollute results
8. ★ **No query expansion** → missed matches on math notation variants
9. ★ **LaTeX normalization incomplete** → `\[` vs `$$` inconsistency

---

## Phase 1: Data Quality & Chunking Overhaul ⚡

**Goal**: Clean, properly-sized, semantically-coherent chunks with correct math notation.

### Checklist
- [ ] **1.1 Fix Unicode artifacts in parsed_docs.jsonl**
  - Replace `/uni2032` → `'` (prime), `/uni2260.alt10` → `≠`, `/uni2260.alt1a` → `≠`, `/uni2033` → `''`, `/uni2212.boldH` → `−`, `/uni00A0` → space
  - Add comprehensive Unicode cleanup in `cleaner.py`
  - Re-run all regex replacements on existing chunks

- [ ] **1.2 Increase chunk_size from 512 to 1024 tokens (not characters)**
  - Switch from `len()` to token-count via `tiktoken` or model tokenizer
  - Chunk size: 1024 tokens, overlap: 128 tokens
  - This ensures each chunk can hold a complete theorem + explanation (~1/2 page)

- [ ] **1.3 Add semantic boundary detection**
  - Split on `## Section` headers as now (good)
  - Add split on `THEOREM`, `Definition`, `EXAMPLE` markers
  - Keep theorem/example with its content (don't split between title and body)

- [ ] **1.4 Filter/fix tiny chunks**
  - Remove chunks < 100 chars that are noise (TOC entries, page numbers)
  - Merge very short chunks (< 200 chars) with adjacent chunk
  - Preserve only meaningful short content (formulas, key equations)

- [ ] **1.5 Normalize LaTeX consistently**
  - All `\[` → `$$`, `\]` → `$$`, `\(` → `$`, `\)` → `$`
  - Fix broken LaTeX brackets and unbalanced delimiters

- [ ] **1.6 Verify with sample**
  - After cleanup, manually inspect 20 random chunks from different chapters
  - Confirm theorem boundaries, math notation, no garbled text

---

## Phase 2: Retrieval Optimization 🎯

**Goal**: Maximize relevant chunk recall, improve ranking, fix dedup issues.

### Checklist
- [ ] **2.1 Fix reranker dedup logic** (`endpoints.py:70-73`)
  - Current: `kept = {t for t, _ in reranked}` — uses page_content as dict key → collision risk + loses ordering
  - Fix: preserve original order and filter by index instead

- [ ] **2.2 Add MMR (Maximum Marginal Relevance) for diversity**
  - After reranking top-5, apply MMR to ensure diverse chunks (not 5 similar chunks from same page)
  - Lambda=0.7 for balance between relevance and diversity

- [ ] **2.3 Implement query expansion**
  - Pre-process queries: expand math notation synonyms (`L'Hôpital` ↔ `l'hopital`, `L'Hospital`)
  - Add common alternative phrasings for key calculus concepts
  - Use LLM for query rewriting (optional, Phase 2 enhancement)

- [ ] **2.4 Tune retrieval parameters**
  - Optimize `top_k_retrieve`: test 20/30/40 → measure recall
  - Optimize `top_k_rerank`: test 3/5/7 → measure precision@k
  - Consider adding score threshold filter (min relevance score)

- [ ] **2.5 Evaluate hybrid search balance**
  - Currently dense + BM25 with default weights
  - Test α=0.5 (equal), α=0.7 (dense-heavy), α=0.3 (sparse-heavy)
  - Math content benefits from keyword (sparse) search

---

## Phase 3: Generation Enhancement 🤖

**Goal**: Produce accurate, well-cited, structured answers with no hallucination.

### Checklist
- [ ] **3.1 Strengthen system prompt**
  - Add explicit citation format: `[N]` after each claim, not just at end
  - Instruction: "If context is insufficient, say exactly what is missing — do not guess"
  - Structured output: separate "Answer", "Key Points", "Formula" sections
  - Emphasize: "Cite the exact source [N] for every formula and theorem"

- [ ] **3.2 Add answer validation**
  - Check that all cited chunks actually contain the cited claims
  - Flag un-cited claims for review
  - Simple regex: verify `[N]` maps to a real chunk index

- [ ] **3.3 Implement self-consistency check**
  - Generate 3 answers at temperature=0.3
  - Compare for factual consistency
  - Return majority-consistent answer

- [ ] **3.4 Add context-aware truncation**
  - Dynamic context size based on available tokens
  - Prioritize reranker-scored chunks within token budget
  - Ensure full theorem statements aren't truncated

---

## Phase 4: Infrastructure & Monitoring 🏗️

**Goal**: Persistent, observable, maintainable pipeline.

### Checklist
- [ ] **4.1 Finalize persistent Qdrant storage** (DONE — Phase 0)
  - [x] Switched from pickle to `QdrantClient(path=...)`
  - [x] BM25 state persisted to `/tmp/qdrant_calculus_bm25.pkl`
  - [ ] Add BM25 auto-reload on vector store init (have `from_pickle` loading)

- [ ] **4.2 Add retrieval + generation metrics endpoint**
  - `/metrics` endpoint exposing:
    - avg retrieval score per query
    - citation count per answer
    - number of chunks used
    - generation latency
    - rate limit hits (for 9-router)

- [ ] **4.3 Add unit tests**
  - Test chunking doesn't split theorems
  - Test BM25 query produces non-empty sparse vectors
  - Test reranker preserves ordering
  - Test LaTeX normalization is consistent
  - Test query expansion finds math synonyms

- [ ] **4.4 Logging & observability**
  - Log query → retrieved chunk IDs → reranked scores → generated answer
  - Store in structured format for RAGAS re-evaluation
  - Track accuracy drift over time

---

## Phase 5: Advanced Optimizations 🚀

**Goal**: Push accuracy beyond 95% with advanced techniques.

### Checklist
- [ ] **5.1 Upgrade embedding model**
  - Current: `all-MiniLM-L6-v2` (384-dim, 22M params)
  - Option A: `intfloat/e5-small-v2` (384-dim, better math)
  - Option B: `BAAI/bge-small-en-v1.5` (384-dim, general purpose)
  - Option C: `sentence-transformers/all-mpnet-base-v2` (768-dim, ~2x better but slower)
  - Re-index all chunks with new model

- [ ] **5.2 Implement HyDE (Hypothetical Document Embeddings)**
  - Generate a "hypothetical ideal chunk" from the query using LLM
  - Use that for retrieval instead of raw query
  - Particularly effective for math/technical queries

- [ ] **5.3 Add chunk-level metadata enrichment**
  - Extract and store: theorem numbers, equation counts, key terms
  - Enable filtering by theorem/equation type
  - Add section hierarchy: Chapter > Section > Subsection > Theorem

- [ ] **5.4 Implement contextual compression**
  - Use LLM to extract only the relevant sentences from retrieved chunks
  - Reduces noise, increases precision
  - Important: verify compressions don't lose mathematical content

- [ ] **5.5 Add structured citation verification**
  - Cross-reference each `[N]` in answer against actual chunk content
  - Verify the formula/claim actually exists in the cited chunk
  - Report verification score with each answer

---

## Verification & Rollback Plan

After each phase:
1. Run `scripts/diagnose_pipeline.py` — verify retrieval accuracy
2. Run `scripts/evaluate_deep.py` — verify answer accuracy
3. Compare against baseline metrics
4. If any metric drops: rollback the phase changes and investigate

### Current Baseline (after critical fix):
```
Retrieval accuracy (keyword): 100%
Queries with citations:       20/20
Avg answer length:            825 chars
```

### Target (after all phases):
```
Retrieval accuracy (keyword):  100%
Answer accuracy (deep eval):   > 95%
Math notation correctness:     100%
Theorem boundary integrity:    100%
Zero hallucination rate:       100%
```
