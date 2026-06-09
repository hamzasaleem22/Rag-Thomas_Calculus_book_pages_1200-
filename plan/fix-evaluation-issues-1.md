---
goal: Fix all critical issues identified in the RAG evaluation — OOD handling, confidence calibration, hallucination, LaTeX fidelity, and latency bottlenecks
version: 1.0
date_created: 2026-06-09
owner: RAG Book Team
status: 'Completed'
tags: ['performance', 'quality', 'robustness', 'latency', 'fix']
---

# Introduction

![Status: Completed](https://img.shields.io/badge/status-Completed-brightgreen)

Comprehensive plan to fix all 5 categories of issues identified during the 6-phase evaluation of the RAG pipeline. Each issue is traced to its root cause(s) with a specific, measurable fix. After all fixes, rerun the evaluation to verify improvements.

## 1. Requirements & Constraints

- **REQ-001**: OOD queries must be refused gracefully (score > 0.80, currently 0.000)
- **REQ-002**: Confidence calibration must detect uncertainty and say "I don't know" (score > 0.80, currently 0.000)
- **REQ-003**: ch16 divergence theorem hallucination must drop to < 0.30 (currently 0.80)
- **REQ-004**: Multi-hop hallucination must drop to < 0.30 (currently 0.60-0.80)
- **REQ-005**: LaTeX fidelity must improve to > 0.90 (currently 0.764)
- **REQ-006**: P50 total latency must drop to < 15s (currently ~30s)
- **CON-001**: All fixes must use local resources only (no new API dependencies)
- **CON-002**: Must not break existing streaming or non-streaming API endpoints
- **CON-003**: Must not change the RAG schema of responses (citation format, response model)
- **CON-004**: Rate limits must remain in place to avoid 9-router throttling
- **PAT-001**: Follow existing code style (no comments, explicit error handling, rate limiting)

## 2. Implementation Steps

### Implementation Phase 1: OOD Handling & Confidence Calibration

- GOAL-001: Add OOD/refusal instructions to the system prompt and add a pre-generation relevance check so the system stops fabricating answers for out-of-scope or low-confidence queries.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Add OOD/refusal instructions to system prompt in `generator.py:_build_messages` (lines 85-135). Insert a new paragraph after line 124 ("STYLE: Concise...") with explicit rules: (1) If the question is not about calculus/mathematics, respond with "I'm sorry, I can only answer questions about calculus from Thomas' Calculus, 14th Edition." (2) If the retrieved documents don't contain sufficient information to answer, say "I don't have enough information in the textbook to answer this question." (3) Never fabricate formulas or theorems not found in the provided context. | | |
| TASK-002 | Add a pre-generation relevance check in `endpoints.py:retrieve_and_rerank` (after line 104, before returning docs). If `docs` is non-empty, compute a relevance score by checking how many of the user's query terms appear in the top-3 retrieved documents. If the overlap is < 10% (very low relevance), inject a warning document prepended to the docs list that says "WARNING: The retrieved textbook passages may not be relevant to this query." This triggers the OOD prompt behavior. | | |
| TASK-003 | Add a retrieval confidence check in `endpoints.py:query` (lines 115-137). After docs are returned, if docs is empty OR the relevance signal is low, skip generation entirely and return a graceful refusal response: "I can only answer questions based on Thomas' Calculus, 14th Edition. Your question doesn't appear to be covered in this textbook." | | |
| TASK-004 | Add confidence score threshold to `endpoints.py`. Compute a simple score: the fraction of query tokens (stopwords removed) that appear in at least one retrieved document's `page_content`. If < 0.15, route to the refusal path. | | |

### Implementation Phase 2: Reduce Hallucination on ch16 & Multi-hop

- GOAL-002: Fix the divergence theorem hallucination by improving query expansion, chunk content quality, and compression heuristics.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-005 | Add more synonym mappings in `expansion.py:MATH_SYNONYMS` (around line 11-12). Add: `"divergence theorem"` → `["gauss theorem", "gauss's theorem", "flux integral theorem"]`, `"flux"` → `["surface integral", "outward flux", "net flux"]`. Also add `"gauss law"` → `["gauss's law", "electric flux"]` for completeness. | | |
| TASK-006 | Improve compression heuristics in `compression.py:compress_documents`. Add a check: if the total character length of all chunks is < 8000 chars (~2000 tokens), skip compression entirely. This preserves full context for short but critical queries. Also add a check: if any chunk already contains ALL of the query's key terms (non-stopword tokens from query), don't compress that specific chunk (it's already relevant enough). | | |
| TASK-007 | Improve `math_cleaner.py` vector calculus artifact handling. Add cleanup for: `/uni2207` → `\nabla`, `/uni2202` → `\partial`, `/uni222B` → `\int`, `/uni222C` → `\iint`, `/uni222D` → `\iiint`, `/uni2A0F` → `\oint`, `/uni00D7` → `\times`. These are critical for the divergence theorem content which uses many vector calculus symbols. | | |
| TASK-008 | Verify by searching `parsed_docs.jsonl` for `/uni2207` (nabla), `/uni2202` (partial), `/uni222B` (integral) glyph codes that appear in ch16 content and ensure they are mapped in `_GLYPH_MAP`. Read `backend/data/markdown/parsed_docs.jsonl` around the ch16 chunks (IDs 1073-1087) to confirm. | | |

### Implementation Phase 3: Improve LaTeX Fidelity

- GOAL-003: Post-process LLM answers more aggressively to achieve > 0.90 LaTeX fidelity score.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-009 | Expand `latex_sanitizer.py:_wrap_bare_latex` regex pattern (line 127-132) to include more LaTeX commands: `\mathbf`, `\overrightarrow`, `\vec`, `\hat`, `\tilde`, `\bar`, `\overline`, `\underline`, `\oint`, `\iint`, `\iiint`, `\nabla`, `\partial`, `\times`, `\cdot`, `\circ`, `\propto`, `\forall`, `\exists`, `\implies`, `\iff`, `\mapsto`, `\hookrightarrow`, `\hookleftarrow`, `\uparrow`, `\downarrow`, `\updownarrow`, `\langle`, `\rangle`, `\lfloor`, `\rfloor`, `\lceil`, `\rceil`, `\mathcal`, `\mathbb`, `\mathrm`, `\textbf`, `\textit`, `\text`. | | |
| TASK-010 | Add a new sanitization step in `latex_sanitizer.py:sanitize_answer` (between steps 5 and 6, around line 73-75): "Fix unpaired $ delimiters". Add a function `_fix_unpaired_dollars(text: str) -> str` that ensures $...$ pairs are balanced. If a $ has no matching partner, escape it as `\$` or remove it. Also ensure `$$...$$` blocks are balanced. | | |
| TASK-011 | Add a new step in `latex_sanitizer.py:sanitize_answer`: "Fix superscript grouping". Add `_fix_superscript_braces(text: str) -> str` that converts patterns like `f^(k)(x)` → `f^{(k)}(x)`, `x^2y` → `x^{2}y` when the exponent is more than one character. This catches a common LLM mistake. | | |
| TASK-012 | Add a new step: "Fix missing backslash on known commands". Add `_fix_missing_backslash(text: str) -> str` that catches bare "sin", "cos", "tan", "log", "ln", "lim", "infty", "pi" when they appear to be used as LaTeX commands (preceded by math context or inside $...$ without backslash). Only fix when inside math delimiters. | | |
| TASK-013 | Strengthen the system prompt in `generator.py:_build_messages` (after line 122, the "OTHER RULES" section): Add a bolded warning at the very end: "CRITICAL: Review your output for LaTeX errors before responding. Every formula must have: (1) $ or $$ delimiters, (2) braces for all \\frac arguments, (3) braces for all subscripts/superscripts." | | |

### Implementation Phase 4: Reduce Latency (Compression & Reranker)

- GOAL-004: Cut P50 total latency from ~30s to < 15s by optimizing compression and making reranker optional/configurable.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-014 | Add a new config parameter `use_compression` (bool, default True) to `settings` in `config.py`. This allows disabling compression entirely when latency is critical. | | |
| TASK-015 | Modify `compression.py:compress_documents` to add a content-hash-based cache. Before calling `compress_chunk`, compute `hash(query + doc.page_content[:200])` and check a module-level dict `_compression_cache: dict[str, str]`. If found, return the cached result. This avoids re-compressing the same chunk for similar queries. | | |
| TASK-016 | Replace the per-chunk rate limiting in `compression.py:compress_chunk` with a single rate limit call before the compression batch. Currently each of 5 chunks waits `min(request_delay * 0.1, 0.5)` = ~0.5s. Instead, apply one global delay of 0.5s before starting the batch, then send all 5 compression calls without per-call delays. This saves ~2s per query. | | |
| TASK-017 | Add a "compression mode" setting to `config.py`: `compression_mode: str = "auto"` with values `"auto"` (current behavior), `"skip"` (skip all compression), `"batch"` (one LLM call for all chunks). Implement the `"batch"` mode in `compression.py` by creating a single prompt that asks the LLM to extract relevant sentences from ALL provided chunks at once, rather than one call per chunk. This reduces 5 calls → 1 call. | | |
| TASK-018 | In `endpoints.py:retrieve_and_rerank` (line 81), make the reranker skippable via a global threshold. If `settings.rerank_skip_threshold > 0` and the total characters of retrieved docs < threshold, skip reranking. Default: 0 (always rerank). This lets users skip reranking for short queries where BM25/dense is sufficient. | | |
| TASK-019 | Add a new config `rerank_skip_threshold: int = 0` to `config.py`. Also add `use_reranker: bool = True` to `settings`. Wire `use_reranker` into `endpoints.py:retrieve_and_rerank` so that when False, the reranking block (lines 81-100) is skipped entirely. | | |

### Implementation Phase 5: Verify and Evaluate

- GOAL-005: Run the evaluation script to verify all fixes achieve their targets. Add new comprehensive edge-case test queries.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-020 | Update `backend/scripts/eval_full.py` GROUND_TRUTHS to add 2 new OOD queries: "Write me a poem about calculus" and "What is the meaning of life?" to verify OOD handling is robust. Add 1 new adversarial query with extreme typos: "wh@t 1s d/dx s1n(x)?". | | |
| TASK-021 | Run `python backend/scripts/eval_full.py` and capture Phase 1-6 results. Verify: OOD > 0.80, Confidence Calibration > 0.80, LaTeX Fidelity > 0.90, Total P50 latency < 15s. | | |
| TASK-022 | If any target is not met, iterate on the specific fix and re-run only the relevant phase. Document any additional fixes needed. | | |

## 3. Alternatives

- **ALT-001 (OOD via classifier)**: Adding a separate ML-based OOD classifier before the RAG pipeline. Rejected because: (1) adds latency, (2) requires training data, (3) the prompt-based approach is simpler and sufficient.
- **ALT-002 (Skip compression entirely)**: Removing compression as a pipeline stage would save 12.4s but could reduce answer quality for complex queries. The hybrid approach (skip when small, cache results) preserves quality while reducing latency.
- **ALT-003 (Replace reranker with MMR-only)**: Using MMR diversity without the cross-encoder model. Rejected because NDCG@5 is currently 1.0 with the reranker — removing it entirely might hurt ranking quality even if keyword coverage is unchanged.
- **ALT-004 (Async compression calls)**: Running all 5 compression calls concurrently with asyncio. Rejected because the 9-router may not handle concurrent requests well and rate limiting becomes harder.

## 4. Dependencies

- **DEP-001**: No new library dependencies. All fixes use existing imports (re, hashlib for caching, existing transformers/LLM calls).
- **DEP-002**: The compression cache requires `hashlib` (stdlib) for content hashing.
- **DEP-003**: The evaluation requires existing Qdrant at `/tmp/qdrant_calculus_db` and BM25 at `/tmp/qdrant_calculus_bm25.pkl`.

## 5. Files

- **FILE-001**: `backend/app/generation/generator.py` — System prompt OOD/refusal instructions, LaTeX warning (TASK-001, TASK-013)
- **FILE-002**: `backend/app/api/endpoints.py` — Pre-generation relevance check, retrieval confidence check (TASK-002, TASK-003, TASK-004, TASK-018, TASK-019)
- **FILE-003**: `backend/app/retrieval/expansion.py` — New synonym mappings (TASK-005)
- **FILE-004**: `backend/app/retrieval/compression.py` — Skip heuristic, content cache, batch mode, rate limit optimization (TASK-006, TASK-015, TASK-016, TASK-017)
- **FILE-005**: `backend/app/retrieval/math_cleaner.py` — New glyph mappings for vector calculus (TASK-007, TASK-008)
- **FILE-006**: `backend/app/retrieval/latex_sanitizer.py` — Expanded command coverage, unpaired dollar fix, superscript fix, missing backslash fix (TASK-009, TASK-010, TASK-011, TASK-012)
- **FILE-007**: `backend/app/config.py` — New config parameters: `use_compression`, `compression_mode`, `use_reranker`, `rerank_skip_threshold` (TASK-014, TASK-019)
- **FILE-008**: `backend/scripts/eval_full.py` — Additional OOD, adversarial test queries (TASK-020)

## 6. Testing

- **TEST-001**: Run `python backend/tests/test_chunking.py` to verify existing tests still pass after changes.
- **TEST-002**: Manually test OOD: send "What is the capital of France?" to `POST /query` → should refuse gracefully.
- **TEST-003**: Manually test empty query: send `{"query": ""}` → should return refusal.
- **TEST-004**: Run `python backend/scripts/eval_full.py` after all fixes to confirm metrics improve.
- **TEST-005**: Verify streaming endpoint still works: `POST /query/stream` with normal query.
- **TEST-006**: Verify LaTeX output by querying "State the divergence theorem" and checking the answer has proper $$ delimiters, \\frac{}{} braces, and \\nabla glyph.

## 7. Risks & Assumptions

- **RISK-001**: The compression cache (TASK-015) may grow unbounded for diverse queries. Mitigation: cap at 1000 entries using LRU eviction.
- **RISK-002**: The OOD prompt-based approach (TASK-001) may cause false positives for legitimate but unusual calculus questions. Mitigation: the OOD trigger fires only when retrieval relevance is very low (< 15% token overlap), not based on prompt alone.
- **RISK-003**: Batch compression mode (TASK-017) may produce lower quality extractions than per-chunk calls. Mitigation: batch mode is optional via `compression_mode` config, default stays as per-chunk.
- **ASSUMPTION-001**: The 9-router supports concurrent requests for batch compression. If not, the batch mode falls back gracefully to sequential calls.
- **ASSUMPTION-002**: The `/uni2207` and `/uni2202` glyph codes actually appear in the ch16 parsed content. If not, TASK-008 will verify and the mappings will still be harmless no-ops.

## 8. Related Specifications / Further Reading

- Full evaluation results: `AGENTS.md` (Final Evaluation — All Phases Complete section)
- Evaluation code: `backend/scripts/eval_full.py`
- Phase 4 latency breakdown: AGENTS.md Phase 4: Latency & Throughput
- Phase 5 robustness details: AGENTS.md Phase 5: Robustness & Safety
- Phase 6 cost-quality tradeoff: AGENTS.md Phase 6: Cost & Efficiency
