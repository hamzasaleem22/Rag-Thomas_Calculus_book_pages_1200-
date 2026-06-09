---
goal: Comprehensive RAG Evaluation — Thomas Calculus
version: 1.0
date_created: 2026-06-09
owner: RAG Book Team
status: 'In progress'
tags: evaluation, rag, metrics, quality, robustness, latency
---

# Introduction

![Status: In progress](https://img.shields.io/badge/status-In%20progress-yellow)

Evaluate the Thomas Calculus RAG pipeline across 6 dimensions: Retrieval Quality, Generation Quality, End-to-End Metrics, Latency & Throughput, Robustness & Safety, and Cost & Efficiency. All via a single standalone script using 9-router + local models.

## 1. Requirements & Constraints

- **REQ-001**: All metrics must be computable with open-source tools, 9-router, or existing local models
- **REQ-002**: No new model downloads — only use models already cached or available via 9-router
- **REQ-003**: Rate limit of 3-5 seconds between queries + existing `request_delay` (10s)
- **REQ-004**: Console-only output, reported phase-by-phase
- **REQ-005**: 25-30 ground truth queries across 6 categories (standard, multi-hop, OOD, adversarial, edge, consistency)
- **CON-001**: Must import modules directly (not hit live API) for per-stage latency profiling
- **CON-002**: RAGAS answer_relevancy may fail with non-OpenAI API — fallback to custom LLM-as-judge

## 2. Implementation Steps

### Implementation Phase 1: Build evaluation script skeleton + ground truth

- GOAL-001: Create `backend/scripts/eval_full.py` with ground truth dataset, utility functions, and phase 1 retrieval evaluation

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Create eval_full.py with imports, ground truth (25 queries), judge client, rate limiter | ✅ | 2026-06-09 |
| TASK-002 | Implement Phase 1: Retrieval Quality (Context Precision, Recall, NDCG, MRR, Hit Rate, BM25 vs Dense vs Hybrid) | ✅ | 2026-06-09 |
| TASK-003 | Implement Phase 2: Generation Quality (Faithfulness, Relevancy, Correctness, Hallucination, LaTeX) | ✅ | 2026-06-09 |
| TASK-004 | Implement Phase 3: End-to-End Metrics (RAGAS composite, F1, EM, Consistency, CSAT proxy) | ✅ | 2026-06-09 |
| TASK-005 | Implement Phase 4: Latency & Throughput (P50/P95/P99 per stage, TTFT) | ✅ | 2026-06-09 |
| TASK-006 | Implement Phase 5: Robustness & Safety (OOD, adversarial, multi-hop, edge cases) | ✅ | 2026-06-09 |
| TASK-007 | Implement Phase 6: Cost & Efficiency ($/query, token efficiency, reranker tradeoff) | ✅ | 2026-06-09 |

### Implementation Phase 2: Execute Phase 1 — Retrieval Quality

- GOAL-002: Run the retrieval evaluation on standard queries and report results

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-008 | Run Phase 1 evaluation script | ✅ | 2026-06-09 |
| TASK-009 | Report results and update AGENTS.md | ✅ | 2026-06-09 |

### Implementation Phase 3: Execute Phase 2 — Generation Quality

- GOAL-003: Run the generation evaluation on all queries and report results

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-010 | Run Phase 2 evaluation | | |
| TASK-011 | Report results and update AGENTS.md | | |

### Implementation Phase 4: Execute Phase 3 — End-to-End Metrics

- GOAL-004: Run end-to-end metrics and report

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-012 | Run Phase 3 evaluation | | |
| TASK-013 | Report results and update AGENTS.md | | |

### Implementation Phase 5: Execute Phase 4 — Latency & Throughput

- GOAL-005: Profile latency per stage and report

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-014 | Run Phase 4 latency evaluation | | |
| TASK-015 | Report results and update AGENTS.md | | |

### Implementation Phase 6: Execute Phase 5 — Robustness & Safety

- GOAL-006: Evaluate OOD, adversarial, multi-hop, edge cases

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-016 | Run Phase 5 robustness evaluation | | |
| TASK-017 | Report results and update AGENTS.md | | |

### Implementation Phase 7: Execute Phase 6 — Cost & Efficiency

- GOAL-007: Evaluate cost and token efficiency

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-018 | Run Phase 6 cost evaluation | | |
| TASK-019 | Report results and update AGENTS.md | | |

### Implementation Phase 8: Final Report

- GOAL-008: Collate all results into a comprehensive final evaluation summary

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-020 | Produce final collated report in console | | |
| TASK-021 | Update AGENTS.md with all results | | |

## 3. Alternatives

- **ALT-001**: Using RAGAS entirely — rejected because answer_relevancy is incompatible with non-OpenAI providers and we need custom metrics (LaTeX, OOD, latency).
- **ALT-002**: Hitting live API endpoints — rejected because per-stage latency profiling requires direct module access.

## 4. Dependencies

- **DEP-001**: Qdrant at `/tmp/qdrant_calculus_db` must be populated (indexed data exists)
- **DEP-002**: BM25 state at `/tmp/qdrant_calculus_bm25.pkl` must exist and be fitted
- **DEP-003**: 9-router at `http://localhost:20128/v1` must be running with `kr/claude-sonnet-4.5`
- **DEP-004**: `all-MiniLM-L6-v2` sentence-transformers model must be cached locally

## 5. Files

- **FILE-001**: `backend/scripts/eval_full.py` — the evaluation orchestrator (new)
- **FILE-002**: `AGENTS.md` — updated with results after each phase
- **FILE-003**: `docs/superpowers/specs/2026-06-09-rag-evaluation-design.md` — the design spec (already written)

## 6. Testing

- **TEST-001**: Each phase function can run independently with `python -c "from scripts.eval_full import phase_1; phase_1()"`
- **TEST-002**: Assert that all metric scores are in range [0.0, 1.0]
- **TEST-003**: Assert that latency values are positive floats

## 7. Risks & Assumptions

- **RISK-001**: 9-router rate limits. Mitigation: 3-5s delays + 10s request_delay.
- **RISK-002**: RAGAS answer_relevancy incompatibility. Mitigation: custom LLM-as-judge fallback.
- **ASSUMPTION-001**: BM25 is fitted and Qdrant collection exists.
- **ASSUMPTION-002**: `kr/claude-sonnet-4.5` is available (verified — 9-router reports it).

## 8. Related Specifications / Further Reading

- `docs/superpowers/specs/2026-06-09-rag-evaluation-design.md`
- `backend/scripts/evaluate.py` — existing RAGAS evaluation
- `backend/scripts/evaluate_deep.py` — existing deep evaluation with term matching
- `backend/scripts/diagnose_pipeline.py` — existing pipeline diagnostic
