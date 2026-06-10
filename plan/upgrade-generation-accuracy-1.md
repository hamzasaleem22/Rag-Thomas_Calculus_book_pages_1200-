---
goal: Maximum Generation Accuracy for Calculus RAG — NLI Verification, Claim Decomposition, Self-Consistency, UQ
version: 1.0
date_created: 2026-06-10
owner: hamza
status: 'Planned'
tags: feature, quality, accuracy, nli, verification, evaluation
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

Comprehensive plan to achieve maximal generation accuracy in the Thomas' Calculus RAG pipeline. Addresses all identified gaps from the literature review (RagChecker, SELF-RAG, FVA-RAG, Conformal-RAG, HALT-RAG, Franq) with a focus on claim-level NLI-based faithfulness verification, semantic self-consistency, uncertainty quantification, and sub-question decomposition.

## 1. Requirements & Constraints

- **REQ-001**: All generated claims must be verifiable against retrieved source chunks with measurable confidence
- **REQ-002**: Citation verification must use NLI entailment (not just lexical overlap) as primary signal
- **REQ-003**: Answers must be decomposable into atomic factual claims for fine-grained verification
- **REQ-004**: Self-consistency must use semantic similarity, not word overlap
- **REQ-005**: Each claim must carry a calibrated confidence score
- **REQ-006**: Low-confidence claims must trigger abstention or qualification
- **REQ-007**: Multi-hop queries must be decomposed into sub-questions with per-sub-query retrieval
- **REQ-008**: Evaluation must report claim-level metrics (precision, recall, F1, hallucination rate)
- **CON-001**: All NLI inference must run locally on CPU (no API costs, no external dependencies)
- **CON-002**: Maximum added latency per query ≤ 3s for NLI + claim extraction combined
- **CON-003**: No breaking changes to existing API response schema
- **PAT-001**: Follow existing code structure — new modules go in `backend/app/retrieval/`
- **PAT-002**: Use LRU caches for NLI and claim extraction results (like compression.py)
- **PAT-003**: Keep evaluation-only code in `backend/scripts/`, not in `backend/app/`

## 2. Implementation Steps

### Implementation Phase 1 — NLI Faithfulness Verifier

- GOAL-001: Create local NLI entailment model wrapper to replace lexical token-overlap as primary verification signal

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Create `backend/app/retrieval/nli_verifier.py` with `NLIVerifier` class that loads `roberta-large-mnli` from HuggingFace transformers. Model loads lazily on first call. Cache predictions with LRU (max 1000 entries) keyed by MD5(claim_text + doc_text[:200]). | | |
| TASK-002 | Implement `entailment_score(claim: str, context: str) -> float` — returns softmax probability of ENTAILMENT label. Add `batch_verify(claims: list[Claim], contexts: list[str]) -> list[float]` for batched inference. | | |
| TASK-003 | Implement `verify_entailment_batch(claim_texts: list[str], doc_texts: list[str]) -> list[dict]` — returns list of `{entailment, neutral, contradiction, label}` per claim using torch.no_grad() batching (max 16 pairs per batch). | | |
| TASK-004 | Add model caching to avoid reloading: store tokenizer + model in module-level dict, reset on exception. Handle HuggingFace `OOM` or missing model gracefully → fall back to lexical verification. | | |
| TASK-005 | Validate model loads and infer on CPU: `python3 -c "from app.retrieval.nli_verifier import NLIVerifier; v=NLIVerifier(); print(v.entailment_score('derivative of sin is cos', 'the derivative of sin(x) is cos(x)'))"` — must return >0.9. | | |

### Implementation Phase 2 — Claim-Level Answer Decomposition

- GOAL-002: Decompose generated answer text into atomic factual claims for fine-grained verification

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-006 | Create `backend/app/retrieval/claim_extractor.py` with `AtomicClaimExtractor` class. Define `Claim` dataclass: `text: str`, `sentence_index: int`, `is_mathematical: bool`, `has_citation: bool`, `citation_indices: list[int]`, `is_core_claim: bool`. | | |
| TASK-007 | Implement rule-based `_split_claims(text: str) -> list[str]`: split by sentence boundaries (`.!?`), then further split compound sentences on `"and"`, `"or"`, `"which"` where each conjunct is an independent factual assertion. Handle math: don't split inside `$...$` or `$$...$$`. For example: "The derivative of sin(x) is cos(x) and the derivative of cos(x) is -sin(x)" → two claims. | | |
| TASK-008 | Implement LLM-based `_extract_with_llm(text: str) -> list[str]` as fallback: prompt the primary LLM to decompose complex answers into atomic claims. Only triggered when rule-based extraction produces 1 claim from a >100-char answer (indicating splitting failed). Cache LLM results with LRU (200 entries). | | |
| TASK-009 | Implement `extract_claims(answer: str, documents: list[Document]) -> list[Claim]`: parses citations, calls `_split_claims`, tags each with `is_mathematical` (contains LaTeX commands or `$`), `is_core_claim` (contains key math concepts: theorem, definition, formula, rule, or any LaTeX expression), and `citation_indices`. | | |
| TASK-010 | Validate: `python3 -c "from app.retrieval.claim_extractor import AtomicClaimExtractor; e=AtomicClaimExtractor(); claims=e.extract_claims('The derivative of sin(x) is cos(x) [1]. The chain rule states dy/dx = dy/du * du/dx [2].'); print(len(claims))"` — must return 2 claims. | | |

### Implementation Phase 3 — Integrate NLI + Claims into Pipeline

- GOAL-003: Wire NLI verification and claim extraction into citation_verifier.py and generator.py

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-011 | Update `backend/app/retrieval/citation_verifier.py`: add `_nli_verifier` singleton (lazy-loaded `NLIVerifier`). Add new function `verify_claim_nli(claim: Claim, doc: Document) -> dict` that returns `{nli_entailment, nli_neutral, nli_contradiction, composite_score, verified}`. | | |
| TASK-012 | Update composite score formula in `verify_citation()`: new weights = 0.5 * NLI_entailment + 0.2 * token_overlap + 0.2 * entity_overlap + 0.1 * key_terms_score. If NLI model fails to load, fall back to old formula (0.5 * token + 0.3 * entity + 0.2 * key_terms). | | |
| TASK-013 | Replace `verify_all_citations()` to work at claim level: call `extract_claims()` → per-claim NLI verification → aggregate results. Return `{results: dict[doc_idx, list[claim_result]], all_verified, avg_score, citation_count, claim_count, claims_by_doc}`. | | |
| TASK-014 | Update `_validate_answer()` in `generator.py`: use new claim-level verification. If any core claim (is_core_claim=True) fails verification, regenerate with specific instruction to fix those claims. Track number of regenerations (max 2). | | |
| TASK-015 | Update `endpoints.py` query response: add `claim_confidence_scores` to response metadata. For each claim, include `{text, confidence, is_verified}` in the API response for frontend transparency. | | |
| TASK-016 | Run tests: `cd backend && python -m pytest tests/ -v` — all 21 existing tests must pass. Verify NLI integration doesn't break existing citation flow. | | |

### Implementation Phase 4 — Semantic Self-Consistency

- GOAL-003: Upgrade self-consistency from word-overlap voting to claim-level semantic aggregation

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-017 | Update `generator.py` `_self_consistency()`: replace sentence-overlap scoring with claim-level semantic similarity. For each sample, extract claims via `AtomicClaimExtractor`. Compare claims across samples using sentence-transformers (`all-MiniLM-L6-v2`, already in dependencies) cosine similarity. | | |
| TASK-018 | Implement claim-level majority voting: for each claim position (aligned by semantic similarity clustering), pick the version supported by the most samples. If samples disagree on a claim, use the highest-NLI-score version (requires Phase 1). | | |
| TASK-019 | Set `use_self_consistency: True` and `self_consistency_samples: 3` in `config.py`. Add `self_consistency_agg_method: str = "claim_vote"` config option. Keep `max_tokens: 2048`. | | |
| TASK-020 | Add claim agreement score to output: what fraction of claims are consistent across all N samples. Returns `{answer, claim_agreement_rate, n_samples}`. | | |

### Implementation Phase 5 — Uncertainty Quantification & Abstention

- GOAL-004: Calibrated confidence per claim with abstention when uncertain

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-021 | Create `backend/app/retrieval/confidence.py` with `ConfidenceScorer` class. Input signals per claim: (1) NLI entailment score, (2) NLI contradiction score, (3) token overlap score, (4) entity overlap score, (5) self-consistency agreement rate (if self-consistency enabled), (6) number of citations supporting the claim, (7) whether it's a core claim. | | |
| TASK-022 | Implement `score_claim(claim: Claim, verification_result: dict, consistency_rate: float) -> float`: composite confidence = w1 * (1 - nli_contradiction) + w2 * nli_entailment + w3 * consistency_rate + w4 * citation_support. Weights: [0.25, 0.35, 0.25, 0.15]. Clamp to [0, 1]. | | |
| TASK-023 | Implement `should_abstain(claim: Claim, confidence: float) -> bool`: returns True if confidence < 0.5 AND claim is a core claim (is_core_claim=True). Returns True if confidence < 0.3 for any claim. | | |
| TASK-024 | Integrate abstention into `generator.py`: after generation and verification, check each claim. For low-confidence claims, append qualification text (e.g., "Note: I'm less certain about [claim]"). If substantial abstention (>30% of claims), regenerate with instruction to use only verified information. | | |
| TASK-025 | Add `confidence_threshold_abstain: float = 0.5` and `confidence_threshold_warn: float = 0.7` to `config.py`. | | |

### Implementation Phase 6 — Sub-Question Decomposition for Multi-Hop

- GOAL-005: Integrate multi-hop query decomposition into the retrieval pipeline

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-026 | Update `expansion.py`: improve `decompose_multi_hop()` to also detect implicit multi-hop queries (not just those with "and"/"then"). Add heuristics: presence of multiple math concepts across chapters (e.g., "L'Hôpital" + "Taylor series"), comparative questions, compound problems. | | |
| TASK-027 | Update `retrieve_and_rerank()` in `endpoints.py`: call `decompose_multi_hop()` on each query. If sub-queries detected, run retrieval per sub-query independently, merge results with deduplication (by chunk_id), and pass merged doc set to generator. | | |
| TASK-028 | Update `generator.py` `_build_messages()`: when multi-hop query detected, add structured prompt instructing the model to answer each sub-part sequentially with separate citations. Use format: "Part 1: [answer] [citations]. Part 2: [answer] [citations]." | | |
| TASK-029 | Validate: query "Use L'Hôpital's rule to find lim sin(x)/x, then confirm using Taylor series" → should retrieve from both chapters 4 (L'Hôpital) and 10 (Taylor series). Verify with eval: `cd backend && python -c "from app.api.endpoints import retrieve_and_rerank; docs=retrieve_and_rerank('Use L Hopital rule to find limit sin(x)/x then confirm with Taylor series', rerank=False); chapters={d.metadata.get('chapter_number') for d in docs}; print(chapters)"` → should include both ch4 and ch10. | | |

### Implementation Phase 7 — Enhanced Evaluation Framework

- GOAL-006: Update eval with NLI-based claim-level metrics, noise sensitivity, and context utilization

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-030 | Update `eval_full.py` Phase 2 (Generation Quality): replace keyword-coverage faithfulness with NLI-based claim faithfulness. For each generated answer, extract claims via `AtomicClaimExtractor`, verify each with `NLIVerifier`, compute supported/total ratio. Report `nli_claim_faithfulness` as primary metric. | | |
| TASK-031 | Add claim-level metrics to eval_full.py Phase 2: `claim_precision` (verified claims / total claims), `claim_recall` (ground-truth claims covered / total ground-truth claims), `claim_f1`. Compute ground-truth claims by extracting claims from ground-truth answer. | | |
| TASK-032 | Add noise sensitivity metrics (RagChecker-style): for each query, inject 2 irrelevant chunks into retrieved set. Measure: `relevant_noise_sensitivity` (incorrect claims entailed in relevant chunks), `irrelevant_noise_sensitivity` (incorrect claims entailed in irrelevant chunks), `hallucination_rate` (incorrect claims not entailed in any chunk). Compute per query and average. | | |
| TASK-033 | Add context utilization metric: `context_utilization` = |{ground-truth claims entailed in both retrieved chunks AND response}| / |{ground-truth claims entailed in retrieved chunks}|. Higher = generator better uses retrieved info. | | |
| TASK-034 | Add abstention/calibration metrics: `abstention_rate` (fraction of queries with any abstention), `confidence_gap` (avg confidence of correct vs incorrect claims). | | |
| TASK-035 | Create `backend/scripts/eval_accuracy.py` — standalone accuracy-focused eval script that runs only Phase 2 + Phase 3 (generation + end-to-end) with the new NLI-based metrics. Outputs a single JSON report with all claim-level metrics. | | |
| TASK-036 | Update `backend/scripts/evaluate_chunks.py` to verify no chunk contains NLI-verifiable contradictions (self-contradictory chunks flagged for review). | | |

## 3. Alternatives

- **ALT-001 (LLM-as-judge NLI)**: Using the existing LLM API for entailment checking instead of a local NLI model. Rejected because: (a) adds ~3-5s per claim verification, (b) uses API quota on kr models (already rate-limited), (c) local model runs in ~100ms per claim on CPU with <1GB RAM.
- **ALT-002 (GPT-4o-mini for NLI)**: Would need a new API connection. Rejected for consistency — all inference should use the local 9-router or local models. 
- **ALT-003 (DeBERTa-large-mnli)**: Higher accuracy than RoBERTa-large-mnli but 2x slower and 3x more RAM. Rejected because roberta-large-mnli achieves 90.4% on MNLI (DeBERTa is 91.3%) — marginal gain for 2x latency cost.
- **ALT-004 (No abstention mechanism)**: Let the answer stand as-is regardless of confidence. Rejected because the user's goal is 100% accuracy — low-confidence claims must be flagged.

## 4. Dependencies

- **DEP-001**: `transformers` >= 4.30.0 (already in requirements.txt for cross-encoder reranker) — needed for NLI model
- **DEP-002**: `torch` (already in requirements.txt) — needed for NLI inference
- **DEP-003**: `sentence-transformers` (already in requirements.txt for self-consistency eval) — needed for semantic claim aggregation
- **DEP-004**: RoBERTa-large-mnli model downloaded on first use (~1.5GB disk) — verify `~/.cache/huggingface/hub/` has space
- **DEP-005**: NLI cache directory at `/tmp/nli_cache/` — automatically created, max 500MB

## 5. Files

| File | Purpose |
|------|---------|
| `backend/app/retrieval/nli_verifier.py` | **NEW** — Local NLI entailment model wrapper |
| `backend/app/retrieval/claim_extractor.py` | **NEW** — Atomic claim decomposition from answers |
| `backend/app/retrieval/confidence.py` | **NEW** — Confidence scoring and abstention logic |
| `backend/app/retrieval/citation_verifier.py` | **MODIFY** — Add NLI + claim-level verification |
| `backend/app/generation/generator.py` | **MODIFY** — Integrate claim verification, semantic self-consistency, abstention |
| `backend/app/api/endpoints.py` | **MODIFY** — Integrate sub-question decomposition, add claim confidence to response |
| `backend/app/api/schemas.py` | **MODIFY** — Add ClaimConfidence model to response schema |
| `backend/app/config.py` | **MODIFY** — Add new config options (self_consistency on, confidence thresholds) |
| `backend/app/retrieval/expansion.py` | **MODIFY** — Enhance multi-hop decomposition |
| `backend/scripts/eval_full.py` | **MODIFY** — NLI-based claim metrics, noise sensitivity, context utilization |
| `backend/scripts/eval_accuracy.py` | **NEW** — Focused accuracy eval script |
| `backend/scripts/evaluate_chunks.py` | **MODIFY** — Add NLI-based chunk contradiction check |

## 6. Testing

| Test | Description |
|------|-------------|
| TEST-001 | NLI model loads and produces correct entailment: `entailment_score("derivative of sin is cos", "The derivative of sin(x) is cos(x)")` > 0.9. `entailment_score("derivative of sin is cos", "The sky is blue")` < 0.1. |
| TEST-002 | Claim extractor correctly splits compound sentences: "A and B" → 2 claims. Does not split inside math delimiters. |
| TEST-003 | Citation verifier uses NLI as primary signal: previously passing lexical tests still pass, previously failing semantic tests now pass (e.g., "the derivative of sine is cosine" matches "d/dx sin(x) = cos(x)"). |
| TEST-004 | Self-consistency with 3 samples produces same answer structure (same claims, same citations). Claim agreement rate > 0.8 for standard queries. |
| TEST-005 | Low-confidence claims trigger abstention: inject a deliberately unsupported claim into generation → confidence < 0.3 → abstention triggered. |
| TEST-006 | Multi-hop query retrieves from multiple chapters: "L'Hôpital + Taylor" → docs from both ch4 and ch10. |
| TEST-007 | Noise sensitivity: adding 2 irrelevant chunks to top-5 should not degrade accuracy > 10%. |
| TEST-008 | All existing 21 pytest tests pass: `cd backend && python -m pytest tests/ -v`. |
| TEST-009 | End-to-end: `cd backend && python -m scripts.eval_accuracy` completes without error and produces JSON report. |

## 7. Risks & Assumptions

- **RISK-001 (NLI model download)**: roberta-large-mnli is ~1.5GB. On first load, download may take 2-5 minutes. Mitigation: add startup download script that user can run before server starts: `python3 -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('roberta-large-mnli'); AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')"`.
- **RISK-002 (CPU inference latency)**: roberta-large-mnli on CPU takes ~200-300ms per pair. For 5 claims × 5 docs = 25 pairs → ~5-7.5s. Mitigation: batch verification (16 pairs at once → ~1.5s total). Cache for repeated pairs.
- **RISK-003 (RAM usage)**: roberta-large-mnli uses ~1.5GB RAM loaded. Current system uses ~2GB (sentence-transformers + cross-encoder). Total ~3.5GB — acceptable on 8GB+ system. Monitor with `nvidia-smi` or `free -h`.
- **ASSUMPTION-001**: The primary LLM (cx/gpt-5.5 when kr models are exhausted) cooperates with claim-level prompt instructions. If it ignores the prompt, claim extraction will still catch unverifiable claims.
- **ASSUMPTION-002**: Claim extraction may over-split (too many fine-grained claims) or under-split (too coarse). The 2-strategy approach (rule-based + LLM fallback) handles both extremes.
- **ASSUMPTION-003**: The NLI model trained on general-domain MNLI transfers acceptably to calculus/math domain. If accuracy is poor, fine-tune on a small set of calculus-specific claim-context pairs.

## 8. Related Specifications / Further Reading

- RagChecker: A Fine-grained Framework for Diagnosing RAG — https://arxiv.org/abs/2408.08067
- SELF-RAG: Self-Reflective Retrieval Augmented Generation — https://arxiv.org/abs/2310.11511
- FVA-RAG: Falsification-Verification Alignment for Mitigating Sycophantic Hallucinations — https://arxiv.org/abs/2512.07015
- HALT-RAG: Task-Adaptable Framework for Hallucination Detection with Calibrated NLI Ensembles — https://arxiv.org/abs/2509.07475
- Conformal-RAG: Response Quality Assessment via Conditional Conformal Factuality — https://arxiv.org/abs/2506.20978
- Franq: Faithfulness-Aware Uncertainty Quantification for Fact-Checking RAG — https://arxiv.org/abs/2505.21072
