# Generation Accuracy — Implementation Checklist

## Phase 1: NLI Verifier
- [ ] `nli_verifier.py` — `NLIVerifier` class with roberta-large-mnli
- [ ] `entailment_score(claim, context) → float`
- [ ] `batch_verify(claims, contexts) → list[dict]`
- [ ] LRU cache (1000 entries, MD5 keyed)
- [ ] Graceful fallback if model unavailable

## Phase 2: Claim Extractor
- [ ] `claim_extractor.py` — `AtomicClaimExtractor` class + `Claim` dataclass
- [ ] Rule-based `_split_claims()` — sentence splitting + coordination resolution
- [ ] LLM-based `_extract_with_llm()` fallback
- [ ] `extract_claims(answer, docs) → list[Claim]`
- [ ] Citation parsing + math detection + core-claim tagging

## Phase 3: Pipeline Integration
- [ ] `citation_verifier.py` — NLI as primary signal (weight 0.5)
- [ ] `verify_all_citations()` → claim-level verification
- [ ] `generator.py` — `_validate_answer()` uses claim-level NLI
- [ ] Regeneration on failed core claims (max 2 retries)
- [ ] `endpoints.py` — claim confidence in response

## Phase 4: Semantic Self-Consistency
- [ ] `_self_consistency()` — claim-level semantic aggregation
- [ ] Claim majority voting across samples
- [ ] `config.py` — `use_self_consistency: True`, `self_consistency_samples: 3`

## Phase 5: Uncertainty & Abstention
- [ ] `confidence.py` — `ConfidenceScorer` with 7-signal composite
- [ ] `should_abstain(claim, confidence) → bool`
- [ ] Abstention: qualification text + optional regeneration
- [ ] Config: `confidence_threshold_abstain: 0.5`, `confidence_threshold_warn: 0.7`

## Phase 6: Sub-Question Decomposition
- [ ] `expansion.py` — improved `decompose_multi_hop()`
- [ ] `endpoints.py` — per-sub-query retrieval + merge
- [ ] `generator.py` — structured multi-part prompt

## Phase 7: Evaluation
- [ ] `eval_full.py` — NLI claim faithfulness, precision/recall/F1
- [ ] `eval_full.py` — noise sensitivity (relevant + irrelevant)
- [ ] `eval_full.py` — context utilization metric
- [ ] `eval_full.py` — abstention rate + confidence gap
- [ ] `eval_accuracy.py` — standalone accuracy eval script

## Validation
- [ ] NLI: `entailment_score("derivative of sin is cos", "context") > 0.9`
- [ ] NLI: `entailment_score("sky is blue", "math context") < 0.1`
- [ ] Claim split: compound sentences → 2+ claims
- [ ] Multi-hop: L'Hôpital + Taylor → ch4 + ch10 docs
- [ ] Noise: +2 irrelevant chunks → accuracy drop < 10%
- [ ] `cd backend && python -m pytest tests/ -v` — all 21 pass
- [ ] `cd backend && python -m scripts.eval_accuracy` — completes cleanly
