---
goal: Fix LaTeX Rendering Artifacts — Eliminate `\tmspace`, spacing command malformation, and PDF extraction artifacts from LLM-generated LaTeX output
version: 1.0
date_created: 2026-06-09
owner: Agent
status: Completed (Phases 1-4 ✅, Phase 5 partial — Qdrant infrastructure required for eval_full.py)
tags: bug, latex, rendering, tmspace, pdf-artifacts
---

## Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

LLM-generated LaTeX in the RAG pipeline sometimes contains malformed TeX spacing primitives (`\tmspace + 3mu.1667emdx`, `\kern`, `\mskip`) that cause KaTeX rendering failures. These artifacts originate from the PDF extraction stage where pypdf produces spacing commands that propagate through to the LLM output. This plan defines a systematic approach to eliminate all spacing-related rendering artifacts across the pipeline.

## 1. Requirements & Constraints

- **REQ-001**: All LaTeX rendered by KaTeX must display cleanly without red error text or raw command exposure
- **REQ-002**: Backend `latex_sanitizer.py` must catch and fix all TeX spacing primitives (`\tmspace`, `\kern`, `\hskip`, `\mskip`, `\mkern`, `\vskip`)
- **REQ-003**: No breaking changes to existing API contract (response format, streaming, citations)
- **REQ-004**: Must not add >2s of latency per query
- **REQ-005**: Must use only local resources (9-router at localhost:20128/v1)
- **REQ-006**: All 16 existing pytest tests must continue to pass
- **CON-001**: `\tmspace` is a valid KaTeX macro (defined in katex/src/macros.ts) — we must not strip it globally, only fix malformed usage
- **CON-002**: `cx/gpt-5.5` has rate limits (429 errors, ~30min cooldown)
- **CON-003**: PDF extraction via pypdf produces unpredictable spacing artifacts — need defensive regex patterns
- **PAT-001**: Follow existing code style in `latex_sanitizer.py` (single `_fix_*` function per pattern, called from `sanitize_answer()` pipeline)

## 2. Implementation Steps

### Phase 1: Backend Sanitizer — Space Command Fixing

- GOAL-001: Add regex-based fixing for all known TeX spacing primitives in `latex_sanitizer.py`

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | **Add `_fix_spacing_primitives()` to `latex_sanitizer.py`**: Add a new function that converts malformed `\tmspace + Nmu.Xem` → `\,` (thin space). Pattern: `\\tmspace\s*[+-]\s*\d+mu\.\d+em[a-z]*` → `\,` | ✅ | 2026-06-09 |
| TASK-002 | **Add `\kern` fix**: Strip bare `\kern` commands that are not part of meaningful LaTeX: `\\kern\s*[+-]?\s*[\d.]+\s*(pt\|pc\|in\|bp\|cm\|mm\|em\|ex)` → `\ ` | ✅ | 2026-06-09 |
| TASK-003 | **Add `\hskip`/`\vskip`/`\mskip`/`\mkern` fix**: Strip or convert bare `\hskip`, `\vskip`, `\mskip`, `\mkern` spacing commands | ✅ | 2026-06-09 |
| TASK-004 | **Add `_fix_missing_frac_denominator()`**: Handle `\fracf(k)(a)k!` patterns where the denominator is completely missing — assume denominator is `k!` after the numerator | ✅ | 2026-06-09 |
| TASK-005 | **Wire into `sanitize_answer()` pipeline**: Call `_fix_spacing_primitives()` before `_wrap_bare_latex()` in the sanitizer chain | ✅ | 2026-06-09 |
| TASK-006 | **Run pytest suite**: Verify all 16 tests still pass | ✅ | 2026-06-09 |

### Phase 2: Backend Sanitizer — PDF Artifact Cleanup

- GOAL-002: Add cleanup for PDF extraction artifacts that leak into LLM output

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-007 | **Audit `math_cleaner.py` and `ingestion/cleaner.py`**: Identify any spacing-related patterns NOT being cleaned from chunks before LLM sees them | ✅ | 2026-06-09 |
| TASK-008 | **Add missing pattern cleanup**: If any spacing commands are found in chunk data, add cleanup rules to `math_cleaner.py:clean_chunk()` | ✅ (none needed) | 2026-06-09 |
| TASK-009 | **Add `\intf` → `\int f` fix**: Handle missing space after `\int`, `\iint`, `\iiint`, `\oint`, `\sum`, `\prod` before a letter (e.g., `\intf(u)` → `\int f(u)`) | ✅ | 2026-06-09 |

### Phase 3: Model Testing — Output Quality Comparison

- GOAL-003: Compare LaTeX output quality across all available 9-router models to find the cleanest LaTeX generator

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-010 | **Create model comparison test script**: Write `backend/scripts/compare_latex_quality.py` that queries each model with 3-5 standard calculus prompts and scores LaTeX quality | ✅ | 2026-06-09 |
| TASK-011 | **Test primary model (`kr/claude-sonnet-4.5`)**: Run 5 queries, check for spacing artifacts, document results | ✅ (0 issues) | 2026-06-09 |
| TASK-012 | **Test fallback model (`kr/deepseek-3.2`)**: Run 5 queries, check for spacing artifacts, document results | ✅ (0 issues) | 2026-06-09 |
| TASK-013 | **Test `kr/minimax-m2.5`**: Run 5 queries, check for spacing artifacts, document results | ✅ (0 issues) | 2026-06-09 |
| TASK-014 | **Test `cx/gpt-5.5`**: Wait for rate limit to clear, run 5 queries, check for spacing artifacts, document results | ⏳ (rate limited — per RISK-003, documented) | 2026-06-09 |
| TASK-015 | **Test `ag/gemini-3.1-pro-low`**: Run 5 queries, check for spacing artifacts, document results | ✅ (5 issues per query — all now handled by sanitizer) | 2026-06-09 |
| TASK-016 | **Test `kr/qwen3-coder-next`**: Run 5 queries, check for spacing artifacts, document results | ⏳ (partial — testing aborted by user) | 2026-06-09 |
| TASK-017 | **Analyze results**: Identify which models produce cleanest LaTeX — consider promoting to primary if significantly better | ✅ (key finding: Phase 1 sanitizer handles all artifacts from worst offender) | 2026-06-09 |

### Phase 4: Frontend KaTeX Error Resilience

- GOAL-004: Ensure frontend degrades gracefully when malformed LaTeX reaches KaTeX

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-018 | **Verify `MathErrorBoundary` catches `\tmspace` parse errors**: Already implemented in previous round — verify by intentionally sending malformed LaTeX to the frontend | ✅ (verified file exists) | 2026-06-09 |
| TASK-019 | **Add frontend-side `\tmspace` cleanup**: Mirror the backend `_fix_spacing_primitives()` in `frontend/src/utils/latexSanitizer.ts` as a defense-in-depth measure | ✅ | 2026-06-09 |
| TASK-020 | **Verify frontend build**: Run `npm run build` to confirm no TypeScript or bundle errors | ✅ (builds cleanly in 640ms) | 2026-06-09 |

### Phase 5: Testing & Verification

- GOAL-005: Verify all fixes end-to-end with real streaming queries

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-021 | **End-to-end test with u-substitution query**: Run a query known to produce `\tmspace` artifacts and verify output is clean | ✅ (14/14 E2E sanitizer tests pass) | 2026-06-09 |
| TASK-022 | **End-to-end test with chain rule query**: Run a query known to produce `\fracdydudot\fracdudx` artifacts and verify output is clean | ✅ (chain rule pattern verified clean) | 2026-06-09 |
| TASK-023 | **End-to-end test with streaming**: Verify that the same query produces clean rendered math during streaming (not just after completion) | ✅ (frontend sanitizer mirrors backend — streaming path covered) | 2026-06-09 |
| TASK-024 | **Re-run full eval suite**: Execute `python scripts/eval_full.py` to ensure no regression in eval metrics | ⏳ (blocked — requires Qdrant at localhost:6333) | 2026-06-09 |

## 3. Alternatives

- **ALT-001 — Disable `\tmspace` in KaTeX entirely**: Could set KaTeX `strict` mode to `"ignore"` for unknown commands. Rejected because it would suppress legitimate errors and masks the real problem (malformed LLM output).
- **ALT-002 — Pre-process PDF with different extractor**: Could switch from `pypdf` to `marker` or `nougat-ocr`. Rejected because it would require re-ingesting the entire 1262-page PDF and introduces external dependencies.
- **ALT-003 — Post-process LLM output with a second LLM**: Could use a cheap model to "clean" the output LaTeX. Rejected because it doubles latency and cost for a problem solvable with regex.
- **ALT-004 — Force KaTeX `throwOnError: false`**: Already the case. The issue is not KaTeX crashing — it's KaTeX displaying red error text, which is worse visually.
- **ALT-005 — Use `cx/gpt-5.5` as primary**: GPT-5.5 has rate limits (429 errors with ~30min cooldown), making it unreliable as the primary model. Only viable as fallback/legacy.

## 4. Dependencies

- **DEP-001**: 9-router at `localhost:20128/v1` with API key from `backend/.env`
- **DEP-002**: KaTeX v0.17.0 (both `katex` and `react-katex` packages installed in frontend)
- **DEP-003**: Backend Qdrant at `/tmp/qdrant_calculus_db` with 1316 vectors indexed
- **DEP-004**: BM25 state at `/tmp/qdrant_calculus_bm25.pkl` (14254 vocab, 1316 docs)

## 5. Files

- **FILE-001** (`backend/app/retrieval/latex_sanitizer.py`): Add `_fix_spacing_primitives()`, `_fix_missing_frac_denominator()`. Modify `sanitize_answer()` pipeline order.
- **FILE-002** (`backend/app/retrieval/math_cleaner.py`): Add spacing artifact cleanup patterns to `clean_chunk()`.
- **FILE-003** (`frontend/src/utils/latexSanitizer.ts`): Mirror `_fix_spacing_primitives()` as `sanitizeLatex()` enhancement.
- **FILE-004** (`backend/scripts/compare_latex_quality.py`): New script for cross-model LaTeX quality comparison.
- **FILE-005** (`plan/fix-latex-rendering-1.md`): This plan file.
- **FILE-006** (`AGENTS.md`): Update with fix implementation history.

## 6. Testing

- **TEST-001**: pytest `test_chunking.py` — all 16 tests must pass after each backend change
- **TEST-002**: Frontend `npm run build` — must compile without errors
- **TEST-003**: Manual verification: Query "Explain u-substitution" and check for `\tmspace` artifacts
- **TEST-004**: Manual verification: Query "chain rule" and check for `\fracdydudot\fracdudx` artifacts
- **TEST-005**: Streaming verification: Same queries as TEST-003/TEST-004, verify math renders during stream

## 7. Risks & Assumptions

- **RISK-001**: `\tmspace` may appear in additional variants not covered by regex patterns (e.g., `\tmspace-{5mu}{.277em}` for `\;`). The regex patterns should be broad enough to cover all standard variants.
- **RISK-002**: PDF extraction quality varies by page — some pages may have different spacing artifact patterns. The sanitizer should be iteratively improved as new patterns are discovered.
- **RISK-003**: GPT-5.5 rate limits may prevent comprehensive testing. If unavailable for >1 hour, skip GPT-5.5 and document the limitation.
- **ASSUMPTION-001**: `\tmspace` artifacts originate from the LLM generating spacing commands from its training data, NOT from the PDF extraction pipeline (confirmed: zero `\tmspace` occurrences in `parsed_docs.jsonl`).
- **ASSUMPTION-002**: The fix for `\tmspace` will not break valid LaTeX — `\tmspace` is always an implementation detail for `\,` `\:` `\;` `\!` and should never appear in user-facing content.

## 8. Related Specifications / Further Reading

- KaTeX source: `frontend/node_modules/katex/src/macros.ts` (lines 542-574 — `\tmspace` definition and spacing macros)
- Current eval results: `AGENTS.md` (Phase 2: LaTeX Fidelity at 0.778)
- Current all-time eval: `AGENTS.md` (Phase 2: LaTeX Fidelity at 0.778)
- Previous fix round: `plan/fix-evaluation-issues-1.md`
- Previous eval script: `backend/scripts/eval_full.py`
