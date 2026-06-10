import time
import re
from typing import Optional, Generator

from openai import OpenAI
from langchain_core.documents import Document
from app.config import settings
from app.retrieval.latex_sanitizer import sanitize_answer
from app.retrieval.citation_verifier import verify_all_citations, get_cited_documents, verify_answer_claims


class Generator:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.openai_api_key or "no-key",
            base_url=settings.openai_base_url,
        )
        self.primary_model = settings.llm_model
        self.fallback_model = settings.llm_model_fallback
        self.legacy_model = settings.llm_model_legacy
        self.last_request_time = 0.0
        self._response_cache: dict[str, dict] = {}

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < settings.request_delay:
            time.sleep(settings.request_delay - elapsed)
        self.last_request_time = time.time()

    def _call_llm(self, messages: list[dict], model: str, temperature: float = 0.1, max_tokens: int = 2048, stream: bool = False):
        models_to_try = [model, self.fallback_model, self.legacy_model]
        if model not in models_to_try:
            models_to_try.append(model)

        tried = set()
        for attempt in range(settings.max_retries):
            for m in models_to_try:
                if m in tried:
                    continue
                tried.add(m)
                try:
                    self._rate_limit()
                    return self.client.chat.completions.create(
                        model=m,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=stream,
                    )
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str:
                        print(f"  Rate limited on {m}, waiting 30s...")
                        time.sleep(30)
                        continue
                    if attempt < settings.max_retries - 1:
                        print(f"  {m} failed: {err_str[:80]}, retrying...")
                        time.sleep(5)
                    continue
            if attempt < settings.max_retries - 1:
                time.sleep(10)

        raise RuntimeError(f"All models exhausted after {settings.max_retries} retries")

    def format_context(self, documents: list[Document]) -> str:
        lines = []
        for i, doc in enumerate(documents):
            meta = doc.metadata
            page = meta.get("page", "?")
            chapter = meta.get("chapter", "")
            section = meta.get("section", "")
            source = f"[{i+1}]"
            if chapter:
                source += f" {chapter}"
            if section:
                source += f", {section}"
            source += f" (p.{page})"
            lines.append(f"{source}\n{doc.page_content}\n")
        return "\n".join(lines)

    def _build_messages(self, query: str, documents: list[Document], history: list[dict] | None = None) -> list[dict]:
        context = self.format_context(documents)

        system_prompt = """You are a concise math tutor for Thomas' Calculus, 14th Edition.

RESPONSE STRUCTURE:
1. **Answer:** 2-3 sentences directly answering the question.
2. **Key Points:** 2-4 short bullet points (one sentence each).
3. **Formula:** (only if the answer involves a formula) The key formula on its own line.

CITATIONS: Add [N] after each factual claim. Keep it brief.

MATH FORMATTING — CRITICAL RULES:
- ALL math MUST be inside $...$ (inline) or $$...$$ (display).
- NEVER write bare LaTeX commands outside of $...$ delimiters.
- NEVER output a formula twice (once as LaTeX and once as Unicode).

LaTeX SYNTAX — YOU MUST USE BRACES:
  \\frac{numerator}{denominator}  ← ALWAYS use {braces} for both parts
  \\sum_{k=0}^{\\infty}           ← ALWAYS use {braces} for sub/superscripts
  f^{(n)}(x)                      ← ALWAYS use {braces} for exponents

CORRECT EXAMPLES (copy this style exactly):
  $\\frac{d}{dx}[\\sin(x)] = \\cos(x)$
  $\\int_0^1 x^2 \\, dx = \\frac{1}{3}$
  $\\lim_{x \\to 0} \\frac{\\sin(x)}{x} = 1$
  $\\sum_{k=0}^{\\infty} \\frac{f^{(k)}(a)}{k!}(x - a)^k$
  $f'(x) = 2x$
  $$\\frac{dy}{dx} = f'(g(x)) \\cdot g'(x)$$

WRONG (never do this):
  \\fracf(x)g(x)     ← MISSING BRACES
  \\fracdydx         ← MISSING ALL BRACES (must be \\frac{dy}{dx})
  \\fracdydudot\\fracdudx  ← MISSING BRACES AND \\cdot (use \\frac{dy}{du} \\cdot \\frac{du}{dx})
  \\sum k=0 \\infty  ← MISSING BRACES AND $ DELIMITERS
  f (n) (x)          ← SPACES INSTEAD OF ^{(n)}
  d¸ots              ← GARBLED COMMAND (use \\dots)

OTHER RULES:
- Use $f(x)$ not plain "f(x)" outside delimiters.
- Use $\\sin$, $\\cos$, $\\tan$, $\\ln$, $\\lim$ — always inside $.
- Keep formulas SHORT. Break complex ones into steps.
- If source text has garbled math, RECONSTRUCT clean LaTeX.

SCOPE & UNCERTAINTY — CRITICAL RULES:
- ONLY answer questions about calculus, math, or related topics from Thomas' Calculus, 14th Edition.
- You MAY answer questions about the book's structure, table of contents, chapter organization, and what topics are covered in the textbook.
- If the question is NOT about calculus/mathematics AND NOT about the textbook itself (e.g. history, geography, general knowledge unrelated to the book), respond with:
  "I'm sorry, I can only answer questions about calculus from Thomas' Calculus, 14th Edition."
- If the retrieved documents do NOT contain sufficient information to answer, respond with:
  "I don't have enough information in the textbook to answer this question."
- NEVER fabricate formulas, theorems, or definitions not found in the provided context.
- If you are uncertain about a claim, do not guess. State what you know and note the uncertainty.

STYLE: Concise. ChatGPT-style. Max 300 words. No filler.

EXAMPLE RESPONSE:
**Answer:**
The Taylor series of $f(x)$ at $x = a$ is an infinite sum using derivatives of $f$ at $a$ [1].

**Key Points:**
- The Taylor series is $\\sum_{k=0}^{\\infty} \\frac{f^{(k)}(a)}{k!}(x - a)^k$ [1]
- When $a = 0$, it is called the Maclaurin series [2]

**Formula:**
$$f(x) = \\sum_{k=0}^{\\infty} \\frac{f^{(k)}(a)}{k!}(x - a)^k$$"""

        messages = [{"role": "system", "content": system_prompt}]

        if history:
            for msg in history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        user_prompt = f"""Context (cite as [N]):
{context}

Question: {query}

Follow the structure: Answer, Key Points, Formula (if applicable). Cite every claim.

CRITICAL: Review your output for LaTeX errors before responding. Every formula must have: (1) $ or $$ delimiters, (2) braces for all \\frac arguments, (3) braces for all subscripts and superscripts."""
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _verify_citations(self, answer: str, documents: list[Document]) -> list[Document]:
        return get_cited_documents(answer, documents)

    def _validate_answer(self, answer: str, documents: list[Document]) -> tuple[bool, str, dict]:
        cited = get_cited_documents(answer, documents)
        if not cited:
            return False, "No citations found in answer", {}

        verification = verify_all_citations(answer, documents)
        if not verification["all_verified"]:
            for doc_idx, result in verification["results"].items():
                for claim_result in result["claims"]:
                    if not claim_result["verified"]:
                        return False, (
                            f"Citation [{doc_idx + 1}] failed verification "
                            f"(score={claim_result['composite_score']}, nli_label={claim_result.get('nli_label','N/A')})"
                        ), verification
        if verification["citation_count"] < 1:
            return False, f"Only {verification['citation_count']} citation(s), need at least 1", verification

        # Also verify with atomic claim extraction for detailed claim-level analysis
        claim_results = verify_answer_claims(answer, documents)
        failed_core = [r for r in claim_results if r.is_core_claim and not r.verified]
        if failed_core:
            msg = f"Core claims failed: {', '.join(r.claim_text[:50] for r in failed_core[:3])}"
            return False, msg + " — regenerating", verification

        return True, "", verification

    def _self_consistency(self, query: str, documents: list[Document], history: list[dict] | None = None) -> dict:
        from app.retrieval.claim_extractor import AtomicClaimExtractor

        messages = self._build_messages(query, documents, history)

        answers = []
        for i in range(settings.self_consistency_samples):
            try:
                response = self._call_llm(
                    messages,
                    model=self.primary_model,
                    temperature=settings.self_consistency_temperature,
                    max_tokens=2048,
                )
                answers.append(response.choices[0].message.content)
            except Exception as e:
                print(f"  Self-consistency sample {i+1} failed: {e}")
                continue
            if i < settings.self_consistency_samples - 1:
                time.sleep(settings.request_delay)

        if not answers:
            raise RuntimeError("All self-consistency samples failed")

        if len(answers) == 1 or not settings.self_consistency_use_semantic:
            best_answer = answers[0]
            if len(answers) > 1:
                # Legacy: sentence overlap voting
                sentence_sets = []
                for ans in answers:
                    sentences = set(re.findall(r'[^.!?]+[.!?]', ans))
                    sentence_sets.append(sentences)
                best_idx = 0
                best_score = -1
                for i, s_set in enumerate(sentence_sets):
                    score = sum(
                        len(s_set & other)
                        for j, other in enumerate(sentence_sets)
                        if j != i
                    )
                    if score > best_score:
                        best_score = score
                        best_idx = i
                best_answer = answers[best_idx]
        else:
            # Semantic: claim-level clustering with sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as np
                from sklearn.cluster import DBSCAN

                extractor = AtomicClaimExtractor()

                # Extract claims from each sample
                all_claims_with_sample = []
                for idx, ans in enumerate(answers):
                    claims = extractor.extract_claims(ans)
                    for claim in claims:
                        all_claims_with_sample.append((claim.text, idx))

                if len(all_claims_with_sample) < 2:
                    best_answer = answers[0]
                else:
                    claim_texts = [c[0] for c in all_claims_with_sample]
                    embedder = SentenceTransformer('all-MiniLM-L6-v2')
                    embeddings = embedder.encode(claim_texts, show_progress_bar=False)

                    # Cluster claims by semantic similarity
                    clustering = DBSCAN(eps=0.3, min_samples=1, metric='cosine')
                    labels = clustering.fit_predict(embeddings)

                    # Score each claim cluster by cross-sample agreement
                    cluster_sample_count = {}
                    for i, label in enumerate(labels):
                        sample_idx = all_claims_with_sample[i][1]
                        if label not in cluster_sample_count:
                            cluster_sample_count[label] = set()
                        cluster_sample_count[label].add(sample_idx)

                    # Score each sample: reward claims that appear across many samples
                    best_idx = 0
                    best_score = -1
                    for idx in range(len(answers)):
                        score = 0.0
                        for i, label in enumerate(labels):
                            if all_claims_with_sample[i][1] == idx:
                                cross_samples = len(cluster_sample_count[label])
                                score += cross_samples  # More cross-sample agreement = higher score
                        if score > best_score:
                            best_score = score
                            best_idx = idx

                    best_answer = answers[best_idx]
                    print(f"  Semantic self-consistency: sample {best_idx+1}/{len(answers)} chosen (score={best_score:.1f})")
            except Exception as e:
                print(f"  Semantic self-consistency failed: {e}, falling back to first sample")
                best_answer = answers[0]

        citations = self._verify_citations(best_answer, documents)
        sanitized_answer = sanitize_answer(best_answer)

        valid, msg, verification = self._validate_answer(sanitized_answer, documents)
        claim_confidences = []
        for doc_idx, result in verification.get("results", {}).items():
            for claim_result in result.get("claims", []):
                claim_confidences.append({
                    "doc_idx": doc_idx,
                    "composite_score": claim_result.get("composite_score", 0),
                    "nli_label": claim_result.get("nli_label", "neutral"),
                    "verified": claim_result.get("verified", False),
                })

        return {
            "answer": sanitized_answer,
            "citations": citations,
            "raw_documents": documents,
            "_samples": len(answers),
            "claim_verification": claim_confidences,
        }

    def generate(self, query: str, documents: list[Document], history: list[dict] | None = None) -> dict:
        cache_key = query.strip().lower()
        if settings.response_cache_enabled and cache_key in self._response_cache:
            cached = self._response_cache[cache_key]
            if len(documents) > 0:
                cached["citations"] = self._verify_citations(cached["answer"], documents)
                cached["raw_documents"] = documents
            return cached

        if settings.use_self_consistency and len(documents) > 0:
            result = self._self_consistency(query, documents, history)
            if settings.response_cache_enabled:
                self._response_cache[cache_key] = {
                    "answer": result["answer"],
                    "citations": result["citations"],
                    "claim_verification": result.get("claim_verification", []),
                }
                if len(self._response_cache) > settings.response_cache_max_size:
                    oldest = next(iter(self._response_cache))
                    del self._response_cache[oldest]
            return result

        messages = self._build_messages(query, documents, history)
        response = self._call_llm(
            messages,
            model=self.primary_model,
            temperature=0.1,
            max_tokens=2048,
        )
        answer = response.choices[0].message.content
        answer = sanitize_answer(answer)
        citations = self._verify_citations(answer, documents)

        valid, msg, verification = self._validate_answer(answer, documents)
        if not valid and len(documents) > 0:
            print(f"  Answer validation: {msg} — regenerating with stricter prompt")

            # Build targeted instruction from failed claims
            failed_claims = []
            for doc_idx, result in verification.get("results", {}).items():
                for claim_result in result.get("claims", []):
                    if not claim_result.get("verified", True):
                        nli_label = claim_result.get("nli_label", "")
                        score = claim_result.get("composite_score", 0)
                        failed_claims.append(f"- Claim {doc_idx + 1}: score={score}, nli_label={nli_label}")

            if failed_claims:
                failed_text = "\n".join(failed_claims)
                messages[-1]["content"] = (
                    messages[-1]["content"]
                    + "\n\nThe following claims in your previous answer could not be verified:\n"
                    + failed_text
                    + "\n\nPlease provide a corrected answer. Only include claims supported by the source material."
                )
            else:
                messages[-1]["content"] = (
                    messages[-1]["content"]
                    + "\n\nIMPORTANT: You MUST include citations like [1], [2] after each claim. Every formula needs a citation."
                )
            response = self._call_llm(
                messages,
                model=self.primary_model,
                temperature=0.1,
                max_tokens=2048,
            )
            answer = response.choices[0].message.content
            answer = sanitize_answer(answer)
            citations = self._verify_citations(answer, documents)

        # Attach claim-level confidence scores with UQ
        claim_confidences = []
        average_confidence = 0.0
        abstain_rate = 0.0
        try:
            from app.retrieval.confidence import ConfidenceScorer
            atomic_claims = verify_answer_claims(answer, documents)
            scorer = ConfidenceScorer()
            for r in atomic_claims:
                c = scorer.score_claim(
                    claim_text=r.claim_text,
                    nli_score=r.nli_score,
                    nli_label=r.nli_label,
                    token_overlap=r.token_overlap_score,
                    entity_overlap=r.entity_overlap_score,
                    is_mathematical=r.is_mathematical,
                )
                claim_confidences.append({
                    "doc_idx": r.source_doc_idx if r.source_doc_idx is not None else 0,
                    "composite_score": r.composite_score,
                    "nli_label": r.nli_label,
                    "verified": r.verified,
                    "is_core": r.is_core_claim,
                    "confidence": c.composite_confidence,
                    "abstain": c.abstain,
                })

            if claim_confidences:
                average_confidence = sum(c["confidence"] for c in claim_confidences) / len(claim_confidences)
                abstain_rate = sum(1 for c in claim_confidences if c["abstain"]) / len(claim_confidences)
        except Exception as e:
            print(f"  Claim-level verification failed: {e}")
            _, _, final_verification = self._validate_answer(answer, documents)
            for doc_idx, result in final_verification.get("results", {}).items():
                for claim_result in result.get("claims", []):
                    claim_confidences.append({
                        "doc_idx": doc_idx,
                        "composite_score": claim_result.get("composite_score", 0),
                        "nli_label": claim_result.get("nli_label", "neutral"),
                        "verified": claim_result.get("verified", False),
                        "confidence": claim_result.get("composite_score", 0),
                        "abstain": claim_result.get("composite_score", 0) < 0.4,
                    })

        # Append abstention warning if high abstention rate
        if abstain_rate > 0.3:
            answer += "\n\n*Note: Some parts of this answer have low confidence. Consider verifying with the source material.*"

        if settings.response_cache_enabled:
            self._response_cache[cache_key] = {
                "answer": answer,
                "citations": citations,
                "claim_verification": claim_confidences,
                "average_confidence": average_confidence,
            }
            if len(self._response_cache) > settings.response_cache_max_size:
                oldest = next(iter(self._response_cache))
                del self._response_cache[oldest]

        return {
            "answer": answer,
            "citations": citations,
            "raw_documents": documents,
            "cached": False,
            "claim_verification": claim_confidences,
            "average_confidence": average_confidence,
            "abstain_rate": abstain_rate,
        }

    def generate_stream(
        self, query: str, documents: list[Document], history: list[dict] | None = None
    ) -> Generator[str, None, dict]:
        messages = self._build_messages(query, documents, history)
        full_answer = ""
        last_yielded_len = 0

        stream = self._call_llm(
            messages,
            model=self.primary_model,
            temperature=0.1,
            max_tokens=2048,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_answer += delta
                sanitized = sanitize_answer(full_answer)
                new_text = sanitized[last_yielded_len:]
                if new_text:
                    yield new_text
                    last_yielded_len = len(sanitized)

        citations = self._verify_citations(sanitize_answer(full_answer), documents)

        yield {
            "citations": citations,
            "raw_documents": documents,
        }
