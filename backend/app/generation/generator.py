import time
import re
from typing import Optional, Generator

from openai import OpenAI
from langchain_core.documents import Document
from app.config import settings


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

        system_prompt = """You are a precise math tutor for Thomas' Calculus, 14th Edition.

RULES:
1. Answer ONLY using the provided context.
2. Cite EVERY claim, formula, theorem with [N] immediately after the statement.
3. Structure your answer:
   - **Answer**: Direct response
   - **Key Points**: Bulleted list of important concepts
   - **Formula** (if applicable): Exact mathematical expression
4. If the context lacks sufficient information, say EXACTLY what is missing.
5. For formulas, reproduce them exactly as they appear in the source.
6. Never invent notation, theorems, or proofs not present in the context."""

        messages = [{"role": "system", "content": system_prompt}]

        if history:
            for msg in history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        user_prompt = f"""Context (cite as [N]):
{context}

Question: {query}

Follow the structure: Answer, Key Points, Formula (if applicable). Cite every claim."""
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _verify_citations(self, answer: str, documents: list[Document]) -> list[Document]:
        cited_indices = set()
        for doc_idx in range(1, len(documents) + 1):
            if f"[{doc_idx}]" in answer:
                cited_indices.add(doc_idx - 1)
        return [documents[i] for i in sorted(cited_indices)]

    def _validate_answer(self, answer: str, documents: list[Document]) -> tuple[bool, str]:
        cited = self._verify_citations(answer, documents)
        if not cited:
            return False, "No citations found in answer"

        for i, doc in enumerate(documents):
            tag = f"[{i+1}]"
            if tag not in answer:
                continue
            doc_text = doc.page_content.lower()
            answer_claims = self._extract_claims_around_citation(answer, tag)
            for claim in answer_claims:
                claim_terms = set(re.findall(r'\b[a-zA-Z]\w+\b', claim.lower()))
                doc_terms = set(re.findall(r'\b[a-zA-Z]\w+\b', doc_text))
                overlap = len(claim_terms & doc_terms)
                if len(claim_terms) > 2 and overlap < len(claim_terms) * 0.3:
                    return False, f"Citation [{i+1}] does not match claimed content"
        return True, ""

    def _extract_claims_around_citation(self, text: str, tag: str) -> list[str]:
        parts = text.split(tag)
        claims = []
        for i in range(len(parts) - 1):
            before = parts[i].split(". ")[-1][-200:] if parts[i] else ""
            after = parts[i + 1][:200].split(". ")[0] if parts[i + 1] else ""
            claims.append(before + " " + after)
        return claims

    def _self_consistency(self, query: str, documents: list[Document], history: list[dict] | None = None) -> dict:
        from collections import Counter

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

        if len(answers) == 1:
            best_answer = answers[0]
        else:
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

        citations = self._verify_citations(best_answer, documents)
        return {
            "answer": best_answer,
            "citations": citations,
            "raw_documents": documents,
            "_samples": len(answers),
        }

    def generate(self, query: str, documents: list[Document], history: list[dict] | None = None) -> dict:
        if settings.use_self_consistency and len(documents) > 0:
            return self._self_consistency(query, documents, history)

        messages = self._build_messages(query, documents, history)
        response = self._call_llm(
            messages,
            model=self.primary_model,
            temperature=0.1,
            max_tokens=2048,
        )
        answer = response.choices[0].message.content
        citations = self._verify_citations(answer, documents)

        valid, msg = self._validate_answer(answer, documents)
        if not valid and len(documents) > 0:
            print(f"  Answer validation: {msg} — regenerating with stricter prompt")
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
            citations = self._verify_citations(answer, documents)

        return {
            "answer": answer,
            "citations": citations,
            "raw_documents": documents,
        }

    def generate_stream(
        self, query: str, documents: list[Document], history: list[dict] | None = None
    ) -> Generator[str, None, dict]:
        messages = self._build_messages(query, documents, history)
        full_answer = ""

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
                yield delta

        citations = self._verify_citations(full_answer, documents)

        yield {
            "citations": citations,
            "raw_documents": documents,
        }
