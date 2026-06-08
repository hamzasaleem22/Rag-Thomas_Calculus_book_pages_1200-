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
        self.model = settings.llm_model

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

        system_prompt = """You are a helpful math tutor specializing in Thomas' Calculus, 14th Edition. Answer based ONLY on the provided context. For each claim, cite the source as [N] where N is the reference number. If the context doesn't contain enough information, say so."""

        messages = [{"role": "system", "content": system_prompt}]

        if history:
            for msg in history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        user_prompt = f"""Context:
{context}

Question: {query}

Answer the question using only the context above. Include citations like [1], [2] etc."""

        messages.append({"role": "user", "content": user_prompt})
        return messages

    def generate(self, query: str, documents: list[Document], history: list[dict] | None = None) -> dict:
        messages = self._build_messages(query, documents, history)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
        )

        answer = response.choices[0].message.content

        cited_indices = set()
        for doc_idx in range(1, len(documents) + 1):
            if f"[{doc_idx}]" in answer:
                cited_indices.add(doc_idx - 1)

        citations = [documents[i] for i in sorted(cited_indices)]

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
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_answer += delta
                yield delta

        cited_indices = set()
        for doc_idx in range(1, len(documents) + 1):
            if f"[{doc_idx}]" in full_answer:
                cited_indices.add(doc_idx - 1)

        yield {
            "citations": [documents[i] for i in sorted(cited_indices)],
            "raw_documents": documents,
        }
