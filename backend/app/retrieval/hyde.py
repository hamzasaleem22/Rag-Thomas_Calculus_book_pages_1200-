import time
from typing import Optional
from openai import OpenAI
from app.config import settings

_client: Optional[OpenAI] = None
_last_request_time = 0.0

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.openai_api_key or "no-key",
            base_url=settings.openai_base_url,
        )
    return _client


HYPOTHETICAL_PROMPT = """Given a calculus question, write a short passage from a calculus textbook that would perfectly answer it. Use the style of Thomas' Calculus 14th Edition. Include relevant formulas where appropriate.

Question: {query}

Hypothetical textbook passage:"""


def generate_hypothetical_document(query: str) -> str:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < settings.request_delay:
        time.sleep(settings.request_delay - elapsed)
    _last_request_time = time.time()

    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": "You write concise textbook passages that perfectly answer calculus questions."},
                {"role": "user", "content": HYPOTHETICAL_PROMPT.format(query=query)},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  HyDE generation failed: {e}")
        return query
