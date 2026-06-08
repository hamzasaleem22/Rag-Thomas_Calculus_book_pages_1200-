from typing import Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from app.config import settings


_model_cache: dict[str, tuple] = {}

def _load_reranker(model_name: str):
    if model_name not in _model_cache:
        token = settings.get_hf_token() or None
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token=token)
        model.eval()
        _model_cache[model_name] = (tokenizer, model)
    return _model_cache[model_name]


class Reranker:
    def __init__(self, model_name: str = ""):
        self.model_name = model_name or settings.reranker_model

    def rerank(self, query: str, texts: list[str], top_k: int = 0) -> list[tuple[str, float]]:
        try:
            scores = self._rerank_local(query, texts)
        except Exception as e:
            print(f"Local reranker failed: {e}")
            scores = [0.0] * len(texts)

        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        top_k = top_k or settings.top_k_rerank
        results = [(texts[i], s) for i, s in indexed[:top_k]]
        return results

    def _rerank_local(self, query: str, texts: list[str]) -> list[float]:
        tokenizer, model = _load_reranker(self.model_name)
        pairs = [(query, t) for t in texts]
        inputs = tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        with torch.no_grad():
            outputs = model(**inputs)
            scores = outputs.logits.view(-1).float().tolist()
        return scores
