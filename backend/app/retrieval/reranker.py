import math
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

    def rerank_with_mmr(self, query: str, texts: list[str], top_k: int = 0, lambda_mmr: float = 0.7) -> list[tuple[str, float]]:
        try:
            scores = self._rerank_local(query, texts)
        except Exception as e:
            print(f"Local reranker failed: {e}")
            scores = [0.0] * len(texts)

        top_k = top_k or settings.top_k_rerank
        if len(texts) <= top_k:
            return list(zip(texts, scores))

        selected = []
        remaining = list(range(len(texts)))

        def _similarity(i: int, j: int) -> float:
            ti = texts[i].lower().split()
            tj = texts[j].lower().split()
            if not ti or not tj:
                return 0.0
            inter = len(set(ti) & set(tj))
            union = len(set(ti) | set(tj))
            return inter / union if union else 0.0

        for _ in range(top_k):
            if not remaining:
                break
            best_idx = -1
            best_score = -float("inf")

            for idx in remaining:
                relevance = scores[idx]
                if selected:
                    max_sim = max(_similarity(idx, sel) for sel in selected)
                else:
                    max_sim = 0.0
                mmr_score = lambda_mmr * relevance - (1 - lambda_mmr) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return [(texts[i], scores[i]) for i in selected]
