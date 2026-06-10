import math
import re
from collections import Counter
from typing import Optional

from app.config import settings


class Reranker:
    def __init__(self, model_name: str = ""):
        self.model_name = model_name or settings.reranker_model
        self._model_cache: dict = {}

    def rerank(self, query: str, texts: list[str], top_k: int = 0, method: str = "hybrid") -> list[tuple[str, float]]:
        top_k = top_k or settings.top_k_rerank
        if not texts:
            return []

        if method == "token_overlap":
            scores = self._score_token_overlap(query, texts)
        elif method == "bm25":
            scores = self._score_bm25(query, texts)
        elif method == "cross_encoder":
            scores = self._rerank_local(query, texts) if self._cross_encoder_available() else self._score_token_overlap(query, texts)
        else:
            cross_scores = self._rerank_local(query, texts) if self._cross_encoder_available() else self._score_token_overlap(query, texts)
            overlap_scores = self._score_token_overlap(query, texts)
            bm25_scores = self._score_bm25(query, texts)
            scores = [
                0.4 * c + 0.3 * o + 0.3 * b
                for c, o, b in zip(cross_scores, overlap_scores, bm25_scores)
            ]

        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        results = [(texts[i], s) for i, s in indexed[:top_k]]
        return results

    def rerank_with_mmr(self, query: str, texts: list[str], top_k: int = 0, lambda_mmr: float = 0.7) -> list[tuple[str, float]]:
        scores = self._score_token_overlap(query, texts)

        top_k = top_k or settings.top_k_rerank
        if len(texts) <= top_k:
            return list(zip(texts, scores))

        selected = []
        remaining = list(range(len(texts)))

        for _ in range(top_k):
            if not remaining:
                break
            best_idx = -1
            best_score = -float("inf")

            for idx in remaining:
                relevance = scores[idx]
                if selected:
                    max_sim = max(self._jaccard_similarity(texts[idx], texts[sel]) for sel in selected)
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

    def _score_token_overlap(self, query: str, texts: list[str]) -> list[float]:
        query_tokens = set(re.findall(r'\b[a-zA-Z]\w+\b', query.lower()))
        stopwords = {'the', 'is', 'at', 'which', 'what', 'how', 'do', 'does', 'a', 'an',
                     'and', 'or', 'of', 'to', 'for', 'in', 'on', 'by', 'with', 'from', 'its',
                     'are', 'be', 'was', 'were', 'been', 'being', 'that', 'this', 'it', 'we'}
        query_tokens -= stopwords

        scores = []
        for text in texts:
            doc_tokens = set(re.findall(r'\b[a-zA-Z]\w+\b', text.lower()))
            if not query_tokens:
                scores.append(0.0)
            else:
                overlap = len(query_tokens & doc_tokens)
                scores.append(overlap / len(query_tokens))

        return scores

    def _score_bm25(self, query: str, texts: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
        query_tokens = re.findall(r'\w+', query.lower())

        doc_token_sets = []
        for text in texts:
            tokens = re.findall(r'\w+', text.lower())
            doc_token_sets.append(tokens)

        num_docs = len(texts)
        if num_docs == 0:
            return []

        doc_freqs = {}
        for tokens in doc_token_sets:
            seen = set()
            for t in tokens:
                if t not in seen:
                    doc_freqs[t] = doc_freqs.get(t, 0) + 1
                    seen.add(t)

        doc_lengths = [len(t) for t in doc_token_sets]
        avg_dl = sum(doc_lengths) / max(num_docs, 1)

        scores = []
        for tokens in doc_token_sets:
            doc_tf = Counter(tokens)
            score = 0.0
            dl = len(tokens)
            for term in query_tokens:
                if term not in doc_freqs:
                    continue
                tf = doc_tf.get(term, 0)
                if tf == 0:
                    continue
                df = doc_freqs[term]
                idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
                score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1))))
            scores.append(score)

        if any(s > 0 for s in scores):
            max_score = max(scores)
            scores = [s / max_score for s in scores]

        return scores

    def _rerank_local(self, query: str, texts: list[str]) -> list[float]:
        try:
            tokenizer, model = self._load_reranker(self.model_name)
            pairs = [(query, t) for t in texts]
            inputs = tokenizer(
                pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
            )
            import torch
            with torch.no_grad():
                outputs = model(**inputs)
                scores = outputs.logits.view(-1).float().tolist()
            return scores
        except Exception as e:
            print(f"  Cross-encoder reranker failed: {e}")
            return self._score_token_overlap(query, texts)

    def _load_reranker(self, model_name: str):
        if model_name not in self._model_cache:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            token = settings.get_hf_token() or None
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, token=token)
            model.eval()
            self._model_cache[model_name] = (tokenizer, model)
        return self._model_cache[model_name]

    def _cross_encoder_available(self) -> bool:
        return False

    def _jaccard_similarity(self, t1: str, t2: str) -> float:
        tokens1 = set(t1.lower().split())
        tokens2 = set(t2.lower().split())
        if not tokens1 or not tokens2:
            return 0.0
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        return len(intersection) / len(union)
