import hashlib
from collections import OrderedDict
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class NLIVerifier:
    def __init__(self, model_name: str = "roberta-large-mnli", device: str = "cpu", cache_max: int = 1000):
        self.device = device
        self.model_name = model_name
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSequenceClassification] = None
        self._loaded = False
        self._cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._cache_max = cache_max

    def _load_model(self):
        if self._loaded:
            return
        print(f"  Loading NLI model {self.model_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        self._loaded = True
        print(f"  NLI model loaded on {self.device}")

    def _make_cache_key(self, claim: str, context: str) -> str:
        raw = claim.strip().lower()[:100] + "|||" + context.strip().lower()[:200]
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_get(self, key: str) -> Optional[tuple[float, str]]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: str, value: tuple[float, str]):
        if len(self._cache) >= self._cache_max:
            self._cache.popitem(last=False)
        self._cache[key] = value

    def entailment_score(self, claim: str, context: str) -> tuple[float, str]:
        cache_key = self._make_cache_key(claim, context)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        self._load_model()

        inputs = self._tokenizer(
            context, claim,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            entail_prob = probs[0, 2].item()
            contra_prob = probs[0, 0].item()

        if entail_prob > contra_prob and entail_prob > 0.5:
            label = "entailment"
            score = entail_prob
        elif contra_prob > entail_prob and contra_prob > 0.5:
            label = "contradiction"
            score = -contra_prob
        else:
            label = "neutral"
            score = entail_prob - contra_prob

        result = (score, label)
        self._cache_put(cache_key, result)
        return result

    def batch_verify(self, claims: list[str], contexts: list[str]) -> list[tuple[float, str]]:
        return [self.entailment_score(c, ctx) for c, ctx in zip(claims, contexts)]

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def cache_size(self) -> int:
        return len(self._cache)
