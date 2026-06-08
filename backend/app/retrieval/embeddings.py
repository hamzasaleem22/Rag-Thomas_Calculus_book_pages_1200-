from typing import Optional

import numpy as np
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from app.config import settings


_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(
            model_name, trust_remote_code=True
        )
    return _model_cache[model_name]


class HFInferenceAPIEmbeddings(Embeddings):
    def __init__(self, model: str = ""):
        self.model_name = model or settings.embedding_model
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = _get_model(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
        )
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        )
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        return embedding
