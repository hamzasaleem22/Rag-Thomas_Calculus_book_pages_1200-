import math
import re
from collections import Counter
from typing import Optional

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_qdrant.sparse_embeddings import SparseEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from app.config import settings
from app.retrieval.embeddings import HFInferenceAPIEmbeddings


class BM25SparseEmbeddings(SparseEmbeddings):
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: dict[int, int] = {}
        self.num_docs = 0
        self.avg_dl = 0.0
        self.doc_lengths: list[int] = []
        self.vocab: dict[str, int] = {}
        self.doc_count = 0
        self.fitted = False

    @classmethod
    def from_pickle(cls, path: str = "/tmp/qdrant_calculus_bm25.pkl") -> "BM25SparseEmbeddings":
        import pickle, os
        if not os.path.exists(path):
            print(f"BM25 state not found at {path}, using empty BM25")
            return cls()
        with open(path, "rb") as f:
            state = pickle.load(f)
        bm25 = cls(k1=state.get("k1", 1.5), b=state.get("b", 0.75))
        bm25.vocab = state.get("vocab", {})
        bm25.doc_freqs = {int(k): v for k, v in state.get("doc_freqs", {}).items()}
        bm25.num_docs = state.get("num_docs", 0)
        bm25.avg_dl = state.get("avg_dl", 0.0)
        bm25.doc_lengths = state.get("doc_lengths", [])
        bm25.fitted = state.get("fitted", False)
        print(f"Loaded BM25 state: {len(bm25.vocab)} vocab, {bm25.num_docs} docs, fitted={bm25.fitted}")
        return bm25

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _build_vocab_and_stats(self, texts: list[str]):
        doc_count = len(texts)
        total_terms = 0
        for text in texts:
            tokens = self._tokenize(text)
            seen = set()
            for t in tokens:
                if t not in self.vocab:
                    self.vocab[t] = len(self.vocab)
                idx = self.vocab[t]
                if idx not in self.doc_freqs:
                    self.doc_freqs[idx] = 0
                if t not in seen:
                    self.doc_freqs[idx] += 1
                    seen.add(t)
            self.doc_lengths.append(len(tokens))
            total_terms += len(tokens)
        self.num_docs = doc_count
        self.avg_dl = total_terms / doc_count if doc_count else 0
        self.fitted = True

    def _compute_bm25(self, tokens: list[str]) -> tuple[list[int], list[float]]:
        if not self.fitted:
            return [], []
        count = Counter(tokens)
        indices = []
        values = []
        if not self.num_docs:
            return [], []

        for token, term_freq in count.items():
            idx = self.vocab.get(token)
            if idx is None:
                continue
            df = self.doc_freqs.get(idx, 0)
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
            dl = len(tokens)
            score = idf * ((term_freq * (self.k1 + 1)) / (term_freq + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)))
            indices.append(idx)
            values.append(score)

        return indices, values

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        if not self.fitted and texts:
            self._build_vocab_and_stats(texts)

        results = []
        for text in texts:
            tokens = self._tokenize(text)
            indices, values = self._compute_bm25(tokens)
            results.append(SparseVector(indices=indices, values=values))
        return results

    def embed_query(self, text: str) -> SparseVector:
        tokens = self._tokenize(text)
        indices, values = self._compute_bm25(tokens)
        return SparseVector(indices=indices, values=values)


class HybridVectorStore:
    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        collection: str = "",
        embedding: Optional[HFInferenceAPIEmbeddings] = None,
        bm25_path: str = "/tmp/qdrant_calculus_bm25.pkl",
    ):
        self.collection = collection or settings.qdrant_collection
        self.client = client or QdrantClient(":memory:")
        self.embedding = embedding or HFInferenceAPIEmbeddings()
        self.sparse_embedding = BM25SparseEmbeddings.from_pickle(bm25_path)
        self._store: Optional[QdrantVectorStore] = None

    @property
    def store(self) -> QdrantVectorStore:
        if self._store is None:
            self._store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection,
                embedding=self.embedding,
                sparse_embedding=self.sparse_embedding,
                retrieval_mode=RetrievalMode.HYBRID,
                validate_collection_config=False,
            )
        return self._store

    def add_documents(self, documents: list[Document]) -> list[str]:
        texts = [doc.page_content for doc in documents]
        self.sparse_embedding.embed_documents(texts)
        return self.store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 0, chapter_filter: Optional[list[int]] = None, **kwargs) -> list[Document]:
        k = k or settings.top_k_retrieve
        if chapter_filter:
            from qdrant_client.models import Filter, FieldCondition, MatchAny
            from langchain_qdrant import RetrievalMode

            original_mode = self.store.retrieval_mode
            self.store.retrieval_mode = RetrievalMode.HYBRID

            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.chapter_number",
                        match=MatchAny(any=chapter_filter),
                    )
                ]
            )
            results = self.store.similarity_search(
                query=query, k=k, filter=qdrant_filter, **kwargs
            )
            self.store.retrieval_mode = original_mode
            return results
        return self.store.similarity_search(query=query, k=k, **kwargs)

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 0, chapter_filter: Optional[list[int]] = None, **kwargs
    ) -> list[tuple[Document, float]]:
        k = k or settings.top_k_retrieve
        if chapter_filter:
            from qdrant_client.models import Filter, FieldCondition, MatchAny

            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.chapter_number",
                        match=MatchAny(any=chapter_filter),
                    )
                ]
            )
            return self.store.similarity_search_with_relevance_scores(
                query=query, k=k, filter=qdrant_filter, **kwargs
            )
        return self.store.similarity_search_with_relevance_scores(query=query, k=k, **kwargs)
