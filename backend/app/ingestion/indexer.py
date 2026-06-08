from typing import Optional
from uuid import uuid4

from langchain_core.documents import Document
from qdrant_client import QdrantClient

from app.config import settings
from app.retrieval.embeddings import HFInferenceAPIEmbeddings
from app.retrieval.vector_store import BM25SparseEmbeddings


def index_documents(
    chunks: list[Document],
    client: Optional[QdrantClient] = None,
    collection: str = "",
    batch_size: int = 64,
) -> int:
    count, _ = index_documents_and_return_bm25(chunks, client, collection, batch_size)
    return count


def index_documents_and_return_bm25(
    chunks: list[Document],
    client: Optional[QdrantClient] = None,
    collection: str = "",
    batch_size: int = 64,
) -> tuple[int, BM25SparseEmbeddings]:
    from qdrant_client import QdrantClient as QC
    from qdrant_client.models import Distance, PointStruct, SparseVectorParams, VectorParams

    collection = collection or settings.qdrant_collection
    client = client or QC(":memory:")

    collections = client.get_collections().collections
    existing = {c.name for c in collections}

    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            sparse_vectors_config={
                "langchain-sparse": SparseVectorParams(),
            },
        )

    embedding = HFInferenceAPIEmbeddings()
    sparse_embedding = BM25SparseEmbeddings()

    texts = [chunk.page_content for chunk in chunks]
    print(f"Embedding {len(texts)} texts with {settings.embedding_model}...")
    dense_vectors = embedding.embed_documents(texts)

    print("Computing BM25 sparse vectors...")
    sparse_vectors = sparse_embedding.embed_documents(texts)

    indexed = 0
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_dense = dense_vectors[i : i + batch_size]
        batch_sparse = sparse_vectors[i : i + batch_size]

        points = []
        for chunk, dense, sparse in zip(batch_chunks, batch_dense, batch_sparse):
            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector={
                        "": dense,
                        "langchain-sparse": sparse,
                    },
                    payload={
                        "page_content": chunk.page_content,
                        "metadata": chunk.metadata,
                    },
                )
            )

        client.upsert(collection_name=collection, points=points)
        indexed += len(points)

    return indexed, sparse_embedding
