"""Index full book into Qdrant and cache."""
import json, time, pickle, sys
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from app.ingestion.indexer import index_documents

chunks = []
with open("data/markdown/parsed_docs.jsonl") as f:
    for line in f:
        data = json.loads(line)
        chunks.append(Document(page_content=data.get("page_content", ""), metadata=data.get("metadata", {})))
print(f"Loaded {len(chunks)} chunks")

client = QdrantClient(":memory:")
t0 = time.time()
count = index_documents(chunks, client=client, batch_size=64)
elapsed = time.time() - t0
print(f"Indexed {count} vectors in {elapsed:.1f}s ({count/elapsed:.0f} chunks/s)")

with open("/tmp/qdrant_calculus.pkl", "wb") as f:
    pickle.dump(client, f)
print("Cached to /tmp/qdrant_calculus.pkl")
