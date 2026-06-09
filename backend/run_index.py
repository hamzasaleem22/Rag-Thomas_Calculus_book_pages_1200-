"""Index full book into Qdrant in-memory and cache BM25 state."""
import json, time, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from app.ingestion.indexer import index_documents_and_return_bm25

chunks = []
with open("data/markdown/parsed_docs.jsonl") as f:
    for line in f:
        data = json.loads(line)
        chunks.append(Document(page_content=data.get("page_content", ""), metadata=data.get("metadata", {})))
print(f"Loaded {len(chunks)} chunks", flush=True)

QDRANT_DB_PATH = "/tmp/qdrant_calculus_db"
client = QdrantClient(path=QDRANT_DB_PATH)

t0 = time.time()
count, bm25 = index_documents_and_return_bm25(chunks, client=client, batch_size=64)
elapsed = time.time() - t0
print(f"Indexed {count} vectors in {elapsed:.1f}s ({count/elapsed:.0f} chunks/s)", flush=True)

# Save BM25 model state so queries produce proper sparse vectors
bm25_state = {
    "vocab": bm25.vocab,
    "doc_freqs": bm25.doc_freqs,
    "num_docs": bm25.num_docs,
    "avg_dl": bm25.avg_dl,
    "doc_lengths": bm25.doc_lengths,
    "fitted": bm25.fitted,
    "k1": bm25.k1,
    "b": bm25.b,
}
with open("/tmp/qdrant_calculus_bm25.pkl", "wb") as f:
    pickle.dump(bm25_state, f)
print(f"Saved BM25 state ({len(bm25.vocab)} vocab, {bm25.num_docs} docs)", flush=True)

print(f"Qdrant DB persisted at: {QDRANT_DB_PATH}", flush=True)
