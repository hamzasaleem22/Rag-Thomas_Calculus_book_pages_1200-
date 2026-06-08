import json
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from app.config import settings
from app.retrieval.vector_store import HybridVectorStore
from app.retrieval.reranker import Reranker
from app.generation.generator import Generator
from app.api.schemas import QueryRequest, QueryResponse, Citation, IngestResponse

app = FastAPI(title="RAG Book - Thomas Calculus")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client: Optional[QdrantClient] = None
vector_store: Optional[HybridVectorStore] = None
reranker: Optional[Reranker] = None
generator: Optional[Generator] = None


class StreamQuery(BaseModel):
    query: str
    top_k: int = 0
    rerank: bool = True
    history: list[dict] = []


def get_vector_store() -> HybridVectorStore:
    global client, vector_store
    if vector_store is None:
        import pickle
        try:
            with open("/tmp/qdrant_calculus.pkl", "rb") as f:
                client = pickle.load(f)
        except FileNotFoundError:
            client = QdrantClient(":memory:")
        vector_store = HybridVectorStore(client=client)
    return vector_store


def get_reranker() -> Reranker:
    global reranker
    if reranker is None:
        reranker = Reranker()
    return reranker


def get_generator() -> Generator:
    global generator
    if generator is None:
        generator = Generator()
    return generator


def retrieve_and_rerank(query: str, top_k: int = 0, rerank: bool = True):
    store = get_vector_store()
    docs = store.similarity_search(query, k=top_k or settings.top_k_retrieve)

    if rerank and docs:
        r = get_reranker()
        texts = [d.page_content for d in docs]
        reranked = r.rerank(query, texts, top_k=settings.top_k_rerank)
        kept = {t for t, _ in reranked}
        docs = [d for d in docs if d.page_content in kept]

    return docs


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    docs = retrieve_and_rerank(req.query, req.top_k, req.rerank)

    if not docs:
        return QueryResponse(answer="No relevant information found.", citations=[], sources=[])

    gen = get_generator()
    result = gen.generate(req.query, docs)

    citations = [
        Citation(
            text=d.page_content[:300],
            page=d.metadata.get("page"),
            chapter=d.metadata.get("chapter"),
            section=d.metadata.get("section"),
        )
        for d in result["citations"]
    ]

    sources = list({str(d.metadata.get("page", "?")) for d in docs})

    return QueryResponse(answer=result["answer"], citations=citations, sources=sources)


@app.post("/query/stream")
async def query_stream(req: StreamQuery):
    docs = retrieve_and_rerank(req.query, req.top_k, req.rerank)

    if not docs:
        async def no_results():
            yield f"data: {json.dumps({'type': 'error', 'content': 'No relevant information found.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(no_results(), media_type="text/event-stream")

    gen = get_generator()

    async def event_stream():
        for token in gen.generate_stream(req.query, docs, req.history):
            if isinstance(token, dict):
                citations_data = [
                    {
                        "text": d.page_content[:300],
                        "page": d.metadata.get("page"),
                        "chapter": d.metadata.get("chapter"),
                        "section": d.metadata.get("section"),
                    }
                    for d in token.get("citations", [])
                ]
                yield f"data: {json.dumps({'type': 'citations', 'citations': citations_data})}\n\n"
                yield "data: [DONE]\n\n"
            else:
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/ingest", response_model=IngestResponse)
async def ingest():
    from langchain_core.documents import Document
    from app.ingestion.indexer import index_documents

    chunks = []
    with open(f"{settings.data_dir}/markdown/parsed_docs.jsonl") as f:
        for line in f:
            data = json.loads(line)
            chunks.append(Document(page_content=data.get("page_content", ""), metadata=data.get("metadata", {})))

    client_qdrant = get_vector_store().client
    count = index_documents(chunks, client=client_qdrant, batch_size=64)

    import pickle
    with open("/tmp/qdrant_calculus.pkl", "wb") as f:
        pickle.dump(client_qdrant, f)

    return IngestResponse(status="ok", chunks_indexed=count)
