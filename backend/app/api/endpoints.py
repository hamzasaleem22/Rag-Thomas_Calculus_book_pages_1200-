import json
import re
import asyncio
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from qdrant_client import QdrantClient
from app.config import settings
from app.retrieval.vector_store import HybridVectorStore
from app.retrieval.reranker import Reranker
from app.retrieval.expansion import expand_query_text
from app.generation.generator import Generator
from app.api.schemas import QueryRequest, QueryResponse, Citation, IngestResponse
from app.retrieval.citation_verifier import verify_all_citations

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



def get_vector_store() -> HybridVectorStore:
    global client, vector_store
    if vector_store is None:
        from pathlib import Path
        qdrant_db_path = "/tmp/qdrant_calculus_db"
        if Path(qdrant_db_path).exists():
            client = QdrantClient(path=qdrant_db_path)
        else:
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


def _compute_query_doc_relevance(query: str, docs: list, threshold: Optional[float] = None) -> float:
    query_tokens = set(re.findall(r'\b[a-zA-Z]\w+\b', query.lower()))
    stopwords = {'the', 'is', 'at', 'which', 'what', 'how', 'do', 'does', 'a', 'an',
                 'and', 'or', 'of', 'to', 'for', 'in', 'on', 'by', 'with', 'from', 'its'}
    query_tokens -= stopwords
    if not query_tokens:
        return 1.0
    all_doc_text = " ".join(d.page_content.lower() for d in docs[:3])
    matched = sum(1 for t in query_tokens if t in all_doc_text)
    return matched / len(query_tokens)


def retrieve_and_rerank(query: str, top_k: int = 0, rerank: bool = True, use_mmr: bool = False):
    from app.retrieval.expansion import detect_query_type, detect_chapter_reference, decompose_multi_hop, is_multi_hop_query

    store = get_vector_store()
    query_type = detect_query_type(query)
    chapter_refs = detect_chapter_reference(query)

    # Multi-hop decomposition
    sub_queries = decompose_multi_hop(query)
    is_multi = len(sub_queries) > 1

    if is_multi:
        print(f"  Multi-hop: {len(sub_queries)} sub-queries")
        all_docs = []
        seen_content = set()
        for sq in sub_queries:
            sq_docs = _retrieve_single(sq, store, top_k, chapter_refs, query_type)
            for d in sq_docs:
                content_hash = hash(d.page_content[:300])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(d)
        docs = all_docs
        print(f"  Multi-hop merged: {len(docs)} docs (deduped)")
    else:
        docs = _retrieve_single(query, store, top_k, chapter_refs, query_type)

    if not docs:
        return []

    if not settings.use_reranker:
        rerank = False

    if rerank and docs:
        r = get_reranker()
        texts = [d.page_content for d in docs]
        expanded = expand_query_text(query)

        if use_mmr:
            reranked = r.rerank_with_mmr(
                expanded, texts,
                top_k=settings.top_k_rerank,
                lambda_mmr=settings.mmr_lambda,
            )
        else:
            reranked = r.rerank(expanded, texts, top_k=settings.top_k_rerank, method=settings.reranker_method)

        kept_indices = set()
        for t, _ in reranked:
            for idx, d in enumerate(docs):
                if d.page_content == t and idx not in kept_indices:
                    kept_indices.add(idx)
                    break
        docs = [docs[i] for i in sorted(kept_indices)]

    if docs and settings.use_compression:
        from app.retrieval.compression import compress_documents
        docs = compress_documents(query, docs)

    return docs


def _retrieve_single(query: str, store, top_k: int = 0, chapter_refs: list[int] | None = None, query_type: str = "general"):
    from app.retrieval.expansion import expand_query_text

    retrieval_query = query
    hyde_doc = None

    use_hyde_for_query = settings.use_hyde or query_type == "equation"
    if use_hyde_for_query:
        from app.retrieval.hyde import generate_hypothetical_document
        hyde_doc = generate_hypothetical_document(query)
        retrieval_query = query + " " + hyde_doc
        print(f"  HyDE ({query_type}): query + {len(hyde_doc)} chars")

    expanded = expand_query_text(retrieval_query)

    docs = store.similarity_search(
        expanded,
        k=top_k or settings.top_k_retrieve,
        chapter_filter=chapter_refs if chapter_refs else None,
    )

    if use_hyde_for_query and docs and hyde_doc:
        hyde_results = store.similarity_search(
            hyde_doc,
            k=top_k or settings.top_k_retrieve,
            chapter_filter=chapter_refs if chapter_refs else None,
        )
        seen_ids = {id(d) for d in docs}
        for d in hyde_results:
            if id(d) not in seen_ids:
                docs.append(d)
                seen_ids.add(id(d))

    if docs and query.strip():
        adaptive_threshold = settings.relevance_threshold
        if query_type == "equation":
            adaptive_threshold = 0.05
        elif query_type == "definition":
            adaptive_threshold = 0.10
        relevance = _compute_query_doc_relevance(query, docs, adaptive_threshold)
        if relevance < adaptive_threshold:
            print(f"  Low relevance ({relevance:.2f} < {adaptive_threshold}), refusing generation")
            return []

    return docs


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    loop = asyncio.get_event_loop()
    docs = await loop.run_in_executor(None, retrieve_and_rerank, req.query, req.top_k, req.rerank, req.use_mmr)

    if not docs:
        refusal_msg = "I'm sorry, I can only answer questions about calculus from Thomas' Calculus, 14th Edition. Your question doesn't appear to be covered in this textbook."
        return QueryResponse(answer=refusal_msg, citations=[], sources=[])

    gen = get_generator()
    result = gen.generate(req.query, docs)

    answer = result["answer"]
    verification = verify_all_citations(answer, docs)
    cited_docs = result["citations"]

    citations = []
    for d in cited_docs:
        doc_idx = None
        for i, doc in enumerate(docs):
            if doc.page_content == d.page_content:
                doc_idx = i
                break
        confidence = 1.0
        if doc_idx is not None and doc_idx in verification["results"]:
            claim_results = verification["results"][doc_idx]["claims"]
            if claim_results:
                confidence = sum(c["composite_score"] for c in claim_results) / len(claim_results)

        citations.append(Citation(
            text=d.page_content[:300],
            page=d.metadata.get("page"),
            chapter=d.metadata.get("chapter"),
            section=d.metadata.get("section"),
            chunk_id=d.metadata.get("chunk_id"),
            confidence_score=round(confidence, 3),
        ))

    sources = list({str(d.metadata.get("page", "?")) for d in docs})

    claim_verification = result.get("claim_verification", [])

    return QueryResponse(
        answer=result["answer"],
        citations=citations,
        sources=sources,
        claim_verification=claim_verification,
    )


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    loop = asyncio.get_event_loop()
    docs = await loop.run_in_executor(None, retrieve_and_rerank, req.query, req.top_k, req.rerank, req.use_mmr)

    if not docs:
        refusal_msg = "I'm sorry, I can only answer questions about calculus from Thomas' Calculus, 14th Edition. Your question doesn't appear to be covered in this textbook."
        async def no_results():
            yield f"data: {json.dumps({'type': 'token', 'content': refusal_msg})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(no_results(), media_type="text/event-stream")

    gen = get_generator()

    async def event_stream():
        for token in gen.generate_stream(req.query, docs, req.history):
            if isinstance(token, dict):
                answer_text = ""
                citations_data = []
                cited_docs = token.get("citations", [])

                # Build verification on the final answer
                # Note: during streaming we don't have the full answer yet,
                # so we estimate confidence from chunk metadata
                for d in cited_docs:
                    doc_idx = None
                    for i, doc in enumerate(docs):
                        if doc.page_content == d.page_content:
                            doc_idx = i
                            break
                    # Use equation_density as proxy for content richness
                    eq_density = d.metadata.get("equation_density", 0.0)
                    confidence = min(1.0, 0.7 + eq_density * 2)

                    citations_data.append({
                        "text": d.page_content[:300],
                        "page": d.metadata.get("page"),
                        "chapter": d.metadata.get("chapter"),
                        "section": d.metadata.get("section"),
                        "chunk_id": d.metadata.get("chunk_id"),
                        "confidence_score": round(confidence, 3),
                    })

                yield f"data: {json.dumps({'type': 'citations', 'citations': citations_data})}\n\n"
                yield "data: [DONE]\n\n"
            else:
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/ingest", response_model=IngestResponse)
async def ingest():
    from langchain_core.documents import Document
    from app.ingestion.indexer import index_documents_and_return_bm25
    import pickle

    chunks = []
    with open(f"{settings.data_dir}/markdown/parsed_docs.jsonl") as f:
        for line in f:
            data = json.loads(line)
            chunks.append(Document(page_content=data.get("page_content", ""), metadata=data.get("metadata", {})))

    client_qdrant = get_vector_store().client
    count, bm25 = index_documents_and_return_bm25(chunks, client=client_qdrant, batch_size=64)

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

    return IngestResponse(status="ok", chunks_indexed=count)


@app.get("/metrics")
async def metrics():
    store = get_vector_store()
    try:
        count_result = store.client.count(collection_name=settings.qdrant_collection)
        chunk_count = count_result.count
    except Exception:
        chunk_count = 0
    return {
        "chunk_count": chunk_count,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
        "reranker_model": settings.reranker_model,
        "top_k_retrieve": settings.top_k_retrieve,
        "top_k_rerank": settings.top_k_rerank,
        "use_mmr": settings.mmr_lambda,
        "use_hyde": settings.use_hyde,
        "use_self_consistency": settings.use_self_consistency,
        "request_delay": settings.request_delay,
    }
