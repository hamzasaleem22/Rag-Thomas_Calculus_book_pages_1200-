from typing import Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    top_k: int = 0
    rerank: bool = True


class Citation(BaseModel):
    text: str
    page: Optional[int] = None
    chapter: Optional[str] = None
    section: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    sources: list[str]


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
