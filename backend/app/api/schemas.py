from typing import Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    top_k: int = 0
    rerank: bool = True
    use_mmr: bool = False
    history: list[dict] = []


class Citation(BaseModel):
    text: str
    page: Optional[int] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    chunk_id: Optional[int] = None
    confidence_score: Optional[float] = None


class ClaimConfidence(BaseModel):
    doc_idx: int
    composite_score: float
    nli_label: str
    verified: bool
    is_core: bool = False
    confidence: float = 0.0
    abstain: bool = False


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    sources: list[str]
    claim_verification: list[ClaimConfidence] = []
    average_confidence: float = 0.0
    abstain_rate: float = 0.0


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
