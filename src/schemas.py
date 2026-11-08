from typing import List, Literal
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="User search query")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of context chunks to retrieve")

class DocumentChunk(BaseModel):
    id: str
    source: str
    content: str
    score: float = Field(default=0.0, description="RRF or similarity ranking score")

class RAGResponse(BaseModel):
    query: str
    retrieved_chunks: List[DocumentChunk]
    generated_answer: str
    hallucination_risk: Literal["Low", "Medium", "High"]
    latency_ms: float
    grounded_in_sources: bool
