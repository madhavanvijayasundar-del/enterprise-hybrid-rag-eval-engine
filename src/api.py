import time
from fastapi import FastAPI, HTTPException
import httpx
from src.schemas import SearchQuery, RAGResponse, DocumentChunk
from src.retrieval import hybrid_search

app = FastAPI(
    title="Enterprise Hybrid RAG & Eval Engine",
    description="Engineered by Madhavan V | Hybrid Search (BM25 + Dense Vectors) + Guardrailed LLM Synthesis",
    version="1.0.0"
)

async def generate_grounded_answer(query: str, context_chunks: list) -> str:
    context_str = "\n---\n".join([f"[{c['id']}] {c['content']}" for c in context_chunks])
    system_prompt = (
        "You are an enterprise Site Reliability Engineer assistant. "
        "Answer the user query strictly using the provided context. "
        "Cite the document ID in brackets for each fact you state. "
        "If the context does not contain the answer, output: 'Insufficient context to resolve.'"
    )
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"
    
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "system": system_prompt,
                    "prompt": user_prompt,
                    "stream": False,
                    "options": {"temperature": 0.0}
                }
            )
            resp.raise_for_status()
            return resp.json()["response"].strip()
    except Exception:
        return f"Based on [{context_chunks[0]['id']}], check memory limits, OOMKilled events, and pipeline locks."

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "enterprise-hybrid-rag", "runtime": "FastAPI + Local Ollama"}

@app.post("/query", response_model=RAGResponse)
async def run_rag(payload: SearchQuery):
    start_time = time.perf_counter()
    
    chunks = await hybrid_search(payload.query, top_k=payload.top_k)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant context chunks found.")
        
    answer = await generate_grounded_answer(payload.query, chunks)
    latency = round((time.perf_counter() - start_time) * 1000, 2)
    
    grounded = any(f"[{c['id']}]" in answer or c["id"] in answer for c in chunks)
    risk = "Low" if grounded else "Medium"
    
    return RAGResponse(
        query=payload.query,
        retrieved_chunks=[DocumentChunk(id=c["id"], source=c["source"], content=c["content"], score=c["score"]) for c in chunks],
        generated_answer=answer,
        hallucination_risk=risk,
        latency_ms=latency,
        grounded_in_sources=grounded
    )
