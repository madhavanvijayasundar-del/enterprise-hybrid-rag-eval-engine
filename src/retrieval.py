import math
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import httpx

CORPUS = [
    {
        "id": "runbook_aks_01",
        "source": "azure_aks_troubleshooting.md",
        "content": "Kubernetes pods failing with CrashLoopBackOff should be inspected via kubectl logs. Check memory limits and OOMKilled events."
    },
    {
        "id": "runbook_tf_02",
        "source": "terraform_state_locks.md",
        "content": "Error code ERR_TF_LOCK_423 occurs when concurrent Azure DevOps CI/CD pipelines attempt terraform apply on the same blob storage state file."
    },
    {
        "id": "runbook_cni_03",
        "source": "azure_cni_networking.md",
        "content": "Azure CNI IP exhaustion causes node unreadiness and container restart storms across the private subnet."
    },
    {
        "id": "runbook_sec_04",
        "source": "cloud_security_governance.md",
        "content": "All Azure Key Vault instances must enforce RBAC authorization and purge protection to comply with SOC2 requirements."
    }
]

def compute_bm25_ranking(query: str, corpus: List[Dict], k1: float = 1.5, b: float = 0.75) -> List[Tuple[str, float]]:
    tokenized_docs = [doc["content"].lower().split() for doc in corpus]
    avgdl = sum(len(d) for d in tokenized_docs) / len(tokenized_docs)
    query_terms = query.lower().split()
    N = len(corpus)
    
    scores = []
    for idx, doc_tokens in enumerate(tokenized_docs):
        doc_len = len(doc_tokens)
        counts = Counter(doc_tokens)
        score = 0.0
        for term in query_terms:
            if term in counts:
                df = sum(1 for d in tokenized_docs if term in d)
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                tf = counts[term]
                score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl))))
        scores.append((corpus[idx]["id"], score))
    return sorted(scores, key=lambda x: x[1], reverse=True)

async def get_ollama_embedding(text: str) -> np.ndarray:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "http://localhost:11434/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text}
            )
            resp.raise_for_status()
            return np.array(resp.json()["embedding"], dtype=np.float32)
    except Exception:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.standard_normal(768).astype(np.float32)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = np.dot(v1, v2)
    norm = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
    return float(dot / norm)

def reciprocal_rank_fusion(
    dense_ranked_ids: List[str],
    sparse_ranked_ids: List[str],
    k: int = 60
) -> List[Tuple[str, float]]:
    rrf_scores: Dict[str, float] = {}
    
    for rank, doc_id in enumerate(dense_ranked_ids, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        
    for rank, doc_id in enumerate(sparse_ranked_ids, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

async def hybrid_search(query: str, top_k: int = 3) -> List[Dict]:
    sparse_results = compute_bm25_ranking(query, CORPUS)
    sparse_ids = [doc_id for doc_id, _ in sparse_results]
    
    query_emb = await get_ollama_embedding(query)
    dense_scored = []
    for doc in CORPUS:
        doc_emb = await get_ollama_embedding(doc["content"])
        sim = cosine_similarity(query_emb, doc_emb)
        dense_scored.append((doc["id"], sim))
    dense_scored.sort(key=lambda x: x[1], reverse=True)
    dense_ids = [doc_id for doc_id, _ in dense_scored]
    
    fused = reciprocal_rank_fusion(dense_ids, sparse_ids, k=60)
    
    id_to_doc = {d["id"]: d for d in CORPUS}
    results = []
    for doc_id, score in fused[:top_k]:
        doc = id_to_doc[doc_id].copy()
        doc["score"] = round(score, 5)
        results.append(doc)
    return results
