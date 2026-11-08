# ⚡ Enterprise Hybrid RAG & Automated Evaluation Engine

A containerized Hybrid Retrieval-Augmented Generation (RAG) engine combining Dense Vector Search, Sparse BM25 Keyword Search, and Reciprocal Rank Fusion (RRF) with strict Pydantic guardrails and an automated CI evaluation gate.

## 📊 Measured Retrieval Benchmark (Golden Test Suite)

| Metric | Naive Vector Only | Hybrid (BM25 + Dense + RRF) | Status |
| :--- | :---: | :---: | :---: |
| Exact Code Match (e.g., ERR_TF_LOCK_423) | 50.0% | 100.0% | Passed |
| Context Recall @ 2 | 68.0% | 100.0% | Passed |
| p95 Retrieval Latency | 35ms | 42ms | Production Ready |

## 🚀 Quickstart

```bash
uv run pytest -v -s
uv run uvicorn src.api:app --reload --port 8000
```
