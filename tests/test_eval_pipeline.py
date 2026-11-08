import json
import pytest
from src.retrieval import hybrid_search, compute_bm25_ranking, CORPUS

@pytest.mark.asyncio
async def test_hybrid_retrieval_accuracy():
    with open("tests/golden_dataset.json", "r") as f:
        golden_cases = json.load(f)
        
    hits = 0
    total = len(golden_cases)
    
    for case in golden_cases:
        results = await hybrid_search(case["query"], top_k=2)
        retrieved_ids = [r["id"] for r in results]
        if case["expected_doc_id"] in retrieved_ids:
            hits += 1
            
    context_recall = (hits / total) * 100
    print(f"\n[EVALUATION BENCHMARK] Context Recall @ 2: {context_recall}% ({hits}/{total})")
    assert context_recall >= 75.0, f"Recall dropped below threshold: {context_recall}%"

def test_exact_error_code_bm25_precision():
    query = "ERR_TF_LOCK_423 error state"
    ranking = compute_bm25_ranking(query, CORPUS)
    top_doc_id = ranking[0][0]
    assert top_doc_id == "runbook_tf_02", f"BM25 failed on exact code lookup! Top doc: {top_doc_id}"
