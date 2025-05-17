def reciprocal_rank_fusion(results_list, k=60):
    fused = {}
    for results in results_list:
        for query_id, doc_scores in results.items():
            fused.setdefault(query_id, {})
            for rank, (doc_id, _) in enumerate(sorted(doc_scores.items(), key=lambda x: -x[1])):
                fused[query_id].setdefault(doc_id, 0)
                fused[query_id][doc_id] += 1 / (k + rank)
    return fused