import argparse 
import os
import logging
import pathlib
import numpy as np

from src.evaluate.utils import *
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.reranking import Rerank
from beir.reranking.models import CrossEncoder
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer

score_function = "cos_sim"
hostname = "localhost"
initialize = True # True, will delete existing index with same name and reindex all documents


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate hybrid retrieval with BEIR")

    parser.add_argument(
        "--top_k", 
        type=int, 
        default=10,
        help="Number of top documents to retrieve and rerank"
    )

    parser.add_argument(
        "--biencoder_model_name", 
        type=str, 
        default="pubmedbert",
        help="Name of the SentenceTransformer (bi-encoder) model (inside biomed-ir-synthesis/ckpts/)"
    )

    parser.add_argument(
        "--crossencoder_model_name", 
        type=str, 
        default="pubmedbert",
        help="Name of the CrossEncoder (reranker) model (inside biomed-ir-synthesis/ckpts/)"
    )

    return parser.parse_args()

def evaluate(
    dense_model_name = "checkpoint-pubmedbert", 
    reranker_model_name = "checkpoint-pubmedbert-10000", 
    top_k = 50
):
    datasets = ['trec-covid', 'nfcorpus', 'scifact', 'scidocs']
    dense_model_path = f"./ckpts/biencoder-checkpoints/checkpoint-{dense_model_name}"
    reranker_model_path = f"./ckpts/crossencoder-checkpoints/checkpoint-{reranker_model_name}"
    dense_model = models.SentenceBERT(dense_model_path)
    reranker = Rerank(CrossEncoder(reranker_model_path), batch_size=32)
    for dataset in datasets :
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        out_dir = os.path.join(".", "datasets")
        data_path = util.download_and_unzip(url, out_dir)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

        bm25_model = BM25(index_name=dataset, hostname=hostname, initialize=initialize)
        dense_retriever = EvaluateRetrieval(DRES(dense_model), score_function= score_function, k_values=[top_k])
        dense_results = dense_retriever.retrieve(corpus, queries)
        bm25_results = bm25_model.search(corpus, queries, top_k, score_function)

        rrf_results = reciprocal_rank_fusion([bm25_results, dense_results])
        new_corpus = {}
        for query_id in rrf_results:
            if len(rrf_results[query_id]) > top_k:
                for doc_id, _ in sorted(rrf_results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]:
                    new_corpus[doc_id] = corpus[doc_id]
            else:
                for doc_id in rrf_results[query_id]:
                    new_corpus[doc_id] = corpus[doc_id]
        ce_results = reranker.rerank(new_corpus, queries, rrf_results, top_k=top_k)
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, ce_results, [top_k])
        mrr = EvaluateRetrieval.evaluate_custom(qrels, ce_results, metric="mrr", k_values=[top_k])
        
        results_dir = os.path.join(".", "results")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, f"{dense_model_name}-retriever-{reranker_model_name}-reranker.txt"), "w") as fw:
            fw.write(f"NDCG@{top_k}: {ndcg[f'NDCG@{top_k}']}\n")
            fw.write(f"MAP@{top_k}: {_map[f'MAP@{top_k}']}\n")
            fw.write(f"Recall@{top_k}: {recall[f'Recall@{top_k}']}\n")
            fw.write(f"Precision@{top_k}: {precision[f'P@{top_k}']}\n")
            fw.write(f"MRR@{top_k}: {mrr[f'MRR@{top_k}']}\n")
        print(f"NDCG@{top_k}: {ndcg[f'NDCG@{top_k}']}")
        print(f"MAP@{top_k}: {_map[f'MAP@{top_k}']}")
        print(f"Recall@{top_k}: {recall[f'Recall@{top_k}']}")
        print(f"Precision@{top_k}: {precision[f'P@{top_k}']}")
        print(f"MRR@{top_k}: {mrr[f'MRR@{top_k}']}")

        with open(os.path.join(results_dir, f"{dataset}-ce-top{top_k}.txt"), "w") as f:
            for query_id in ce_results:
                for doc_id, score in sorted(ce_results[query_id].items(), key=lambda x: -x[1]):
                    f.write(f"{query_id}\t{doc_id}\t{score}\n")

if __name__ == "__main__":
    args = parse_args()
    evaluate(args.biencoder_model_name, args.crossencoder_model_name, args.top_k)