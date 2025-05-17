import gradio as gr
import os
import json
import heapq

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.reranking import Rerank
from beir.reranking.models import CrossEncoder
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.evaluation import EvaluateRetrieval
from src.evaluate.utils import reciprocal_rank_fusion  # Make sure this utility is accessible

# Configuration
TOP_K = 10
DATASET_NAME = "nfcorpus"
DENSE_MODEL_PATH = "./ckpts/biencoder-checkpoints/checkpoint-pubmedbert"
CROSSENCODER_MODEL_PATH = "./ckpts/crossencoder-checkpoints/checkpoint-pubmedbert"

# Load models
dense_model = models.SentenceBERT(DENSE_MODEL_PATH)
cross_encoder = CrossEncoder(CROSSENCODER_MODEL_PATH)
reranker = Rerank(cross_encoder, batch_size=64)
bm25_model = BM25(index_name=DATASET_NAME, hostname="localhost", initialize=False)
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET_NAME}.zip"
out_dir = os.path.join(".", "datasets")
data_path = util.download_and_unzip(url, out_dir)
# Load dataset
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# Invert corpus for metadata display
corpus_dict = {doc_id: doc["text"] for doc_id, doc in corpus.items()}

# Gradio search function
def search_biomedical_docs(user_query):
    # Retrieval: BM25 + Dense + RRF
    dense_retriever = EvaluateRetrieval(DRES(dense_model), score_function="cos_sim", k_values=[TOP_K])
    dense_results = dense_retriever.retrieve(corpus, {0: user_query})
    bm25_results = bm25_model.search(corpus, {0: user_query}, TOP_K, "dot")

    rrf_results = reciprocal_rank_fusion([bm25_results, dense_results])

    # Restrict to top_k from RRF
    top_docs = list(heapq.nlargest(TOP_K, rrf_results[0].items(), key=lambda x: x[1]))
    top_doc_ids = [doc_id for doc_id, _ in top_docs]
    new_corpus = {doc_id: corpus[doc_id] for doc_id in top_doc_ids}

    # Rerank
    rerank_results = reranker.rerank(new_corpus, {0: user_query}, {0: {doc_id: 1.0 for doc_id in new_corpus}}, top_k=TOP_K)
    reranked_doc_ids = [doc_id for doc_id, _ in sorted(rerank_results[0].items(), key=lambda x: x[1], reverse=True)]

    # Return top results
    return [f"• {corpus_dict[doc_id][:300]}..." for doc_id in reranked_doc_ids]

# Launch Gradio
demo = gr.Interface(
    fn=search_biomedical_docs,
    inputs=gr.Textbox(placeholder="Enter a biomedical query...", label="Query"),
    outputs=gr.outputs.Textbox(label="Top Retrieved Passages"),
    title="Biomedical IR Demo (PubMedBERT)",
    description=f"Hybrid search (BM25 + Dense + Cross-Encoder) over '{DATASET_NAME}' using your fine-tuned models."
)

if __name__ == "__main__":
    demo.launch(share=True)