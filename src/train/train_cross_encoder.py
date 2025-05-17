import os
import json
import torch
import numpy as np
import hashlib
import tqdm
import shutil
import argparse
import logging
import kagglehub
import wandb
from sentence_transformers import (
    SentenceTransformer,
    losses,
)
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderModelCardData,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.evaluation import (
    CrossEncoderNanoBEIREvaluator,
    CrossEncoderRerankingEvaluator,
)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss, CachedMultipleNegativesRankingLoss
from sentence_transformers.evaluation import SequentialEvaluator
from sentence_transformers.util import mine_hard_negatives

from torch.utils.data import DataLoader
from accelerate import Accelerator
from datetime import datetime
from datasets import Dataset, load_from_disk

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

parser = argparse.ArgumentParser(description="Cross-Encoder Training Script")
parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation")
parser.add_argument("--model_name", type=str, default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", help="Pretrained model name")
parser.add_argument("--data_path", type=str, default="/root/training/datasets/cross_encoder_dataset_finetuned_bert_base", help="Path to dataset")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 precision")
parser.add_argument("--bf16", action="store_true", default=True, help="Use BF16 precision")
parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of dataloader workers")
parser.add_argument("--eval_steps", type=int, default=2000, help="Evaluation frequency (in steps)")
parser.add_argument("--save_steps", type=int, default=2000, help="Model save frequency (in steps)")
parser.add_argument("--save_total_limit", type=int, default=2, help="Maximum number of checkpoints to save")
parser.add_argument("--logging_steps", type=int, default=200, help="Logging frequency (in steps)")
parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the model")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

args = parser.parse_args()

# TRAINING AND LOGGING CONFIGS

NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
MODEL_NAME = args.model_name
DATA_PATH = args.data_path
OUTPUT_DIR = args.output_dir
LOGGING_STEPS = args.logging_steps
EVAL_STEPS = args.eval_steps
SAVE_STEPS = args.save_steps
SAVE_LIMIT = args.save_total_limit
DATALOADER_NUM_WORKERS = args.dataloader_num_workers
GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
LEARNING_RATE = args.learning_rate
WARMUP_RATIO = args.warmup_ratio
FP16 = args.fp16
BF16 = args.bf16
SEED = args.seed
total_data = load_from_disk(DATA_PATH)

split_dataset = total_data.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

WANDB_KEY = os.environ["WANDB_KEY"]
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# LOAD MODEL

model = CrossEncoder(
    MODEL_NAME,
    model_card_data=CrossEncoderModelCardData(
        language="en",
        license="apache-2.0",
    ),
    tokenizer_kwargs={"truncation": True, "padding": True},
    device = "cuda:0"
)

# INIT WANDB LOGGING

wandb.login(key=WANDB_KEY)
run_name = f"{MODEL_NAME}-{DATA_PATH.split('/')[-1]}-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
wandb.init(
    project=f"biomed-ir-synthesis-reranker",
    name=run_name,
)
# LOSS (MNRL)
loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size = 32)

# EVALUATOR
nano_beir_evaluator = CrossEncoderNanoBEIREvaluator(
    dataset_names=["nfcorpus", "scidocs", "scifact"],
    batch_size=BATCH_SIZE,
)
reranking_evaluator = CrossEncoderRerankingEvaluator(
    samples=[
        {
            "query": sample["query"],
            "positive": [sample["document"]],
            "negative": [sample[column_name] for column_name in eval_dataset.column_names[2:]],
        }
        for sample in eval_dataset
    ],
    batch_size=BATCH_SIZE,
    name="dev",
    always_rerank_positives=False,
)

evaluator = SequentialEvaluator([reranking_evaluator, nano_beir_evaluator])

args = CrossEncoderTrainingArguments(
    output_dir=f'{OUTPUT_DIR}/{run_name}',
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    fp16=FP16, 
    bf16=BF16,
    dataloader_num_workers=DATALOADER_NUM_WORKERS,
    load_best_model_at_end=True,
    metric_for_best_model="dev_ndcg@10",
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_LIMIT,
    logging_steps=LOGGING_STEPS,
    logging_first_step=True,
    run_name=run_name, 
    seed=SEED,
)

trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
    evaluator=evaluator,
)
trainer.train()