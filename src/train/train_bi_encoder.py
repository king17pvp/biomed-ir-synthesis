import os
import json
import hashlib
import tqdm
import logging
import wandb
import argparse

from sentence_transformers import SentenceTransformer, InputExample, models, losses, LoggingHandler
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from accelerate import Accelerator
from datetime import datetime

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
parser = argparse.ArgumentParser(description="Training Bi-encoder")
parser.add_argument("--model_name", type=str, default="nlpie/tiny-biobert", help="Name or path of the pretrained model")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation")
parser.add_argument("--threshold", type=float, default=5.0, help="Score threshold for filtering examples")
parser.add_argument("--dataset_path", type=str, default="/kaggle/input/uslme-data", help="Path to dataset folder")
parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps for training")
parser.add_argument("--eval_steps", type=int, default=1000, help="Evaluation steps")
parser.add_argument("--output_path", type=str, default="./biobert-ir-model", help="Path to save the trained model")
parser.add_argument("--checkpoint_path", type=str, default="./checkpoints", help="Path to save intermediate checkpoints")
parser.add_argument("--save_steps", type=int, default=1000, help="Steps interval to save checkpoints")
parser.add_argument("--save_total_limit", type=int, default=3, help="Max number of checkpoints to keep")
args = parser.parse_args()

# CONFIGS
WANDB_KEY = os.environ["WANDB_KEY"]
MODEL_NAME = args.model_name
BATCH_SIZE = args.batch_size
THRESHOLD = args.threshold
DATA_PATH = args.dataset_path
NUM_EPOCHS = args.num_epochs
OUTPUT_PATH = args.output_path
CHECKPOINT_PATH = args.checkpoint_path
SAVE_STEPS = args.save_steps
SAVE_LIMIT = args.save_total_limit
WARMUP_STEPS = args.warmup_steps
EVAL_STEPS = args.eval_steps

# LOAD DATA
with open(f'{DATA_PATH}/indexed_data.jsonl', 'r', encoding='utf8') as fr:
    total_datas = [json.loads(line) for line in tqdm.tqdm(fr)]
with open(f'{DATA_PATH}/scores_eval.json', 'r') as fr:
    data = json.load(fr)
print("Total data before being filtered:", len(total_datas))

filtered_ids = [k for k in data.keys() if data[k] > THRESHOLD]
total_datas = [x for x in total_datas if data[x['id']] > THRESHOLD]
# print(json.dumps(total_datas[0], indent = 2))
print("Total data after filtered:", len(total_datas))

train_samples = []
id_map = {}
for data in tqdm.tqdm(total_datas):
    query, doc = data['query'].lower(), data['document'].lower()
    id_ = "_".join(data['id'].split("_")[:-1])
    train_samples.append((id_, InputExample(texts = [query, doc])))

# LOAD MODEL
word_embedding_model = models.Transformer(MODEL_NAME)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
train_loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size = BATCH_SIZE)

train_samples_split, eval_samples_split = train_test_split(
    train_samples, test_size=0.1, random_state=42
)
train_samples_split = [x[1] for x in train_samples_split]
train_dataloader = DataLoader(train_samples_split, shuffle=True, batch_size=BATCH_SIZE)
def hash_text(text):
    return hashlib.md5(text.encode()).hexdigest()

queries = {}
corpus = {}
relevant_docs = {}

for id_, example in tqdm.tqdm(eval_samples_split):
    query_text = example.texts[0]
    doc_text = example.texts[1]
    
    # query_id = hash_text(query_text)
    doc_id = hash_text(doc_text)

    queries[id_] = query_text
    corpus[doc_id] = doc_text

    if id_ not in relevant_docs:
        relevant_docs[id_] = set()
    relevant_docs[id_].add(doc_id)
    
evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    show_progress_bar=True,
    precision_recall_at_k=[1, 3, 5, 10],
)
wandb.login(key=WANDB_KEY)
run_name = f"{MODEL_NAME}-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
wandb.init(
    project="biomed-ir-synthesis",
    name=run_name,  
)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=NUM_EPOCHS,
    warmup_steps=WARMUP_STEPS,
    evaluation_steps=EVAL_STEPS,
    output_path=f'{OUTPUT_PATH}/{MODEL_NAME}',
    use_amp=True,
    show_progress_bar=True,
    checkpoint_path=f'{OUTPUT_PATH}/{MODEL_NAME}',  
    checkpoint_save_steps=EVAL_STEPS,        
    checkpoint_save_total_limit=SAVE_LIMIT    
)