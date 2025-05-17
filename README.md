# biomed-ir-synthesis
Optimizing Biomedical IR through Document Synthesis and Contextual Embeddings is a capstone project of course "Natural Language Processing" at Hanoi University of Science and Technology. The project focused on enhancing biomedical information retrieval by synthesizing query-relevant documents and leveraging contextualized language models such as BioBERT and MedCPT.

## How to run data synthesization

1. Create virtual environment and install required packages(or conda environment) 
```bash
cd biomed-ir-synthesis
python -m venv .venv #for virtual environment
conda create -n bio_medical_ir_synthesis_env python=3.10 -y # for conda environment
```

2. Activate current virtual (or conda environment)
```bash
source ./.venv/Scripts/activate # for virtual environment on linux-based terminal
.\.venv\Scripts\activate.ps1 # for virtual environment on powershell
.\.venv\Scripts\activate # for virtual environment on command prompt
conda activate bio_medical_ir_synthesis_env # for conda environment
```

3. Place the corpus (in forms of .txt), each line in the .txt file represent a document. There can be many corpus files inside the data folder. The directory tree of the project should be like the following:
```
biomed-ir-synthesis/
├── test_api.py
├── data/
│   ├──[corpus_name_1].txt
│   ├──[corpus_name_2].txt
│   ├──[corpus_name..].txt
│   ├──[corpus_name_n].txt
├── tools/
│   ├── __init__.py
│   └── api.py      
```

4. Create a DeepSeek API Key and place it in a .env file
```
DEEPSEEK_KEY=<YOUR_KEY>
```

5. Run data synthesis by the following command
```bash
python generate_queries.py --input_dir <INPUT_DIR> --output_dir <OUTPUT_DIR>
```

6. Deactivate the environment after finished
```bash
deactivate # for virtual environment
conda deactivate # for conda environment
```

## Training 

### For Training bi-encoders

```bash
python src/train/train_bi_encoder.py \
    --model_name nlpie/tiny-biobert \
    --batch_size 32 \ 
    --threshold 5.0 \
    --dataset_path path/to/json \
    --num_epochs 5 \
    --warmup_steps 1000 \
    --eval_steps 1000 \ 
    --output_path path/to/output \
    --checkpoint_path path/to/checkpoint \ 
    --save_steps 1000 \ 
    --save_total_limit 3
```

### For training cross-encoders
```bash
python src/train/train_cross_encoder.py \
    --num_epochs 1 \ 
    --model_name nlpie/tiny-biobert \
    --batch_size 32 \ 
    --data_path path/to/data \
    --learning_rate 2e-5 \
    --eval_steps 2000 \ 
    --save_steps 2000 \ 
    --output_dir path/to/output \
    --fp16 \
    --save_total_limit 2
```

## How to run evaluation

For ```BioASQ```, the dataset is not publicly accessible, check out the [BEIR Insturction](https://github.com/beir-cellar/beir/tree/main/examples/dataset#2-bioasq) repo for the reproduction of the dataset.

* Firstly, install elastic search for BM25
```bash
wget -q https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-oss-7.9.2-linux-x86_64.tar.gz
wget -q https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-oss-7.9.2-linux-x86_64.tar.gz.sha512
tar -xzf elasticsearch-oss-7.9.2-linux-x86_64.tar.gz
sudo chown -R daemon:daemon elasticsearch-7.9.2/
shasum -a 512 -c elasticsearch-oss-7.9.2-linux-x86_64.tar.gz.sha512
```
Then run it with the terminal:
```bash
sudo -H -u daemon elasticsearch-7.9.2/bin/elasticsearch
```

Checking if Elastic Search is running with:
```bash
curl -sX GET "localhost:9200/"
```

* Secondly, put your bi-encoder, cross-encoder from this [REPO](https://huggingface.co/king17pvp/NLP-summer-2025-biomedical-information-retrieval/tree/main) into ```ckpts/biencoder-checkpoints```, and ```ckpts/crossencoder-checkpoints``` so that the directory tree look as follows:

```
biomed-ir-synthesis/
├── .git/
├── ckpts/
│   ├── biencoder-checkpoints/
│   │   ├── checkpoint-bertbase/
│   │   ├── checkpoint-pubmedbert/
│   │   └── checkpoint-tinybiobert/
│   └── crossencoder-checkpoints/
│       ├── checkpoint-bertbase/
│       ├── checkpoint-pubmedbert/
│       └── checkpoint-tinybiobert/
```

* Then run the evaluation process by:

```bash
python run_evaluate.py \ 
    --biencoder_model_name pubmedbert \
    --crossencoder_model_name pubmedbert \ 
    --top_k 10 
```

**NOTES**: Only ```pubmedbert```, ```tinybiobert```, and ```bert``` are valid options for both bi-encoder and cross-encoder arguments 