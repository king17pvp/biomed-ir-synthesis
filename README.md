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
4. Deactivate the environment after finished
```bash
deactivate # for virtual environment
conda deactivate # for conda environment
```