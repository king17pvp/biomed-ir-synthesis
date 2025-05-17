import os
import argparse
import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from tools.api import *

from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
deekseek_api_key = os.getenv("DEEPSEEK_KEY")
API = DeepSeekAPI(deekseek_api_key)
data = []

def process_line(line):
    try:
        id, document = line
        queries = API.generate_queries(document=document)["query"]
        data.append({
            "id": id,
            "document": document,
            "queries": queries
        })
    except Exception as e:
        print(f"Error processing document: {document[:30]}... \n{e}")
def main(args):
    os.makedirs(args.output_directory, exist_ok=True)
    for corpus_filename in os.listdir(args.input_directory):
        if not corpus_filename.endswith(".txt"):
            continue
        if "test" in corpus_filename:
            continue
        input_path = os.path.join(args.input_directory, corpus_filename)
        output_path = os.path.join(args.output_directory, corpus_filename.replace(".txt", "queries.json"))
        global data
        data = []
        documents = []
        metadata = []
        with open(input_path, 'r', encoding='utf-8') as fr:
            tmp = fr.readlines()
            for doc in tmp:
                id, doc = doc.strip().split("\t")
                metadata.append((id, doc))
            # documents = [line.strip() for line in fr.readlines()]
        for line in metadata:
            document = line[1]
            if args.max_documents:
                if len(document) > args.max_documents:
                    document = document[:args.max_documents]
            documents.append((line[0], document))
        # for line in tqdm.tqdm(documents):
        #     process_line(line)
                
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_line, line) for line in tqdm.tqdm(documents)]
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result:
                    data.append(result)
        # for line in tqdm.tqdm(documents):
        #     try:
        #         id, document = line
        #         queries = API.generate_queries(document=document)["query"]
                
        #     except Exception as e:
        #         print(f"Error processing document: {document[:30]}... \n{e}")
        #         continue
        with open(output_path, 'w', encoding='utf-8') as fw:
            json.dump(data, fw, indent=2, ensure_ascii=False)
        print(f"Saved at {output_path}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments to run synthesization for unlabled corpus")
    parser.add_argument("--input_directory", type=str, default="data", help="Path to input directory of text files")
    parser.add_argument("--output_directory", type=str, default="output", help = "Path to the output directory of generated queries")
    parser.add_argument("--max_documents", type=int, default=None, help="Maximum number of documents to process per file")  
    args = parser.parse_args()
    main(args)