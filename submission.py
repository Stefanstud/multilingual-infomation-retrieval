import pandas as pd
import json
from tqdm.notebook import tqdm
import numpy as np
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import multiprocessing as mp
import os 
from sentence_transformers import SentenceTransformer
import math
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sentence_transformers import util
import torch
from tqdm import tqdm
import sys
import csv
import pickle

# Load data
test = pd.read_csv("data/test.csv")
model = SentenceTransformer("all-mpnet-base-v2", trust_remote_code = True)

DOC_EMBEDDINGS_DIR = "embeddings_omg_chunked.pt"
# check if bert_document_embeddings.pt file exists
if os.path.isfile(f'{DOC_EMBEDDINGS_DIR}'):
    print("Loading the document embeddings from the existing .pt file...")
    documents = torch.load(f'{DOC_EMBEDDINGS_DIR}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

document_embeddings = torch.tensor(documents["embeddings"]).to(device)  # Convert and move in one step
final_matrix = torch.zeros((len(test), document_embeddings.shape[1]), device=device)

# Process each query
for index, row in tqdm(test.iterrows(), total=test.shape[0], desc="Encoding queries"):
    query = row['query']

    with torch.no_grad():  # Deactivate autograd for inference
        query_vector = model.encode(query)
        query_vector = torch.tensor(query_vector, dtype=torch.float).to(device)  # Convert and move in one step

    final_matrix[index] = query_vector.squeeze().detach() # Ensure vector is detached and squeezed

print("Query encoding complete.")
final_matrix = final_matrix.to(device)
doc_ids = documents['docids']  
doc_langs = documents['langs']  

unique_langs = sorted(set(test['lang']))  # Ensure all possible languages are included
lang_to_index = {lang: idx for idx, lang in enumerate(unique_langs)}

# Convert language strings in documents and tests to indices
doc_lang_indices = torch.tensor([lang_to_index[lang] for lang in doc_langs], dtype=torch.long, device=device)
test_lang_indices = [lang_to_index[lang] for lang in test['lang']]  # This will be used in the loop

results_final = []
batch_size = 1

for i in tqdm(range(0, len(test), batch_size), desc="Retrieving documents"):
    batch_queries = final_matrix[i:i + batch_size]
    batch_langs = test_lang_indices[i:i + batch_size]  # List of indices for languages in this batch
    similarities = util.pytorch_cos_sim(batch_queries, document_embeddings)

    for j, lang_index in enumerate(batch_langs):
        seen_doc_ids = set()
        unique_results = []
        k = 1000
        increment = 5

        lang_mask = (doc_lang_indices == lang_index).float()  # Convert bool mask to float
        masked_similarities = similarities[j] * lang_mask
        top_k_values, top_k_idx = torch.topk(masked_similarities, k=k)

        current_doc_ids = [doc_ids[idx] for idx in top_k_idx.cpu().numpy()]
        for doc_id in current_doc_ids:
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                unique_results.append(doc_id)
            if len(unique_results) == 10:
                break            

        results_final.append(unique_results[:10])

print("Document retrieval complete.")

def write_submission_csv(results_final, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'docids']) 
        
        for id, row in enumerate(results_final):
            # Format the docids as a string that looks like a Python list
            docids = ', '.join([f"'{docid}'" for docid in row])
            writer.writerow([id, f"[{docids}]"])

output_path = 'submission.csv'
write_submission_csv(results_final, output_path)