import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from FlagEmbedding import BGEM3FlagModel
import csv

class TextDataset(Dataset):
    def __init__(self, data_path, use_chunks=True, words_per_chunk=500):
        with open(data_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)
        self.use_chunks = use_chunks
        self.words_per_chunk = words_per_chunk
        self.items = []
        self._preprocess_data()

    def _preprocess_data(self):
        """Preprocess data based on whether to use chunks or whole documents."""
        if self.use_chunks:
            self._precompute_chunks()
        else:
            self.items = self.data

    def _precompute_chunks(self):
        """Precomputes chunks for all documents."""
        for item in self.data:
            words = item['text'].split()
            for i in range(0, len(words), self.words_per_chunk):
                chunk_text = ' '.join(words[i:i + self.words_per_chunk])
                self.items.append({
                    'text': chunk_text,
                    'docid': item['docid'],
                    'lang': item['lang']
                })

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        if idx >= len(self.items):
            raise IndexError("Index out of range")
        return self.items[idx]

def extract_embeddings(data_loader, model):
    all_embeddings = []
    doc_ids = []
    langs = []
    
    for batch in tqdm(data_loader, desc="Processing batches"):
        with torch.no_grad():
            embeddings = model.encode(
                batch['text'],
                batch_size=128,
                max_length=8192,
                return_dense=True,
            )['dense_vecs']
        
        all_embeddings.extend(torch.tensor(embeddings)) 
        doc_ids.extend(batch['docid'])
        langs.extend(batch['lang'])
        
    return torch.stack(all_embeddings), doc_ids, langs  

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cuda')
print("Loaded BGE-M3 model")

dataset = TextDataset('../data/corpus.json', use_chunks=True, words_per_chunk=500)
data_loader = DataLoader(dataset, batch_size=128, shuffle=False)

embeddings, doc_ids, langs = extract_embeddings(data_loader, model)

torch.save({
    'embeddings': embeddings,
    'docids': doc_ids,
    'langs': langs
}, 'embeddings_bge_m3_chunked.pt')
print("Embeddings saved.")

test = pd.read_csv("../data/test.csv")
print("Loaded test data")

DOC_EMBEDDINGS_DIR = "embeddings_bge_m3_chunked.pt"
print("Loading the document embeddings from the existing .pt file...")
documents = torch.load(DOC_EMBEDDINGS_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
document_embeddings = documents["embeddings"].to(device)
document_embeddings = document_embeddings.half()

final_matrix = torch.zeros((len(test), document_embeddings.shape[1]), device=device, dtype=torch.float16)

for index, row in tqdm(test.iterrows(), total=test.shape[0], desc="Encoding queries"):
    query = row['query']
    
    with torch.no_grad():
        query_vector = model.encode(
            query,
            batch_size=1,
            max_length=8192,
            return_dense=True
        )['dense_vecs']
    
    final_matrix[index] = torch.tensor(query_vector, device=device, dtype=torch.float16).squeeze()

print("Query encoding complete.")

doc_ids = documents['docids']
doc_langs = documents['langs']

unique_langs = sorted(set(test['lang']))
lang_to_index = {lang: idx for idx, lang in enumerate(unique_langs)}
doc_lang_indices = torch.tensor([lang_to_index[lang] for lang in doc_langs], device=device)
test_lang_indices = [lang_to_index[lang] for lang in test['lang']]

results_final = []
batch_size = 1

for i in tqdm(range(0, len(test), batch_size), desc="Retrieving documents"):
    batch_queries = final_matrix[i:i + batch_size].half()  
    batch_langs = test_lang_indices[i:i + batch_size]
    
    similarities = torch.nn.functional.normalize(batch_queries, dim=1) @ \
                   torch.nn.functional.normalize(document_embeddings, dim=1).T

    for j, lang_index in enumerate(batch_langs):
        seen_doc_ids = set()
        unique_results = []
        k = 1000  # Initial number of results to consider
        
        lang_mask = (doc_lang_indices == lang_index).float()
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
            docids = ', '.join([f"'{docid}'" for docid in row])
            writer.writerow([id, f"[{docids}]"])

output_path = 'submission_bge_m3.csv'
write_submission_csv(results_final, output_path)
print(f"Submission file saved to {output_path}")
