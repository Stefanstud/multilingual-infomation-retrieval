import json
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from FlagEmbedding import BGEM3FlagModel
import csv

class TextDataset(Dataset):
    def __init__(self, data_path, use_chunks=True, words_per_chunk=500):
        """
        :param data_path: Path to JSON data file
        :param use_chunks: Whether to split documents into chunks
        :param words_per_chunk: Size of each chunk in words
        """
        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        self.items = []
        if use_chunks:
            for item in data:
                words = item['text'].split()
                for i in range(0, len(words), words_per_chunk):
                    self.items.append({
                        'text': ' '.join(words[i:i + words_per_chunk]),
                        'docid': item['docid'],
                        'lang': item['lang']
                    })
        else:
            self.items = data

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]

def extract_embeddings(data_loader, model):
    """
    :param data_loader: DataLoader containing text data
    :param model: Model for generating embeddings
    :return: Tuple of embeddings, document IDs, and languages
    """
    embeddings, doc_ids, langs = [], [], []
    
    for batch in tqdm(data_loader, desc="Processing batches"):
        with torch.no_grad():
            batch_embeddings = model.encode(
                batch['text'],
                batch_size=128,
                max_length=8192,
                return_dense=True,
            )['dense_vecs']
        
        embeddings.extend(torch.tensor(batch_embeddings))
        doc_ids.extend(batch['docid'])
        langs.extend(batch['lang'])
        
    return torch.stack(embeddings), doc_ids, langs

def main():
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cuda')
    dataset = TextDataset('../data/corpus.json', use_chunks=True, words_per_chunk=500)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    embeddings, doc_ids, langs = extract_embeddings(data_loader, model)
    
    torch.save({
        'embeddings': embeddings,
        'docids': doc_ids,
        'langs': langs
    }, 'embeddings_bge_m3_chunked.pt')
    
    test = pd.read_csv("../data/test.csv")
    documents = torch.load("embeddings_bge_m3_chunked.pt")    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    document_embeddings = documents["embeddings"].to(device).half()
    
    query_embeddings = torch.zeros((len(test), document_embeddings.shape[1]), 
                                  device=device, dtype=torch.float16)
    
    for idx, row in tqdm(test.iterrows(), total=len(test), desc="Encoding queries"):
        with torch.no_grad():
            query_vector = model.encode(
                row['query'],
                batch_size=1,
                max_length=8192,
                return_dense=True
            )['dense_vecs']
        query_embeddings[idx] = torch.tensor(query_vector, device=device, dtype=torch.float16)
    
    doc_langs = documents['langs']
    unique_langs = sorted(set(test['lang']))
    lang_mapping = {lang: i for i, lang in enumerate(unique_langs)}
    doc_lang_indices = torch.tensor([lang_mapping[lang] for lang in doc_langs], device=device)
    test_lang_indices = [lang_mapping[lang] for lang in test['lang']]
    
    results = []
    batch_size = 1    
    for i in tqdm(range(0, len(test), batch_size), desc="Retrieving"):
        batch_queries = query_embeddings[i:i+batch_size]
        batch_langs = test_lang_indices[i:i+batch_size]
        similarities = torch.nn.functional.normalize(batch_queries, dim=1) @ \
                      torch.nn.functional.normalize(document_embeddings, dim=1).T
        
        for j, lang_idx in enumerate(batch_langs):
            lang_mask = (doc_lang_indices == lang_idx).float()
            masked_scores = similarities[j] * lang_mask            
            top_indices = torch.topk(masked_scores, k=1000).indices.cpu().numpy()
            seen_docs = set()
            unique_docs = []
            for idx in top_indices:
                doc_id = documents['docids'][idx]
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    unique_docs.append(doc_id)
                if len(unique_docs) == 10:
                    break
            
            results.append(unique_docs[:10])
    
    with open('submission_bge_m3.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'docids'])
        for idx, docids in enumerate(results):
            writer.writerow([idx, ' '.join(str(docid) for docid in docids)])
    
    print("Processing complete. Submission saved to submission_bge_m3.csv")

if __name__ == "__main__":
    main()