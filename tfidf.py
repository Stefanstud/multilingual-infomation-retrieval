import pandas as pd
import json
from collections import defaultdict
import numpy as np
import math
from sentence_transformers import util
import torch
from tqdm import tqdm
import numpy as np
import csv
tqdm.pandas()
# Load data
test = pd.read_csv("data/test.csv")

with open('data/corpus.json', 'r') as file:
    data = json.load(file) 

df = pd.DataFrame(data)
df['text'] = df['text'].apply(lambda x: x.lower())

def compute_tf(text):
    words = text.split()
    word_count = defaultdict(int)
    for word in words:
        word_count[word] += 1
    tf_dict = {word: count / len(words) for word, count in word_count.items()}
    return tf_dict

def compute_idf(documents):
    """ Assume documents is a list of dictionaries of term frequencies """
    N = len(documents)
    idf_dict = defaultdict(lambda: math.log(N + 1))
    all_terms = set(term for doc in documents for term in doc.keys())
    
    doc_freq = defaultdict(int)
    # tqdm
    for doc in tqdm(documents, desc="Computing IDF"):
        for term in set(doc.keys()):
            doc_freq[term] += 1
    
    for term in all_terms:
        idf_dict[term] = math.log(N / (doc_freq[term] + 1))
    
    return idf_dict

def compute_tfidf(tf, idfs):
    tfidf = {word: tf[word] * idfs[word] for word in tf}
    return tfidf

df['tf'] = df['text'].progress_apply(compute_tf)
idfs = compute_idf(df['tf'].tolist())
df['tfidf'] = df.progress_apply(lambda row: compute_tfidf(row['tf'], idfs), axis=1)

doc_ids = df['docid']  
doc_langs = df['lang']
unique_langs = sorted(set(doc_langs))
lang_to_index = {lang: idx for idx, lang in enumerate(unique_langs)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
doc_lang_indices = torch.tensor([lang_to_index[lang] for lang in doc_langs], dtype=torch.long, device=device)
test_lang_indices = [lang_to_index[lang] for lang in test['lang']]

vocab = sorted(set(word for doc in df['tfidf'] for word in doc))
word_to_index = {word: idx for idx, word in enumerate(vocab)}

print("Completed preprocessing.")

def compute_norm(doc):
    return math.sqrt(sum(val ** 2 for val in doc.values()))

def cosine_similarity(doc1, doc2, norm1, norm2):
    intersection = set(doc1.keys()) & set(doc2.keys())
    dot_product = sum(doc1[x] * doc2[x] for x in intersection)
    
    if norm1 * norm2 == 0:
        return 0  # Avoid division by zero
    return dot_product / (norm1 * norm2)

def retrieve_top_docs_batch(queries, tfidf_matrix, top_n=10):
    queries_tfidf = vectori,zer.transform(queries)
    similarities = cosine_similarity(queries_tfidf, tfidf_matrix)
    top_docs_batch = []
    for sim in similarities:
        top_indices = sim.argsort()[-top_n:][::-1]
        top_docs_batch.append(df['docid'].iloc[top_indices].tolist())
    
    return top_docs_batch

print("Starting document retrieval in batches...")
BATCH_SIZE = 256
results_final = []
queries = test['query'].str.lower().tolist() 

# Process queries in batches
for i in tqdm(range(0, len(queries), BATCH_SIZE), total=len(queries) // BATCH_SIZE + 1):
    batch_queries = queries[i:i+BATCH_SIZE]
    batch_top_docs = retrieve_top_docs_batch(batch_queries, df['tfidf'])
    results_final.extend(batch_top_docs)

print("Document retrieval complete.")

# Write results to a CSV file
def write_submission_csv(results_final, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'docids']) 
        
        for id, row in enumerate(results_final):
            docids = ', '.join([f"'{docid}'" for docid in row])
            writer.writerow([id, f"[{docids}]"])

output_path = 'submission.csv'
write_submission_csv(results_final, output_path)
