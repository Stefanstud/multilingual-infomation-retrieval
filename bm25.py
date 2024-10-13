# Import libraries
import pandas as pd
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm
import csv
import os
import numpy as np
# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load the corpus
with open('data/corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

# Create a mapping from docid to document text and language
docid_to_text = {}
docid_to_lang = {}
for doc in corpus:
    docid_to_text[doc['docid']] = doc['text']
    docid_to_lang[doc['docid']] = doc['lang']

# Create language-specific stopword sets
language_stopwords = {}
for lang in set(docid_to_lang.values()):
    try:
        language_stopwords[lang] = set(stopwords.words(lang))
    except OSError:
        # If NLTK doesn't have stopwords for this language, use an empty set
        language_stopwords[lang] = set()

# Tokenize and preprocess the documents
tokenized_corpus = []
docids = []
for docid, text in tqdm(docid_to_text.items(), desc="Processing documents"):
    lang = docid_to_lang[docid]
    stop_words = language_stopwords.get(lang, set())
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stopwords and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokenized_corpus.append(tokens)
    docids.append(docid)

# Build the BM25 index
from rank_bm25 import BM25Okapi

bm25 = BM25Okapi(tokenized_corpus)
# save 
import pickle
with open('bm25.pkl', 'wb') as f:
    pickle.dump(bm25, f)

# Load the test queries
test_df = pd.read_csv('data/test.csv')

# Preprocess the queries
def preprocess_query(query, lang):
    stop_words = language_stopwords.get(lang, set())
    tokens = word_tokenize(query.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

def preprocess_queries_batch(queries, langs, batch_size=128):
    batched_tokens = []
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i + batch_size]
        batch_langs = langs[i:i + batch_size]
        batch_tokens = [preprocess_query(query, lang) for query, lang in zip(batch_queries, batch_langs)]
        batched_tokens.extend(batch_tokens)
    return batched_tokens

def batch_retrieve_documents(batched_query_tokens, bm25, top_n=10):
    batched_results = []
    for query_tokens in batched_query_tokens:            
        scores = bm25.get_scores(query_tokens)
        top_n_indices = np.argsort(scores)[-top_n:][::-1]
        top_n_docids = [docids[i] for i in top_n_indices]
        batched_results.append(top_n_docids)
    return batched_results

# Example usage:
queries = test_df['query'].tolist()
langs = test_df['lang'].tolist()
batched_query_tokens = preprocess_queries_batch(queries, langs)

# Retrieve documents in batches
results_final = batch_retrieve_documents(batched_query_tokens, bm25)

# Write the submission files
def write_submission_csv(results_final, output_path):
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'docids']) 
        for id, row in enumerate(results_final):
            # Format the docids as a string that looks like a Python list
            docids_str = ', '.join([f"'{docid}'" for docid in row])
            writer.writerow([id, f"[{docids_str}]"])

output_path = 'submission_bm25.csv'
write_submission_csv(results_final, output_path)
print(f"Submission file saved to {output_path}")
