import pandas as pd
import json
from collections import defaultdict, Counter
import numpy as np
import math
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix
import csv
import os


class TFIDF:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.idfs = None
        self.doc_norms = None
    
    def build_vocabulary(self, documents):
        """Build vocabulary from documents"""
        vocab = set()
        for doc in tqdm(documents, desc="Building vocabulary"):
            vocab.update(doc.split())
        
        self.vocab = {term: idx for idx, term in enumerate(vocab)}
        self.inverse_vocab = {idx: term for term, idx in self.vocab.items()}
        return len(self.vocab)
    
    def compute_tf_sparse_old(self, documents):
        """Compute term frequencies for documents and return as sparse matrix"""
        matrix = lil_matrix((len(documents), len(self.vocab)), dtype=np.float32)
        doc_term_counts = []
        
        for doc_idx, doc in tqdm(enumerate(documents), desc="Computing TF"):
            words = doc.split()
            word_count = defaultdict(int)
            for word in words:
                if word in self.vocab:  # Only count words in vocabulary
                    word_count[word] += 1
            
            # Get max frequency for normalization
            max_freq = max(word_count.values()) if word_count else 1
            
            # Fill sparse matrix with normalized term frequencies
            for word, count in word_count.items():
                if word in self.vocab:
                    term_idx = self.vocab[word]
                    matrix[doc_idx, term_idx] = count / max_freq
            
            doc_term_counts.append(word_count)
            
        return csr_matrix(matrix), doc_term_counts
    
    def compute_tf_sparse(self, documents):
        """Compute term frequencies for documents and return as sparse matrix"""
        num_docs = len(documents)
        vocab_size = len(self.vocab)
        matrix = lil_matrix((num_docs, vocab_size), dtype=np.float32)
        doc_term_counts = []
        vocab = self.vocab  # Cache the vocabulary dictionary

        for doc_idx, doc in tqdm(enumerate(documents), total=num_docs, desc="Computing TF"):
            words = doc.split()
            # Filter words that are in the vocabulary
            words_in_vocab = [word for word in words if word in vocab]
            # Efficient counting using Counter
            word_count = Counter(words_in_vocab)
            # Append word counts for the document
            doc_term_counts.append(word_count)
            if not word_count:
                continue  # Skip if no valid words in the document
            # Get max frequency for normalization
            max_freq = max(word_count.values())
            # Prepare indices and data for the row
            term_indices = [vocab[word] for word in word_count.keys()]
            normalized_freqs = [count / max_freq for count in word_count.values()]
            # Directly assign the row in the lil_matrix
            matrix.rows[doc_idx] = term_indices
            matrix.data[doc_idx] = normalized_freqs

        return csr_matrix(matrix), doc_term_counts
    
    def compute_idf(self, doc_term_counts, num_docs):
        """Compute IDF values"""
        doc_freq = defaultdict(int)
        
        # Count document frequency for each term
        for doc_terms in tqdm(doc_term_counts, desc="Computing IDF"):
            for term in doc_terms.keys():
                if term in self.vocab:
                    doc_freq[term] += 1
        
        # Compute IDF for each term in vocabulary
        self.idfs = np.zeros(len(self.vocab))
        for term, idx in self.vocab.items():
            self.idfs[idx] = math.log(num_docs / (doc_freq[term] + 1))
    
    def compute_tfidf_matrix(self, tf_matrix):
        """Compute TF-IDF matrix"""
        # Multiply TF matrix with IDF values
        tfidf_matrix = tf_matrix.multiply(self.idfs)
        return tfidf_matrix
    
    def transform_queries(self, queries):
        """Transform queries into TF-IDF sparse matrix"""
        query_matrix = lil_matrix((len(queries), len(self.vocab)), dtype=np.float32)
        
        for query_idx, query in tqdm(enumerate(queries), desc="Processing queries"):
            words = query.lower().split()
            word_count = defaultdict(int)
            for word in words:
                if word in self.vocab:
                    word_count[word] += 1
            
            # Normalize term frequencies
            max_freq = max(word_count.values()) if word_count else 1
            
            # Fill sparse matrix
            for word, count in word_count.items():
                if word in self.vocab:
                    term_idx = self.vocab[word]
                    query_matrix[query_idx, term_idx] = (count / max_freq) * self.idfs[term_idx]
        
        return csr_matrix(query_matrix)
    
    def compute_cosine_similarities(self, query_matrix, doc_matrix):
        """Compute cosine similarities between queries and documents"""
        # Compute document norms if not already computed
        if self.doc_norms is None:
            self.doc_norms = np.sqrt(doc_matrix.multiply(doc_matrix).sum(axis=1)).A1
        
        # Compute query norms
        query_norms = np.sqrt(query_matrix.multiply(query_matrix).sum(axis=1)).A1
        
        # Compute dot products
        similarities = query_matrix.dot(doc_matrix.T).toarray()
        
        # Normalize by norms
        norms_matrix = np.outer(query_norms, self.doc_norms)
        similarities = np.divide(similarities, norms_matrix, where=norms_matrix != 0)
        
        return similarities

def main():
    # Load data
    test = pd.read_csv("data/test.csv")
    with open('data/corpus_small.json', 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    df['text'] = df['text'].apply(lambda x: x.lower())
    
    # Initialize TFIDF
    tfidf = TFIDF()
    
    # Build vocabulary
    vocab_size = tfidf.build_vocabulary(df['text'])
    print(f"Vocabulary size: {vocab_size}")
    
    # Compute TF matrix
    tf_matrix, doc_term_counts = tfidf.compute_tf_sparse(df['text'])
    
    # Compute IDF and TF-IDF matrix
    tfidf.compute_idf(doc_term_counts, len(df))
    tfidf_matrix = tfidf.compute_tfidf_matrix(tf_matrix)
    print("TF-IDF matrix shape:", tfidf_matrix.shape)
    
    # Process queries in batches
    BATCH_SIZE = 256
    results_final = []
    queries = test['query'].str.lower().tolist()
    
    for i in tqdm(range(0, len(queries), BATCH_SIZE), desc="Processing query batches"):
        batch_queries = queries[i:i+BATCH_SIZE]
        
        # Transform queries to TF-IDF
        query_matrix = tfidf.transform_queries(batch_queries)
        
        # Compute similarities
        similarities = tfidf.compute_cosine_similarities(query_matrix, tfidf_matrix)
        
        # Get top documents
        top_n = 10
        top_indices = similarities.argsort(axis=1)[:, -top_n:][:, ::-1]
        
        # Convert to document IDs
        batch_results = [df['docid'].iloc[indices].tolist() for indices in top_indices]
        results_final.extend(batch_results)
    
    # Write results
    def write_submission_csv(results_final, output_path):
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'docids'])
            for id, row in enumerate(results_final):
                docids = ', '.join([f"'{docid}'" for docid in row])
                writer.writerow([id, f"[{docids}]"])
    
    output_path = 'submission.csv'
    write_submission_csv(results_final, output_path)
    print("Submission file created successfully!")

if __name__ == "__main__":
    main()