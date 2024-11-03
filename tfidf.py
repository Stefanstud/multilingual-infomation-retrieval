import pandas as pd
import json
from collections import defaultdict, Counter
import numpy as np
import math
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix
import csv
import os
from nltk.corpus import stopwords
import string 
import nltk
nltk.download('stopwords')
language_stopwords = {
    "en": set(stopwords.words('english')),
    "fr": set(stopwords.words('french')),
    "de": set(stopwords.words('german')),
    "ar": set(stopwords.words('arabic')),
    "es": set(stopwords.words('spanish')),
    "it": set(stopwords.words('italian')),
    "ko": set(),
}


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
            vocab.update(doc.split()) # tokenize by splittign on whitespace
        
        self.vocab = {term: idx for idx, term in enumerate(vocab)}
        self.inverse_vocab = {idx: term for term, idx in self.vocab.items()}
        return len(self.vocab)

    def compute_tf_sparse(self, documents):
        """Compute term frequencies for documents and return as sparse matrix"""
        num_docs = len(documents)
        vocab_size = len(self.vocab)
        matrix = lil_matrix((num_docs, vocab_size), dtype=np.float32)
        doc_term_counts = []
        vocab = self.vocab  # Cache the vocabulary dictionary

        for doc_idx, doc in tqdm(enumerate(documents), total=num_docs, desc="Computing TF"):
            words = doc.split()
            words_in_vocab = [word for word in words if word in vocab]
            word_count = Counter(words_in_vocab)
            doc_term_counts.append(word_count)
           
            # get max frequency of term in document
            max_freq = max(word_count.values())
            # normalize term frequencies
            term_indices = [vocab[word] for word in word_count.keys()]
            normalized_freqs = [count / max_freq for count in word_count.values()]
            matrix.rows[doc_idx] = term_indices
            matrix.data[doc_idx] = normalized_freqs

        return csr_matrix(matrix), doc_term_counts
    
    def compute_idf(self, doc_term_counts, num_docs):
        """Compute IDF values"""
        doc_freq = defaultdict(int)
        
        for doc_terms in tqdm(doc_term_counts, desc="Computing IDF"):
            for term in doc_terms.keys():
                if term in self.vocab:
                    doc_freq[term] += 1
        
        self.idfs = np.zeros(len(self.vocab))
        for term, idx in self.vocab.items():
            self.idfs[idx] = math.log(num_docs / (doc_freq[term] + 1))
    
    def compute_tfidf_matrix(self, tf_matrix):
        """Compute TF-IDF matrix"""
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
            
            max_freq = max(word_count.values()) if word_count else 1
            
            for word, count in word_count.items():
                if word in self.vocab:
                    term_idx = self.vocab[word]
                    query_matrix[query_idx, term_idx] = (count / max_freq) * self.idfs[term_idx]
        
        return csr_matrix(query_matrix)
    
    def compute_cosine_similarities(self, query_matrix, doc_matrix):
        """Compute cosine similarities between queries and documents"""
        similarities = query_matrix.dot(doc_matrix.T).toarray()        
        return similarities

def main():
    test = pd.read_csv("data/test.csv")
    with open('data/corpus.json', 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)

    print("Cleaning and tokenizing text...")
    # clear stopwords per language and lower 
    # languages 
    langs = df['lang'].unique() 
    for lang in langs:
        stopwords = language_stopwords[lang]
        # okay copilot what the hell hahahahaha
        df.loc[df['lang'] == lang, 'text'] = df.loc[df['lang'] == lang, 'text'].str.lower().apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))

    df['text'] = df['text'].str.translate(str.maketrans('', '', string.punctuation))
    
    tfidf = TFIDF()
    vocab_size = tfidf.build_vocabulary(df['text'])
    print(f"Vocabulary size: {vocab_size}")
    
    tf_matrix, doc_term_counts = tfidf.compute_tf_sparse(df['text'])
    
    # Compute IDF and TF-IDF matrix
    tfidf.compute_idf(doc_term_counts, len(df))
    tfidf_matrix = tfidf.compute_tfidf_matrix(tf_matrix)
    print("TF-IDF matrix shape:", tfidf_matrix.shape)
    np.save('data/tfidf_matrix.npy', tfidf_matrix)    

    # -------------- Retrieval -------------- 
    BATCH_SIZE = 256
    results_final = []
    queries = test['query'].str.lower().tolist()
    for i in tqdm(range(0, len(queries), BATCH_SIZE), desc="Processing query batches"):
        batch_queries = queries[i:i+BATCH_SIZE]
        
        query_matrix = tfidf.transform_queries(batch_queries)      
        similarities = tfidf.compute_cosine_similarities(query_matrix, tfidf_matrix)
        
        top_n = 10
        top_indices = similarities.argsort(axis=1)[:, -top_n:][:, ::-1]
        
        batch_results = [df['docid'].iloc[indices].tolist() for indices in top_indices]
        results_final.extend(batch_results)
    
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