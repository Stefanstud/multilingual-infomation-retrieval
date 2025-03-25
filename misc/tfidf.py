import pandas as pd
import json
import numpy as np
import math
import csv
import string
from tqdm import tqdm
from collections import defaultdict, Counter
from scipy.sparse import lil_matrix, csr_matrix
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)
LANG_CODES = ["en", "fr", "de", "ar", "es", "it"]
language_stopwords = {code: set(stopwords.words(lang)) for code, lang in 
                     zip(LANG_CODES, ['english', 'french', 'german', 'arabic', 'spanish', 'italian'])}
language_stopwords["ko"] = set()  

class TFIDF:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.idfs = None
    
    def build_vocabulary(self, documents):
        """Build vocabulary from documents"""
        vocab = set()
        for doc in tqdm(documents, desc="Building vocabulary"):
            vocab.update(doc.split())
        
        self.vocab = {term: idx for idx, term in enumerate(vocab)}
        self.inverse_vocab = {idx: term for term, idx in self.vocab.items()}
        return len(self.vocab)

    def compute_tf_sparse(self, documents):
        """Compute term frequencies as sparse matrix"""
        num_docs = len(documents)
        matrix = lil_matrix((num_docs, len(self.vocab)), dtype=np.float32)
        doc_term_counts = []

        for doc_idx, doc in tqdm(enumerate(documents), total=num_docs, desc="Computing TF"):
            words = [w for w in doc.split() if w in self.vocab]
            word_count = Counter(words)
            doc_term_counts.append(word_count)           
            if not word_count:
                continue

            # normalized version (by max word freq in the doc) of tfidf
            max_freq = max(word_count.values())
            term_indices = [self.vocab[word] for word in word_count.keys()]
            normalized_freqs = [count / max_freq for count in word_count.values()]
            matrix[doc_idx, term_indices] = normalized_freqs

        return csr_matrix(matrix), doc_term_counts
    
    def compute_idf(self, doc_term_counts, num_docs):
        """Compute IDF values"""
        doc_freq = defaultdict(int)
        for doc_terms in tqdm(doc_term_counts, desc="Computing IDF"):
            for term in doc_terms:
                doc_freq[term] += 1
        
        self.idfs = np.zeros(len(self.vocab))
        for term, idx in self.vocab.items():
            self.idfs[idx] = math.log(num_docs / (doc_freq[term] + 1))
    
    def compute_tfidf_matrix(self, tf_matrix):
        return tf_matrix.multiply(self.idfs)
    
    def transform_queries(self, queries):
        """Transform queries to TF-IDF vectors"""
        query_matrix = lil_matrix((len(queries), len(self.vocab)), dtype=np.float32)
        
        for query_idx, query in enumerate(queries):
            word_count = Counter(word for word in query.split() if word in self.vocab)
            
            if not word_count:
                continue
                
            max_freq = max(word_count.values())
            for word, count in word_count.items():
                term_idx = self.vocab[word]
                query_matrix[query_idx, term_idx] = (count / max_freq) * self.idfs[term_idx]
        
        return csr_matrix(query_matrix)
    
    def compute_cosine_similarities(self, query_matrix, doc_matrix):
        return query_matrix.dot(doc_matrix.T).toarray()

def main():
    test = pd.read_csv("data/test.csv")
    df = pd.DataFrame(json.load(open('data/corpus.json', 'r')))
    
    # preprocess text by language
    for lang in df['lang'].unique():
        mask = df['lang'] == lang
        df.loc[mask, 'text'] = df.loc[mask, 'text'].str.lower().apply(
            lambda x: ' '.join(w for w in x.split() if w not in language_stopwords[lang]))
    
    # remove punct
    df['text'] = df['text'].str.translate(str.maketrans('', '', string.punctuation))
    # initialize TF-IDF model
    tfidf = TFIDF()
    tfidf.build_vocabulary(df['text'])
    tf_matrix, doc_term_counts = tfidf.compute_tf_sparse(df['text'])
    tfidf.compute_idf(doc_term_counts, len(df))
    tfidf_matrix = tfidf.compute_tfidf_matrix(tf_matrix)

    np.save('data/tfidf_matrix.npy', tfidf_matrix)    
    results = []
    queries = test['query'].str.lower().str.translate(str.maketrans('', '', string.punctuation)).tolist()    
    for i in range(0, len(queries), 256):
        query_matrix = tfidf.transform_queries(queries[i:i+256])
        similarities = tfidf.compute_cosine_similarities(query_matrix, tfidf_matrix)
        top_indices = similarities.argsort(axis=1)[:, -10:][:, ::-1]
        results.extend([df['docid'].iloc[indices].tolist() for indices in top_indices])
    
    with open('submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'docids'])
        for idx, docids in enumerate(results):
            writer.writerow([idx, ' '.join(docids)])
    
    print("Submission file created successfully!")

if __name__ == "__main__":
    main()