import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
import pickle
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.sparse import lil_matrix, csr_matrix
from scipy import sparse
import os
import math

class BM25Retriever:
    def __init__(self, corpus_path=None, stopwords_path=None):
        self.corpus_path = corpus_path
        self.stopwords_path = stopwords_path
        self.language_stopwords = self._init_stopwords()
        
    def _init_stopwords(self):
        """Initialize stopwords for all supported languages"""
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('punkt_tab')
        
        language_stopwords = {
            "en": set(stopwords.words('english')),
            "fr": set(stopwords.words('french')),
            "de": set(stopwords.words('german')),
            "ar": set(stopwords.words('arabic')),
            "es": set(stopwords.words('spanish')),
            "it": set(stopwords.words('italian')),
        }
        
        if self.stopwords_path:
            with open(self.stopwords_path, 'r', encoding='utf-8') as f:
                language_stopwords["ko"] = set(f.read().splitlines())
                
        return language_stopwords
    
    @staticmethod
    def save_data(data, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load_data(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def tokenize_korean_simple(text):
        return text.split()
    
    def tokenize(self, docs):
        """Tokenize documents or queries"""
        tokenized_docs = dict()
        
        for doc in tqdm(docs, desc="Tokenizing"):
            docid = doc['docid']
            text = doc['text']
            lang = doc['lang']
            
            text_no_punctuation = "".join([ch for ch in text if ch not in string.punctuation])
            
            if lang == 'ko':
                tokens = self.tokenize_korean_simple(text_no_punctuation)
            else:
                tokens = word_tokenize(text_no_punctuation)
            
            stop_words = self.language_stopwords.get(lang, set())
            filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
            
            tf = Counter(filtered_tokens)
            
            tokenized_docs[docid] = {
                'tf': tf,
                'doc_len': len(filtered_tokens),
                'lang': lang,
            }
            
        return tokenized_docs
    
    def compute_corpus_statistics(self, tokenized_docs):
        """Compute IDF and average document length for each language"""
        idf_by_lang = defaultdict(lambda: defaultdict(int))
        avgdl_by_lang = defaultdict(float)
        doc_len_by_lang = defaultdict(int)
        doc_count_by_lang = defaultdict(int)
        
        for doc in tqdm(tokenized_docs.values(), desc="Computing statistics"):
            lang = doc['lang']
            tokens = doc['tf']
            doc_len_by_lang[lang] += len(tokens)
            doc_count_by_lang[lang] += 1
            
            for token in set(tokens):
                idf_by_lang[lang][token] += 1
                
        for lang, doc_count in doc_count_by_lang.items():
            avgdl_by_lang[lang] = doc_len_by_lang[lang] / doc_count
            
            idf = {}
            for term, freq in idf_by_lang[lang].items():
                idf[term] = math.log((doc_count - freq + 0.5) / (freq + 0.5) + 1)
            idf_by_lang[lang] = idf
            
        return dict(idf_by_lang), avgdl_by_lang
    
    def build_sparse_matrix(self, docs_or_queries, vocab, idfs, avgdl, is_query=False, k1=1.2, b=0.7):
        matrix = lil_matrix((len(docs_or_queries), len(vocab)), dtype=np.float32)
        idx_to_docid = {}
        
        if not is_query:
            for idx, (docid, doc) in tqdm(enumerate(docs_or_queries.items()), desc="Building matrix"):
                idx_to_docid[idx] = docid
                norm_factor = k1 * (1 - b + b * doc['doc_len'] / avgdl[doc['lang']])
                for term, freq in doc['tf'].items():
                    if term in vocab:
                        term_index = vocab[term]
                        tf_adjusted = freq * (k1 + 1) / (freq + norm_factor)
                        matrix[idx, term_index] = tf_adjusted * idfs[doc['lang']].get(term, 0)
        else:
            for idx, (_, query) in enumerate(docs_or_queries.items()):
                for term, freq in query['tf'].items():
                    if term in vocab:
                        term_index = vocab[term]
                        matrix[idx, term_index] = freq
                        
        return csr_matrix(matrix), idx_to_docid
    
    @staticmethod
    def build_vocab(tokenized_corpus):
        """Build vocabulary from tokenized corpus"""
        vocab = set()
        for doc in tokenized_corpus.values():
            vocab.update(doc['tf'].keys())
        return {term: idx for idx, term in enumerate(vocab)}
    
    def retrieve(self, queries, corpus, k=10, cache_dir='../data/'):
        """Main retrieval function"""
        tok_corpus_path = os.path.join(cache_dir, 'tokenized_corpus.pkl')
        bm25_matrix_path = os.path.join(cache_dir, 'bm25_matrix.npz')
        idx_to_docid_path = os.path.join(cache_dir, 'idx_to_docid.pkl')
        
        query_docs = [
            {'docid': idx, 'text': row['query'], 'lang': row['lang']}
            for idx, row in queries.iterrows()
        ]
        
        if os.path.exists(tok_corpus_path):
            tokenized_corpus = self.load_data(tok_corpus_path)
        else:
            tokenized_corpus = self.tokenize(corpus)
            self.save_data(tokenized_corpus, tok_corpus_path)
            
        tokenized_queries = self.tokenize(query_docs)
        
        vocab = self.build_vocab(tokenized_corpus)
        idfs, avgdls = self.compute_corpus_statistics(tokenized_corpus)
        
        if os.path.exists(bm25_matrix_path):
            bm25_matrix = sparse.load_npz(bm25_matrix_path)
            idx_to_docid = self.load_data(idx_to_docid_path)
        else:
            bm25_matrix, idx_to_docid = self.build_sparse_matrix(tokenized_corpus, vocab, idfs, avgdls)
            sparse.save_npz(bm25_matrix_path, bm25_matrix)
            self.save_data(idx_to_docid, idx_to_docid_path)
            
        query_matrix, _ = self.build_sparse_matrix(tokenized_queries, vocab, idfs, avgdls, is_query=True)
        scores_matrix = query_matrix.dot(bm25_matrix.T).toarray()
        
        results = {}
        for i in tqdm(range(len(queries)), desc="Getting top-k results"):
            query_lang = queries.iloc[i]["lang"]
            lang_mask = np.array([1 if corpus[j]["lang"] == query_lang else 0 for j in range(scores_matrix.shape[1])])
            masked_scores = np.where(lang_mask == 1, scores_matrix[i], -np.inf)
            top_k_idx = np.argsort(masked_scores)[::-1][:k]
            results[i] = [idx_to_docid[j] for j in top_k_idx]
            
        return results