import numpy as np
import pickle
import string
import nltk
import os
import csv
import math
import re
from tqdm import tqdm
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.sparse import lil_matrix, csr_matrix
from scipy import sparse
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from unicodedata import normalize
from pyarabic.normalize import normalize_searchtext


class BM25ChunkRetriever:
    def __init__(self, corpus_path=None, stopwords_path=None, chunk_size=500, k1 = 1.2, b = 0.7):
        self.corpus_path = corpus_path
        self.stopwords_path = stopwords_path
        self.chunk_size = chunk_size
        self.language_stopwords = self._init_stopwords()
        self.chunk_to_original_doc = defaultdict(str)
        self.de_stemmer = SnowballStemmer("german")
        self.k1 = k1
        self.b = b

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

    def normalize_korean(self, text):
        # unicode normalisation
        text = normalize('NFKC', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[""』」]', '"', text)
        text = re.sub(r'[''『「]', "'", text)
        text = re.sub(r'^[.…]+', '', text)
        return text.strip()

    def split_into_chunks(self, docid, text, is_query=False):
        """Split document into chunks of specified size"""
        if is_query:
            chunk_size = len(text)
        else:
            chunk_size = self.chunk_size

        tokens = word_tokenize(text)
        return [(docid, tokens[i:i + chunk_size]) 
                for i in range(0, len(tokens), chunk_size)]
  
    def tokenize(self, docs, is_query=False):
        """Tokenize documents into chunks while tracking original document IDs"""
        tokenized_docs = dict()
        chunk_counter = 0
        for doc in tqdm(docs, desc="Tokenizing"):
            docid = doc['docid']
            text = doc['text']
            lang = doc['lang']

            # normalize arabic
            if lang == "ar":
                text = normalize_searchtext(text)
            elif lang == "ko":
                text = self.normalize_korean(text)
            
            chunks = self.split_into_chunks(docid, text, is_query)
            for _, chunk_tokens in chunks:
                tokens = chunk_tokens
                stop_words = self.language_stopwords.get(lang, set())
                filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]

                # remove punctuation
                filtered_tokens = [word for word in filtered_tokens if word not in string.punctuation]

                # stemming
                if lang == "de":
                    filtered_tokens = [self.de_stemmer.stem(word) for word in filtered_tokens]
                
                tf = Counter(filtered_tokens)
                
                # store chunk with unique ID while maintaining mapping to original doc
                chunk_id = chunk_counter

                if not is_query:
                    self.chunk_to_original_doc[chunk_counter] = docid
                
                tokenized_docs[chunk_id] = {
                    'tf': tf,
                    'doc_len': len(filtered_tokens),
                    'lang': lang,
                }
                
                chunk_counter += 1
            
        return tokenized_docs
    
    def build_vocab(self, tokenized_corpus):
        """ Builds a vocabulary from the tokenized corpus."""
        print("Building vocab...")
        vocab = set()
        for doc in tokenized_corpus.values():
            vocab.update(doc['tf'].keys())
        vocab = {term: idx for idx, term in enumerate(vocab)}
        return vocab

    def build_sparse_matrix(self, docs_or_queries, vocab, idfs, avgdl, is_query=False, k1=1.2, b=0.7):
        matrix = lil_matrix((len(docs_or_queries), len(vocab)), dtype=np.float32)
        idx_to_chunkid = {}
        idx_to_chunkid = {}
        
        if not is_query:
            for idx, (chunkid, doc) in tqdm(enumerate(docs_or_queries.items()), desc="Building matrix"):
                idx_to_chunkid[idx] = chunkid
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
                        
        return csr_matrix(matrix), idx_to_chunkid

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
        
    def precompute(self, corpus, cache_dir='../data/'):
        """Precompute all necessary components for retrieval"""
        tok_corpus_path = os.path.join(cache_dir, 'tokenized_corpus_chunks.pkl')
        
        # tokenize corpus and save chunk mapping
        if os.path.exists(tok_corpus_path):
            tokenized_corpus = self.load_data(tok_corpus_path)
            self.chunk_to_original_doc = self.load_data(os.path.join(cache_dir, 'chunk_mapping.pkl'))
        else:                
            tokenized_corpus = self.tokenize(corpus)
            self.save_data(tokenized_corpus, tok_corpus_path)
            self.save_data(self.chunk_to_original_doc, os.path.join(cache_dir, 'chunk_mapping.pkl'))
        
        # build or load vocabulary
        if os.path.exists(os.path.join(cache_dir, 'vocab.pkl')):
            vocab = self.load_data(os.path.join(cache_dir, 'vocab.pkl'))
        else:
            vocab = self.build_vocab(tokenized_corpus)
            self.save_data(vocab, os.path.join(cache_dir, 'vocab.pkl'))
        
        # compute or load statistics
        if os.path.exists(os.path.join(cache_dir, 'idfs.pkl')):
            idfs = self.load_data(os.path.join(cache_dir, 'idfs.pkl'))
            avgdls = self.load_data(os.path.join(cache_dir, 'avgdls.pkl'))
        else:
            idfs, avgdls = self.compute_corpus_statistics(tokenized_corpus)
            self.save_data(idfs, os.path.join(cache_dir, 'idfs.pkl'))
            self.save_data(avgdls, os.path.join(cache_dir, 'avgdls.pkl'))
        
        # build and save BM25 matrix
        if os.path.exists(os.path.join(cache_dir, 'bm25_matrix.npz')):
            bm25_matrix = sparse.load_npz(os.path.join(cache_dir, 'bm25_matrix.npz'))
            idx_to_chunkid = self.load_data(os.path.join(cache_dir, 'idx_to_chunkid.pkl'))
        else:
            bm25_matrix, idx_to_chunkid = self.build_sparse_matrix(tokenized_corpus, vocab, idfs, avgdls, k1=self.k1, b=self.b)
            sparse.save_npz(os.path.join(cache_dir, 'bm25_matrix.npz'), bm25_matrix)
            self.save_data(idx_to_chunkid, os.path.join(cache_dir, 'idx_to_chunkid.pkl'))
        
        # build and save language masks
        if not os.path.exists(os.path.join(cache_dir, 'lang_masks.pkl')):
            unique_langs = ['en', 'de', 'fr', 'es', 'it', 'ar', 'ko']
            lang_masks = {}
            for lang in unique_langs:
                lang_masks[lang] = np.array([1 if tokenized_corpus[idx_to_chunkid[j]]["lang"] == lang else 0 
                                        for j in range(len(tokenized_corpus))])
            self.save_data(lang_masks, os.path.join(cache_dir, 'lang_masks.pkl'))
        else:
            lang_masks = self.load_data(os.path.join(cache_dir, 'lang_masks.pkl'))
        
        return {
            'tokenized_corpus': tokenized_corpus,
            'vocab': vocab,
            'idfs': idfs,
            'avgdls': avgdls,
            'bm25_matrix': bm25_matrix,
            'idx_to_chunkid': idx_to_chunkid,
            'lang_masks': lang_masks
        }

    def retrieve(self, queries, corpus, k=10, cache_dir='../data/'):
        """Retrieve documents using precomputed components"""
        # if any of the files are missing, precompute them
        precomputed = self.precompute(corpus, cache_dir)
        vocab, idfs, avgdls, bm25_matrix, idx_to_chunkid, lang_masks = (
            precomputed['vocab'], precomputed['idfs'], precomputed['avgdls'],
            precomputed['bm25_matrix'], precomputed['idx_to_chunkid'], precomputed['lang_masks']
        )

        # process queries
        query_docs = [
            {'docid': idx, 'text': row['query'], 'lang': row['lang']}
            for idx, row in queries.iterrows()
        ]
        tokenized_queries = self.tokenize(query_docs, is_query=True)
        
        # build query matrix and get scores
        query_matrix, _ = self.build_sparse_matrix(tokenized_queries, vocab, idfs, avgdls, 
                                                is_query=True, k1=self.k1, b=self.b)
        scores_matrix = query_matrix.dot(bm25_matrix.T).toarray()
        
        # get results
        results = {}
        for i in tqdm(range(len(queries)), desc="Getting top-k results"):
            query_lang = queries.iloc[i]["lang"]
            masked_scores = np.where(lang_masks[query_lang] == 1, scores_matrix[i], -np.inf)
            
            # Get top-k; *20 as heuristic 
            top_k_chunk_idx = np.argpartition(masked_scores, -k)[-k*20:]
            top_k_chunk_idx = top_k_chunk_idx[np.argsort(masked_scores[top_k_chunk_idx])[::-1]]
            
            seen_docs = set()
            top_k_docs = []
            for idx in top_k_chunk_idx:
                chunk_id = idx_to_chunkid[idx]
                original_doc = self.chunk_to_original_doc[chunk_id]
                if original_doc not in seen_docs:
                    top_k_docs.append(original_doc)
                    seen_docs.add(original_doc)

                if len(top_k_docs) == k:
                    break

            results[i] = top_k_docs

        return results
    
    def create_submission_csv(self, test_data, corpus, output_path):
        results = self.retrieve(test_data, corpus)
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'docids'])
            for idx, docs in results.items():
                writer.writerow([idx, str(docs)])
