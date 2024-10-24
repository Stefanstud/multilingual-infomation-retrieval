# Import libraries
import pandas as pd
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm
import csv
import numpy as np
import string 
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
from collections import defaultdict, Counter
import math
import pickle
import os
from scipy import sparse

def save_data(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

def load_data(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def tokenize_korean_simple(text):
    tokens = text.split()
    return tokens

def tokenize(docs, language_stopwords):
    tokenized_docs = dict()

    for doc in tqdm(docs, desc="Tokenizing batch"):
        docid = doc['docid']
        text = doc['text']
        lang = doc['lang']

        text_no_punctuation = "".join([ch for ch in text if ch not in string.punctuation])
        
        if lang == 'ko':
            tokens = tokenize_korean_simple(text_no_punctuation)
        else:
            tokens = word_tokenize(text_no_punctuation)
        
        stop_words = language_stopwords.get(lang, set())
        filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]

        tf = Counter()
        tf.update(filtered_tokens)

        tokenized_docs[docid] = {
            'tf': tf,
            'doc_len': len(filtered_tokens),
            'lang': lang,
        }

    return tokenized_docs


def compute_corpus_statistics(tokenized_docs):
    """
    Computes the idf and average document length for each language in the corpus.
    Can be loaded from a file to have a faster runtime.
    """
    idf_by_lang = defaultdict(lambda: defaultdict(int))
    avgdl_by_lang = defaultdict(float)
    doc_len_by_lang = defaultdict(int)
    doc_count_by_lang = defaultdict(int)

    # Loop through each document in tokenized_docs
    for doc in tqdm(tokenized_docs.values(), desc="Calculating document statistics ..."):
        lang = doc['lang']
        tokens = doc['tf']
        total_tokens = len(tokens)
        
        # Update document length statistics for the language
        doc_len_by_lang[lang] += total_tokens
        doc_count_by_lang[lang] += 1

        # Count the number of documents each token appears in (document frequency)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            idf_by_lang[lang][token] += 1

    # Calculate average document length (avgdl) and idf for each language
    for lang, doc_count in tqdm(doc_count_by_lang.items(), desc="Calculating average document length and idf per language ..."):
        avgdl = doc_len_by_lang[lang] / doc_count
        avgdl_by_lang[lang] = avgdl

        idf = {}
        for term, freq in idf_by_lang[lang].items():
            idf[term] = math.log((doc_count - freq + 0.5) / (freq + 0.5) + 1)
        idf_by_lang[lang] = idf

    return idf_by_lang, avgdl_by_lang


def build_sparse_matrix(docs_or_queries, vocab, idfs, avgdl, is_query = False, k1=1.2, b=0.7):
    """Builds a sparse matrix from documents or queries."""
    matrix = lil_matrix((len(docs_or_queries), len(vocab)), dtype=np.float32)
    idx_to_docid = {} # maps int to docid; 0 -> doc-en-23; 1 -> doc-en-3223 ...
     
    if not is_query:
    # idx is used to index the lil_matrix (instead of docid) since it does not accept str as index
        for idx, (docid, doc) in tqdm(enumerate(docs_or_queries.items()), desc = "Building embeddings"):  
            idx_to_docid[idx] = docid # 0 -> doc-en-23; 1 -> doc-en-3223 ...
            norm_factor = k1 * (1 - b + b * doc['doc_len'] / avgdl[doc['lang']]) 
            for term, freq in doc['tf'].items():
                if term in vocab: # example error I got for building query: KeyError: 'getfruitbytypenamehighconcurrentversion'
                    term_index = vocab[term]
                    tf_adjusted = freq * (k1 + 1) / (freq + norm_factor)
                    matrix[idx, term_index] = tf_adjusted * idfs[doc['lang']].get(term, 0)         
    else:
        for idx, (_, query) in enumerate(docs_or_queries.items()):
            for term, freq in query['tf'].items():
                if term in vocab:
                    term_index = vocab[term]
                    matrix[idx, term_index] = freq # * idfs[query['lang']].get(term, 0)

    return csr_matrix(matrix), idx_to_docid # idx_to_docid useful only for corpus

# save results to csv
def write_submission_csv(results_final, output_path):
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'docids'])
        for idx, docs in results_final.items():
            # save as list like this: 0,"['doc-en-0', 'doc-de-14895', 'doc-en-829265', 'doc-en-147113', 'doc-en-644359', 'doc-en-585315', 'doc-en-234047', 'doc-en-14117', 'doc-en-794977', 'doc-en-374766']"
            writer.writerow([idx, str(docs)])

def build_vocab(tokenized_corpus):
    """ Builds a vocabulary from the tokenized corpus. This can also be loaded from this to have a faster runtime"""
    print("Building vocab...")
    vocab = set()
    for doc in tokenized_corpus.values():
        vocab.update(doc['tf'].keys())

    vocab = {term: idx for idx, term in enumerate(vocab)}
    return vocab
   
def main():
    TOK_CORPUS_PATH = '../data/tokenized_corpus.pkl'
    BM25_MATRIX_PATH = '../data/bm25_matrix.npz'
    IDX_TO_DOCID_PATH = '../data/idx_to_docid.pkl'
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

    # load corpus
    print("Loading corpus...")
    if not os.path.exists(TOK_CORPUS_PATH):
        with open('../data/corpus.json', 'r', encoding='utf-8') as f:
            corpus = json.load(f)
    else:
        with open('../data/corpus.json', 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        
    # from txt load korean stopwords
    with open('../data/stopwords-ko.txt', 'r', encoding='utf-8') as f:
        stopwords_ko = f.read().splitlines()

    # load test data
    test_data = pd.read_csv('../data/test.csv')
    
    # prepare test queries in a similar format as the corpus
    test_queries = [
        {'docid': row['id'], 'text': row['query'], 'lang': row['lang']}
        for idx, row in test_data.iterrows()
    ]

    language_stopwords = {
        "en": set(stopwords.words('english')),
        "fr": set(stopwords.words('french')),
        "de": set(stopwords.words('german')),
        "ar": set(stopwords.words('arabic')),
        "es": set(stopwords.words('spanish')),
        "it": set(stopwords.words('italian')), 
        "ko": set(stopwords_ko),
    } 
    
    print("Tokenizing corpus...")
    if os.path.exists(TOK_CORPUS_PATH):
        tokenized_corpus = load_data(TOK_CORPUS_PATH)
    else:
        tokenized_corpus = tokenize(corpus, language_stopwords)
        save_data(tokenized_corpus, TOK_CORPUS_PATH)
    
    print("Tokenizing queries...")
    tokenized_queries = tokenize(test_queries, language_stopwords)
    
    print("Retrieving results...")
    results_final = {}
    bm25_matrix = {}

    # build vocabulary
    vocab = build_vocab(tokenized_corpus)

    # compute corpus stistics
    idfs, avgdls = compute_corpus_statistics(tokenized_corpus)
    
    print("Corpus...")
    # build document and query embeddings using bm25 methodology
    if os.path.exists(BM25_MATRIX_PATH):
        bm25_matrix = sparse.load_npz(BM25_MATRIX_PATH)
        idx_to_docid = load_data(IDX_TO_DOCID_PATH)
    else:
        bm25_matrix, idx_to_docid = build_sparse_matrix(tokenized_corpus, vocab, idfs, avgdls)
    
    print("Query...")
    query_matrix, _ = build_sparse_matrix(tokenized_queries, vocab, idfs, avgdls, is_query = True)

    scores_matrix = query_matrix.dot(bm25_matrix.T)
    scores_matrix = scores_matrix.toarray() # convert from compressed sparse matrix to dense matrix  
    print("Scores computed, getting top_k results...")
    
    # top 10 documents, save in results_final dict
    k = 10
    results_final = {}
    for i in tqdm(range(len(test_data)), desc="Sorting results"): 
        query_lang = test_data.iloc[i]["lang"]  
        lang_mask = np.array([1 if corpus[j]["lang"] == query_lang else 0 for j in range(scores_matrix.shape[1])])
        masked_scores = np.where(lang_mask == 1, scores_matrix[i], -np.inf)
        top_k_idx = np.argsort(masked_scores)[::-1][:k]

        # Map indices to document IDs
        top_k_idx = [idx_to_docid[j] for j in top_k_idx]
        results_final[i] = top_k_idx  

    # save npz bm25
    sparse.save_npz(BM25_MATRIX_PATH, bm25_matrix)
    save_data(idx_to_docid, IDX_TO_DOCID_PATH)

    write_submission_csv(results_final, 'submission.csv')

if __name__ == "__main__":
    main()