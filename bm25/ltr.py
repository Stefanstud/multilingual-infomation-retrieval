import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from scipy.sparse import load_npz, vstack
import pickle
from bm25 import build_sparse_matrix, tokenize
from sklearn.metrics import log_loss
from tqdm import tqdm

TRAIN_DATA_PATH = '../data/train.csv'
DEV_DATA_PATH = '../data/dev.csv'
IDX_TO_DOCID_PATH = '../data/idx_to_docid.pkl'
BM25_MATRIX_PATH = '../data/bm25_matrix.pkl'

def load_data(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

train_data = pd.read_csv(TRAIN_DATA_PATH)
dev_data = pd.read_csv(DEV_DATA_PATH)
bm25_scores = load_data(BM25_MATRIX_PATH)
idx_to_docid = load_data(IDX_TO_DOCID_PATH)
docid_to_idx = {}
for lang in idx_to_docid:
    docid_to_idx[lang] = {v: k for k, v in idx_to_docid[lang].items()}

def create_training_data(data, bm25_matrices, docid_to_idx):
    features = []
    labels = []
    for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing data"):
        query = row['query']
        query_lang = row['lang']
        pos_doc_id = row['positive_docs']
        neg_doc_ids = eval(row['negative_docs'])  
        lang_matrix = bm25_matrices[query_lang]

        pos_features = lang_matrix.getrow(docid_to_idx[query_lang][pos_doc_id]).toarray().flatten()
        for neg_doc_id in neg_doc_ids:
            neg_features = lang_matrix.getrow(docid_to_idx[query_lang][neg_doc_id]).toarray().flatten()

            pairwise_features = pos_features - neg_features
            features.append(pairwise_features)
            labels.append(1)  # Positive label should rank higher

            pairwise_features = neg_features - pos_features
            features.append(pairwise_features)
            labels.append(0)  # Negative label should not rank higher

    return np.array(features), np.array(labels)

features, labels = create_training_data(train_data, bm25_scores, docid_to_idx)
val_features, val_labels = create_features(dev_data, bm25_scores, docid_to_idx)

print("Fitting the model")
model = LogisticRegression(max_iter=1000, verbose=1)
model.fit(features, labels)

# Evaluate the model
val_probabilities = model.predict_proba(val_features)[:, 1]
validation_loss = log_loss(val_labels, val_probabilities)
print(f"Validation Logistic Loss: {validation_loss}")
