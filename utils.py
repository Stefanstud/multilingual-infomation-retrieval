import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
import pandas as pd

class RetrievalDataset(Dataset):
    def __init__(self, data, docid_to_text):
        self.data = data
        self.docid_to_text = docid_to_text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query = self.data.iloc[idx]['query']
        positive_docid = self.data.iloc[idx]['positive_docs']
        negative_docids = eval(self.data.iloc[idx]['negative_docs'])
        
        positive_doc = self.docid_to_text[positive_docid]
        negative_docs = [self.docid_to_text[docid] for docid in negative_docids]  
        return query, positive_doc, negative_docs

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer

def tokenize_batch(text_list, tokenizer, device):
    return tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(device)

def load_data(data_path):
    train_data = pd.read_csv(f"{data_path}/train.csv")
    eval_data = pd.read_csv(f"{data_path}/dev.csv")
    corpus = pd.read_json(f"{data_path}/corpus.json")
    docid_to_text = dict(zip(corpus['docid'], corpus['text']))
    del corpus
    return train_data, eval_data, docid_to_text