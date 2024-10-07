import json
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'text': self.data[idx]['text'],
            'docid': self.data[idx]['docid']
            'lang': self.data[idx]['lang']
        }

def collate_fn(batch):
    texts = [item['text'] for item in batch]
    docids = [item['docid'] for item in batch]
    lang = [item['lang'] for item in batch]
    return {'texts': texts, 'docids': docids, 'lang': lang}

def extract_embeddings(data_loader, model):
    model.eval()  # Switch model to evaluation mode
    all_embeddings = []
    doc_ids = []
    lang = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing batches"):
            embeddings = model.encode(batch['texts'], convert_to_tensor=True, batch_size=len(batch['texts']))
            all_embeddings.append(embeddings)
            doc_ids.extend(batch['docids'])  
            lang.extend(batch['lang'])

    # Concatenate all embeddings from batches
    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    return embeddings_tensor, doc_ids, lang

# Load sentence encoder model
model = SentenceTransformer("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)

# Prepare the dataset and data loader
dataset = TextDataset('data/corpus.json')
data_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=False)  # Smaller batch size for heavier models

# Extract embeddings and docids
embeddings, doc_ids, lang = extract_embeddings(data_loader, model)

# Save embeddings and docids
torch.save({'embeddings': embeddings, 'docids': doc_ids, 'lang': lang}, 'embeddings_with_ids.pt')
print("Embeddings and doc IDs saved to sentence_embeddings_with_ids.pt.")
