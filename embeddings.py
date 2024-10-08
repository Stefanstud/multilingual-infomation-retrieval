import json
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import nltk

class TextDataset(Dataset):
    def __init__(self, data_path, words_per_chunk=100):
        with open(data_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)
        self.words_per_chunk = words_per_chunk
        self.chunks = []
        self._precompute_chunks()

    def _precompute_chunks(self):
        """Precomputes chunks for all documents."""
        for item in self.data:
            words = item['text'].split()
            for i in range(0, len(words), self.words_per_chunk):
                chunk_text = ' '.join(words[i:i + self.words_per_chunk])
                self.chunks.append({
                    'text': chunk_text,
                    'docid': item['docid'],
                    'lang': item['lang']
                })

    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        if idx >= len(self.chunks):
            raise IndexError("Chunk index out of range")
        return self.chunks[idx]
        

def extract_embeddings(data_loader, model):
    model.eval()  # Switch model to evaluation mode
    all_embeddings = []
    doc_ids = []
    langs = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing batches"):
            embeddings = model.encode(batch['text'], convert_to_tensor=True, device=model.device)
            all_embeddings.extend(embeddings)
            doc_ids.extend(batch['docid'])  
            langs.extend(batch['lang'])

    return torch.stack(all_embeddings), doc_ids, langs

# Load sentence encoder model
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")
model.to(torch.device('cuda'))  

# Prepare the dataset and data loader
dataset = TextDataset('data/corpus.json', words_per_chunk=500)
data_loader = DataLoader(dataset, batch_size=32768, shuffle=False)

# Extract embeddings, doc_ids, and languages
embeddings, doc_ids, langs = extract_embeddings(data_loader, model)

# Save embeddings and metadata
torch.save({'embeddings': embeddings, 'docids': doc_ids, 'langs': langs}, 'embeddings_distiluse_chunks.pt')
print("Embeddings and doc IDs saved.")
