import json
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, data_path, use_chunks=True, words_per_chunk=500):
        with open(data_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)
        self.use_chunks = use_chunks
        self.words_per_chunk = words_per_chunk
        self.items = []
        self._preprocess_data()

    def _preprocess_data(self):
        """Preprocess data based on whether to use chunks or whole documents."""
        if self.use_chunks:
            self._precompute_chunks()
        else:
            self.items = self.data

    def _precompute_chunks(self):
        """Precomputes chunks for all documents."""
        for item in self.data:
            words = item['text'].split()
            for i in range(0, len(words), self.words_per_chunk):
                chunk_text = ' '.join(words[i:i + self.words_per_chunk])
                self.items.append({
                    'text': chunk_text,
                    'docid': item['docid'],
                    'lang': item['lang']
                })

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        if idx >= len(self.items):
            raise IndexError("Index out of range")
        return self.items[idx]

def extract_embeddings(data_loader, model):
    model.eval()  # Switch model to evaluation mode
    all_embeddings = []
    doc_ids = []
    langs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing batches"):
            embeddings = model.encode(
                batch['text'], 
                device=model.device, 
                batch_size=512, 
                convert_to_tensor=True
            )
            # Move embeddings to CPU
            embeddings = embeddings.cpu()
            all_embeddings.extend(embeddings)
            doc_ids.extend(batch['docid'])  
            langs.extend(batch['lang'])
            del embeddings  # Free up memory
    
    return torch.stack(all_embeddings), doc_ids, langs

# Load sentence encoder model
model_name_or_path="Alibaba-NLP/gte-multilingual-base"
model = SentenceTransformer(model_name_or_path, trust_remote_code=True)
print(f"Loaded model: {model_name_or_path}")
model.to(torch.device('cuda'))
print(f"Moved model to device: {model.device}")

# Choose whether to use chunks or whole documents
USE_CHUNKS = False  # Set to False to use whole documents
WORDS_PER_CHUNK = 500 if USE_CHUNKS else None

# Prepare the dataset and data loader
dataset = TextDataset('data/corpus.json', use_chunks=USE_CHUNKS, words_per_chunk=WORDS_PER_CHUNK)
print(f"Loaded {len(dataset)} {'chunks' if USE_CHUNKS else 'documents'}")

# Adjust batch size based on whether using chunks or whole documents
BATCH_SIZE = 512 if USE_CHUNKS else 100  
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

embeddings, doc_ids, langs = extract_embeddings(data_loader, model)

output_file = 'embeddings_gte.pt' if USE_CHUNKS else 'embeddings_gte.pt'
torch.save({'embeddings': embeddings, 'docids': doc_ids, 'langs': langs}, output_file)
print(f"Embeddings and doc IDs saved to {output_file}")
