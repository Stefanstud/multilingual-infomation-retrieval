import json
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, data_path, use_chunks=True, words_per_chunk=500):
        """
        :param data_path: Path to the JSON data file
        :param use_chunks: Whether to split documents into chunks
        :param words_per_chunk: Number of words per chunk if we decide to chunk 
        """
        with open(data_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)
        self.items = []
        
        if use_chunks:
            for item in self.data:
                words = item['text'].split()
                for i in range(0, len(words), words_per_chunk):
                    self.items.append({
                        'text': ' '.join(words[i:i + words_per_chunk]),
                        'docid': item['docid'],
                        'lang': item['lang']
                    })
        else:
            self.items = self.data
            
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]

def extract_embeddings(data_loader, model):
    """Extract embeddings for all items in data loader
    :param data_loader: DataLoader containing text data
    :param model: SentenceTransformer model
    :return: tensors of embeddings, document IDs, and languages
    """
    model.eval()
    all_embeddings, doc_ids, langs = [], [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing batches"):
            embeddings = model.encode(batch['text'], device=model.device, 
                                     batch_size=512, convert_to_tensor=True).cpu()
            all_embeddings.extend(embeddings)
            doc_ids.extend(batch['docid'])
            langs.extend(batch['lang'])
    
    return torch.stack(all_embeddings), doc_ids, langs

def main():
    USE_CHUNKS = False
    model_name = "Alibaba-NLP/gte-multilingual-base"
    
    model = SentenceTransformer(model_name, trust_remote_code=True)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Model loaded on {model.device}")    
    
    dataset = TextDataset('data/corpus.json', use_chunks=USE_CHUNKS, 
                         words_per_chunk=500 if USE_CHUNKS else None)
    print(f"Loaded {len(dataset)} {'chunks' if USE_CHUNKS else 'documents'}")
    
    data_loader = DataLoader(dataset, batch_size=512 if USE_CHUNKS else 100, shuffle=False)
    
    embeddings, doc_ids, langs = extract_embeddings(data_loader, model)
    torch.save({'embeddings': embeddings, 'docids': doc_ids, 'langs': langs}, 'embeddings_gte.pt')
    print(f"Saved {len(embeddings)} embeddings to embeddings_gte.pt")

if __name__ == "__main__":
    main()