import json
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['text']

def collate_fn(batch):
    # Tokenization and padding to the maximum length
    return tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=8192)

def extract_embeddings(data_loader, model):
    # Switch model to evaluation mode
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc="Processing batches")):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Get embeddings from the model
            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            # CLS:
            # embeddings = outputs.last_hidden_state[:, 0]
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings)

    # Concatenate all embeddings from batches
    return torch.cat(all_embeddings, dim=0)

# Load model and tokenizer
model_name = "Alibaba-NLP/gte-multilingual-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare the dataset and data loader
dataset = TextDataset('data/corpus.json')
data_loader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn, shuffle=False)

# Extract embeddings
embeddings = extract_embeddings(data_loader, model)

# Save embeddings to a .pt file
torch.save(embeddings, 'gte_multilingual_embeddings.pt')
print("Embeddings saved to embeddings.pt.")
