import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import numpy as np
from sklearn.metrics import average_precision_score
from utils import RetrievalDataset, load_model_and_tokenizer, tokenize_batch, load_data

class ContrastiveLoss(nn.Module):
    """ InfoNCE Loss function """
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, query_embeddings, positive_embeddings, negative_embeddings):
        pos_sim = self.similarity(query_embeddings, positive_embeddings)
        neg_sim = self.similarity(query_embeddings.unsqueeze(1), negative_embeddings).squeeze()
        
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        logits /= self.temperature
        
        # We expect the first column to be the positive sample
        labels = torch.zeros(query_embeddings.size(0), dtype=torch.long).to(query_embeddings.device)
        
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss

def evaluate(model, eval_data, docid_to_text, tokenizer, device, batch_size=4):
    model.eval()
    dataset = RetrievalDataset(eval_data, docid_to_text)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for queries, positive_docs, negative_docs in dataloader:
            query_tokens = tokenize_batch(queries, tokenizer, device)
            positive_tokens = tokenize_batch(positive_docs, tokenizer, device)
            negative_tokens = tokenize_batch([doc for docs in negative_docs for doc in docs], tokenizer, device)
            
            # .pooler_output is using the [CLS], as far as I understand 
            query_embeddings = model(**query_tokens).pooler_output
            positive_embeddings = model(**positive_tokens).pooler_output
            negative_embeddings = model(**negative_tokens).pooler_output.view(len(queries), 20, -1)
            
            scores = torch.cat([
                torch.bmm(query_embeddings.unsqueeze(1), positive_embeddings.unsqueeze(2)).squeeze(),
                torch.bmm(query_embeddings.unsqueeze(1), negative_embeddings.transpose(1, 2)).squeeze()
            ], dim=1)
            
            all_scores.append(scores.cpu().numpy())
            all_labels.append(np.concatenate([np.ones((len(queries), 1)), np.zeros((len(queries), 20))], axis=1))
    
    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    map_score = average_precision_score(all_labels.flatten(), all_scores.flatten())
    mrr = np.mean([1 / (np.where(row == 1)[0][0] + 1) for row in all_labels])
    
    return {"MAP": map_score, "MRR": mrr}

def train(model, train_data, eval_data, docid_to_text, tokenizer, device, config):
    wandb.init(project="multilingual-retrieval", config=config)
    
    # Load data
    train_dataset = RetrievalDataset(train_data, docid_to_text)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # Initialize loss and optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = ContrastiveLoss(temperature=config.temperature)
    
    best_map = 0
    
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (queries, positive_docs, negative_docs) in enumerate(train_dataloader):
            query_tokens = tokenize_batch(queries, tokenizer, device)
            positive_tokens = tokenize_batch(positive_docs, tokenizer, device)
            negative_tokens = tokenize_batch([doc for docs in negative_docs for doc in docs], tokenizer, device)
            
            query_embeddings = model(**query_tokens).pooler_output
            positive_embeddings = model(**positive_tokens).pooler_output
            negative_embeddings = model(**negative_tokens).pooler_output.view(len(queries), 20, -1)
            
            loss = loss_fn(query_embeddings, positive_embeddings, negative_embeddings)
            running_loss += loss.item()
            
            loss = loss / config.accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % config.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            wandb.log({"batch_loss": loss.item()})
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{config.num_epochs}], Step [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_dataloader)
        print(f'Epoch [{epoch+1}/{config.num_epochs}], Average Loss: {epoch_loss:.4f}')
        wandb.log({"epoch_loss": epoch_loss})
        
        # Evaluate the model
        eval_results = evaluate(model, eval_data, docid_to_text, tokenizer, device, config.eval_batch_size)
        print(f"Evaluation results: {eval_results}")
        wandb.log(eval_results)
        
        if eval_results["MAP"] > best_map:
            best_map = eval_results["MAP"]
            torch.save(model.state_dict(), f"{wandb.run.dir}/best_model.pt")
            wandb.save(f"{wandb.run.dir}/best_model.pt")
            print(f"New best model saved with MAP: {best_map}")
    
    wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--model_name", type=str, default="Alibaba-NLP/gte-multilingual-base")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.07)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    model.to(device)

    train_data, eval_data, docid_to_text = load_data(args.data_path)
    
    train(model, train_data, eval_data, docid_to_text, tokenizer, device, args)