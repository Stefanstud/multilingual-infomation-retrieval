import pandas as pd
import json
import ast  # For evaluating the string representation of lists
from sentence_transformers import SentenceTransformer, losses, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from datasets import Dataset
import os  # For creating directories



def prepare_dataset(df, corpus_dict):
    """Preprocesses a DataFrame (train or dev) into a Hugging Face Dataset."""
    queries = []
    positives = []
    negatives = []

    for _, row in df.iterrows():
        query_id = row['query_id']
        query = row['query']
        positive_docid = row['positive_docs']

        try:
            negative_docids = ast.literal_eval(row['negative_docs'])
        except (SyntaxError, ValueError):
            print(f"Warning: Invalid negative_docs format for query {query_id}. Skipping.")
            continue

        if positive_docid not in corpus_dict:
            print(f"Warning: Positive document {positive_docid} not found for query {query_id}. Skipping.")
            continue
        positive_text = corpus_dict[positive_docid]


        negative_texts = []
        for docid in negative_docids:
             if docid in corpus_dict: # check if negative id exists in corpus or not
                negative_texts.append(corpus_dict[docid])
        
        if not negative_texts:
            print(f"Warning: No Negative documents are found for query {query_id}. Skipping.")
            continue

        queries.append(query)
        positives.append(positive_text)
        negatives.append(negative_texts)


    return Dataset.from_dict({
        "anchor": queries,
        "positive": positives,
        "negative": negatives
    })

# # Load the corpus and create the corpus dictionary for faster lookups
with open('data/corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)
corpus_dict = {doc['docid']: doc['text'] for doc in corpus}

# Load and preprocess the train and dev datasets
train_df = pd.read_csv('data/train.csv')
train_dataset = prepare_dataset(train_df, corpus_dict)

dev_df = pd.read_csv('data/dev.csv')
eval_dataset = prepare_dataset(dev_df, corpus_dict)  # Use eval_dataset as this is the standard name in the trainer

# Load the pre-trained model
model_name = "Alibaba-NLP/gte-multilingual-base"
model = SentenceTransformer(model_name, trust_remote_code=True)

# Define loss, training arguments, and trainer (Following example closely)
loss = losses.CachedMultipleNegativesRankingLoss(model)

output_path = 'finetuned-gte-multilingual-base'

# Create the output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

args = SentenceTransformerTrainingArguments(
    output_dir=output_path,
    num_train_epochs=1, 
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8, 
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True, # if gpu is fp16 compatible
    bf16=False,
    evaluation_strategy="steps",
    eval_steps=100, # evaluate model on dev set every 100 steps
    save_strategy="steps",
    save_steps=100, # save model every 100 steps
    save_total_limit=2,  # Keep only the last 2 checkpoints
    logging_steps=100,
    run_name="gte-finetuning"
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, # Add the evaluation dataset
    loss=loss
)

# Train and save the model
trainer.train()
model.save_pretrained(os.path.join(output_path, "final")) # Save within output_path
print(f"Fine-tuned model saved to: {os.path.join(output_path, 'final')}")
