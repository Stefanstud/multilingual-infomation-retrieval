import pandas as pd
import json
import ast
from sentence_transformers import SentenceTransformer, losses, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from datasets import Dataset
import os

def prepare_dataset(df, corpus_dict):
    """
    :param df: DataFrame with query data
    :param corpus_dict: Dictionary mapping docids to document text
    :return: Dataset with anchor-positive-negative triplets
    """
    data = {"anchor": [], "positive": [], "negative": []}
    for _, row in df.iterrows():
        query, positive_docid = row['query'], row['positive_docs']
        try:
            negative_docids = ast.literal_eval(row['negative_docs'])
            if positive_docid not in corpus_dict:
                continue
                
            negative_texts = [corpus_dict[docid] for docid in negative_docids 
                             if docid in corpus_dict]
            
            if not negative_texts:
                continue
                
            data["anchor"].append(query)
            data["positive"].append(corpus_dict[positive_docid])
            data["negative"].append(negative_texts)
            
        except (SyntaxError, ValueError):
            continue
            
    return Dataset.from_dict(data)

def main():
    with open('data/corpus.json', 'r', encoding='utf-8') as f:
        corpus_dict = {doc['docid']: doc['text'] for doc in json.load(f)}

    train_dataset = prepare_dataset(pd.read_csv('data/train.csv'), corpus_dict)
    eval_dataset = prepare_dataset(pd.read_csv('data/dev.csv'), corpus_dict)
    
    print(f"Prepared training dataset with {len(train_dataset)} samples")
    print(f"Prepared evaluation dataset with {len(eval_dataset)} samples")

    model_name = "Alibaba-NLP/gte-multilingual-base"
    model = SentenceTransformer(model_name, trust_remote_code=True)
    output_path = 'finetuned-gte-multilingual-base'
    os.makedirs(output_path, exist_ok=True)

    # config
    args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        run_name="gte-finetuning"
    )

    # train and save
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=losses.CachedMultipleNegativesRankingLoss(model)
    )

    trainer.train()
    model.save_pretrained(os.path.join(output_path, "final"))
    print(f"Fine-tuned model saved to: {os.path.join(output_path, 'final')}")

if __name__ == "__main__":
    main()