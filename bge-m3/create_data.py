import json
import pandas as pd
import ast
from tqdm import tqdm
from pathlib import Path

def load_corpus(corpus_path):
    """Load the document corpus and create a mapping of document IDs to text."""
    print("Loading corpus...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    doc_map = {doc['docid']: doc['text'] for doc in corpus}
    print(f"Loaded {len(doc_map)} documents")
    return doc_map

def create_training_data(train_csv_path, doc_map, output_path):
    """Create training data in the required format."""
    print("Loading and processing training data...")
    
    df = pd.read_csv(train_csv_path)
    print(f"Loaded {len(df)} training examples")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            pos_doc_id = row['positive_docs']
            pos_text = doc_map.get(pos_doc_id)
            
            if not pos_text:
                print(f"Warning: Missing positive document {pos_doc_id}")
                continue
                
            neg_doc_ids = ast.literal_eval(row['negative_docs'])
            
            neg_texts = []
            for neg_id in neg_doc_ids:
                neg_text = doc_map.get(neg_id)
                if neg_text:
                    neg_texts.append(neg_text)
                else:
                    print(f"Warning: Missing negative document {neg_id}")
            
            if not neg_texts:
                print(f"Warning: No valid negative documents for query {row['query_id']}")
                continue
            
            instance = {
                "query": row['query'], 
                "pos": [pos_text],
                "neg": neg_texts
            }
            
            f.write(json.dumps(instance, ensure_ascii=False) + '\n')

def main():
    corpus_path = '../data/corpus.json'
    train_csv_path = '../data/train.csv'
    output_path = '../data/training_data.jsonl'
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc_map = load_corpus(corpus_path)
    create_training_data(train_csv_path, doc_map, output_path)
    
    print(f"Training data has been written to {output_path}")
    print("\nFirst few lines of the output file:")
    with open(output_path, 'r', encoding='utf-8') as f:
        for _ in range(3):
            print(f.readline().strip())

if __name__ == "__main__":
    main()