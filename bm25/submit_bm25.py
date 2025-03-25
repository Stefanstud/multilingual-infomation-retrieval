import json
import pandas as pd
from bm25 import BM25ChunkRetriever

def main():
    with open('../data/corpus.json', 'r', encoding='utf-8') as f:
        corpus = json.load(f)
        
    test_data = pd.read_csv('../data/test.csv')
    retriever = BM25ChunkRetriever(stopwords_path='../data/stopwords-ko.txt')
    retriever.create_submission_csv(test_data, corpus, 'bm25_submission.csv')
    print("Submission file created successfully")

if __name__ == "__main__":
    main()