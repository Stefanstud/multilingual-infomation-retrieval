import json
import pandas as pd
from bm25_claude import BM25Retriever

def evaluate_retrieval(queries, corpus, retriever):
    """
    Evaluate retrieval performance per language
    """
    results = retriever.retrieve(queries, corpus)
    
    results_per_lang = {}
    for lang, lang_data in queries.groupby('lang'):
        recall_at_1 = 0
        top_10_accuracy = 0
        total_queries = len(lang_data)
        
        for i, row in lang_data.iterrows():
            positive_doc = row['positive_docs']
            predicted_docs = results[i]
            
            if positive_doc == predicted_docs[0]:
                recall_at_1 += 1
            if positive_doc in predicted_docs:
                top_10_accuracy += 1
                
        results_per_lang[lang] = {
            'recall_at_1': recall_at_1 / total_queries,
            'top_10_accuracy': top_10_accuracy / total_queries
        }
        
    return results_per_lang

def main():
    with open('../data/corpus.json', 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    dev_data = pd.read_csv('../data/dev.csv')
    
    retriever = BM25Retriever(
        stopwords_path='../data/stopwords-ko.txt'
    )
    
    results = evaluate_retrieval(dev_data, corpus, retriever)
    
    print("\nBM25 Results per Language:")
    print(pd.DataFrame(results).T)
    
if __name__ == "__main__":
    main()