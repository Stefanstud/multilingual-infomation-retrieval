import json
import pandas as pd
import numpy as np
from bm25 import BM25ChunkRetriever
from itertools import product

def evaluate_retrieval(queries, corpus, retriever):
    """
    Evaluate retrieval performance per language with both micro and macro averaging
    """
    results = retriever.retrieve(queries, corpus)

    results_per_lang = {}
    total_recall_1 = 0
    total_top_10 = 0
    total_queries = 0

    for lang, lang_data in queries.groupby('lang'):
        recall_at_1 = 0
        top_10_accuracy = 0
        lang_queries = len(lang_data)
        total_queries += lang_queries

        for i, row in lang_data.iterrows():
            positive_doc = row['positive_docs']
            predicted_docs = results[i]

            if positive_doc == predicted_docs[0]:
                recall_at_1 += 1
                total_recall_1 += 1
            if positive_doc in predicted_docs:
                top_10_accuracy += 1
                total_top_10 += 1

        results_per_lang[lang] = {
            'recall_at_1': recall_at_1 / lang_queries,
            'top_10_accuracy': top_10_accuracy / lang_queries,
            'num_queries': lang_queries
        }

    # Macro average (average of language-level scores)
    macro_avg = {
        'recall_at_1': np.mean([v['recall_at_1'] for k, v in results_per_lang.items()]),
        'top_10_accuracy': np.mean([v['top_10_accuracy'] for k, v in results_per_lang.items()]),
        'averaging': 'macro'
    }

    # Micro average (average across all queries)
    micro_avg = {
        'recall_at_1': total_recall_1 / total_queries,
        'top_10_accuracy': total_top_10 / total_queries,
        'averaging': 'micro'
    }

    # weighted average (en has 0.8 weight, and each of the other languages has 0.2 weight)
    weighted_avg = {
        'recall_at_1': 0.4 * results_per_lang['en']['recall_at_1'] + 0.6 * np.mean([v['recall_at_1'] for k, v in results_per_lang.items() if k != 'en']),
        'top_10_accuracy': 0.8 * results_per_lang['en']['top_10_accuracy'] + 0.6 * np.mean([v['top_10_accuracy'] for k, v in results_per_lang.items() if k != 'en']),
        'averaging': 'weighted'
    }
        

    results_per_lang['macro_average'] = macro_avg
    results_per_lang['micro_average'] = micro_avg
    results_per_lang['weighted_average'] = weighted_avg

    return results_per_lang

def grid_search_bm25(dev_data, corpus, k1_range, b_range, stopwords_path):
    """
    Perform grid search for BM25 hyperparameters
    
    Args:
        dev_data: DataFrame containing development queries
        corpus: Dictionary containing the document corpus
        k1_range: List of k1 values to try
        b_range: List of b values to try
        stopwords_path: Path to stopwords file
        
    Returns:
        best_params: Dictionary containing best parameters
        best_results: Results with best parameters
        all_results: DataFrame containing all evaluation results
    """
    results_list = []
    best_macro_score = -1
    best_micro_score = -1
    best_macro_params = None
    best_micro_params = None
    best_macro_results = None
    best_micro_results = None

    # Try all combinations of parameters
    for k1, b in product(k1_range, b_range):
        retriever = BM25ChunkRetriever(
            stopwords_path=stopwords_path,
            k1=k1,
            b=b
        )
        
        results = evaluate_retrieval(dev_data, corpus, retriever)
        macro_metrics = results['macro_average']
        micro_metrics = results['micro_average']
        
        results_list.append({
            'k1': k1,
            'b': b,
            'macro_recall_at_1': macro_metrics['recall_at_1'],
            'macro_top_10_accuracy': macro_metrics['top_10_accuracy'],
            'micro_recall_at_1': micro_metrics['recall_at_1'],
            'micro_top_10_accuracy': micro_metrics['top_10_accuracy']
        })
        
        print(f"k1: {k1:.2f}, b: {b:.2f}")
        print("Macro avg:", macro_metrics)
        print("Micro avg:", micro_metrics)
        
        # Track best macro scores
        if macro_metrics['top_10_accuracy'] > best_macro_score:
            print(f"New best macro score: {macro_metrics['top_10_accuracy']:.4f}")
            best_macro_score = macro_metrics['top_10_accuracy']
            best_macro_params = {'k1': k1, 'b': b}
            best_macro_results = results
            
        # Track best micro scores
        if micro_metrics['top_10_accuracy'] > best_micro_score:
            print(f"New best micro score: {micro_metrics['top_10_accuracy']:.4f}")
            best_micro_score = micro_metrics['top_10_accuracy']
            best_micro_params = {'k1': k1, 'b': b}
            best_micro_results = results

    all_results = pd.DataFrame(results_list)
    
    return {
        'macro': {
            'params': best_macro_params,
            'results': best_macro_results,
            'score': best_macro_score
        },
        'micro': {
            'params': best_micro_params,
            'results': best_micro_results,
            'score': best_micro_score
        },
        'all_results': all_results
    }

def main():
    # Load data
    with open('../data/corpus.json', 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    dev_data = pd.read_csv('../data/dev.csv')
    

    ## For evaluation
    # retriever = BM25ChunkRetriever(
    #     stopwords_path='../data/stopwords-ko.txt'
    # )
    # results = evaluate_retrieval(dev_data, corpus, retriever)
    # print("\n BM25 Results per Language:")
    # print(pd.DataFrame(results).T)

    ## For grid search
    # Define parameter ranges
    k1_range = np.arange(0.7, 1.9, 0.3)  # 0.7 to 2.0 with 0.3 increments
    b_range = np.array([0.3, 0.5, 0.75, 0.9])  # Common values focused around typical good performance

    # Perform grid search
    results = grid_search_bm25(
        dev_data,
        corpus,
        k1_range,
        b_range,
        stopwords_path='../data/stopwords-ko.txt'
    )   
    
    # Print results
    print("\nGrid Search Results:")
    
    print("\nBest Macro-Average Parameters:")
    print(f"k1: {results['macro']['params']['k1']:.2f}")
    print(f"b: {results['macro']['params']['b']:.2f}")
    print(f"Score: {results['macro']['score']:.4f}")
    
    print("\nBest Micro-Average Parameters:")
    print(f"k1: {results['micro']['params']['k1']:.2f}")
    print(f"b: {results['micro']['params']['b']:.2f}")
    print(f"Score: {results['micro']['score']:.4f}")
    
    # Print top 5 results for both metrics
    print("\nTop 5 Parameter Combinations (Macro):")
    print(results['all_results'].sort_values('macro_top_10_accuracy', ascending=False).head())
    
    print("\nTop 5 Parameter Combinations (Micro):")
    print(results['all_results'].sort_values('micro_top_10_accuracy', ascending=False).head())
    
    # Save results to CSV
    results['all_results'].to_csv('grid_search_results.csv', index=False)

if __name__ == "__main__":
    main()