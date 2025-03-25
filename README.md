# Multilingual retrieval

A from-scratch implementation of the BM25 algorithm for information retrieval. The algorithm is implemented using sparse matrix operations for memory-efficient computation and supports document chunking for processing large docs. Even though chunking might not be very intuitive to use with BM25, on the Kaggle competition (which this repo was built for) this gave the best results. There is also an attempt with bge-m3, however due to comp. resources we didnt use it. 

**Repository Structure**
```
multilingual-infomation-retrieval/
├── bm25/                 
│   ├── bm25.py            # the from scratch implementation
│   ├── evaluate_bm25.py   # eval scripts
│   └── submit_bm25.py     # should give you intution on how to use the retriever
├── data/                  # data files
│   ├── corpus-small.json
│   └── stopwords-ko.txt
├── misc/                  # aux. functions
│   ├── embeddings.py
│   └── tfidf.py
└── results/               # hyperparameter optimization results 
```

## Supported Languages

- English (en)
- French (fr)
- German (de)
- Arabic (ar)
- Spanish (es)
- Italian (it)
- Korean (ko)