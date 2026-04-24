# Vector Store

This directory now stores the lightweight persisted RAG index used by the app.

Expected artifacts include:

- `manifest.json` with source-signature and index metadata
- `chunks.jsonl` with chunked retrieval documents
- `vectorizer.joblib` for TF-IDF query encoding
- `tfidf_matrix.joblib` for lexical retrieval
- `svd.joblib` when dense latent embeddings are available
- `dense_embeddings.npy` for cosine-scored chunk retrieval

The index is rebuilt automatically when the structured files in `rag_data/raw/`
or `rag_data/processed/` change.
