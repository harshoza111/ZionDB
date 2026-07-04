# ZionDB Collections & Public API

This document details the design, architecture, and responsibilities of the public-facing ZionDB API, introduced to make ZionDB usable via named Collections and a Retriever orchestrator.

---

## Why Collections Exist

In vector databases like Qdrant, Milvus, or Pinecone, **Collections** (analogous to tables in relational databases) partition different sets of documents. 

In ZionDB, Collections:
1. **Provide Isolation**: Documents and vectors inside one collection do not interfere with, appear in, or affect queries in another.
2. **Encapsulate Complexity**: Users can write queries and insert documents without dealing with pipelines, embedding providers, storage, or vector indices.
3. **Offer Modular Configurations**: Each collection can use distinct chunking algorithms, splitters, embedding models, or similarity metrics customized for its domain.

---

## Ownership Hierarchy

ZionDB implements strict dependency ownership. The user interacts with the top-level database or collection instance. All underlying storage and search mechanics are hidden implementation details:

```
                            ZionDB (Entrypoint)
                                     │
                           [CollectionManager]
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
           Collection("books")              Collection("papers")
                    │
    ┌──────────────┬──────────────┼──────────────┬──────────────┐
    ▼              ▼              ▼              ▼              ▼
 [Storage]   [VectorIndex]   [Retriever]    [Pipeline]    [Persistence]
```

* **ZionDB**: The global database entry point. Owns the `CollectionManager`.
* **CollectionManager**: Manages collection registration, existence, lookup, and deletion.
* **Collection**: The primary user-facing engine interface. Owns:
  - `Storage` (Data Storage)
  - `VectorIndex` (Nearest Neighbor Index)
  - `Retriever` (Orchestration Engine)
  - `EmbeddingProvider` (Embedding Model Session)
  - `DocumentIndexingPipeline` (Chunking and Text Processing Pipeline)
  - `CollectionPersistence` (Collection Persistence Coordinator)


---

## Retriever Responsibilities

The `Retriever` is an **orchestration layer**. It contains **no indexing or storage logic**, and **no metadata filtering**. 
Its execution sequence is:
1. **Receive SearchRequest**: Accepts a query string and parameter overrides.
2. **Compute Query Embedding**: Sends the query string to the `EmbeddingProvider`.
3. **Index Query**: Calls `VectorIndex.search` with the query embedding vector.
4. **Fetch ChunkRecords**: Takes the matching IDs and fetches the corresponding records from the storage layer.
5. **Construct SearchResult**: Packages the matched `ChunkRecord` and similarity score together into a list of public `SearchResult` objects and returns them.

---

## Public ZionDB API Examples

### Database Initialization
```python
from ziondb import ZionDB, CollectionConfig

# Initialize database
db = ZionDB()
```

### Collection Creation & Config
```python
# Custom config
config = CollectionConfig(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    similarity_metric="cosine",
    splitter_type="regex",
    buffer_size=1,
    threshold_value=50.0
)

# Create a collection
books = db.create_collection("books", config=config)
```

### Ingesting Documents
```python
# Add single document (returns document ID)
doc_id = books.add_document(
    text="ZionDB is a modular vector database built from scratch.",
    metadata={"author": "Harsh Oza"},
    id="doc_01"
)

# Overwrite/Upsert document (calling again with same ID replaces old chunks)
books.add_document(
    text="ZionDB is an educational modular vector database built from scratch.",
    metadata={"author": "Harsh Oza"},
    id="doc_01"
)
```

### Querying Similarity
```python
# Search similarity
results = books.search(
    query="What is ZionDB?",
    top_k=2
)

for result in results:
    print(f"Text: {result.record.text}")
    print(f"Score: {result.score}")
```

---

## Future Compatibility

This design facilitates future extensions without requiring changes to the public API:

1. **Local Persistence**: To add persistence (SQLite or raw serialization), a new storage implementation (`SQLiteStorage`) can be plugged in under `Collection` by updating the config (`storage_type = "sqlite"`). The collection public API methods remain identical.
2. **ANN/HNSW Index**: To support approximate nearest neighbors, we can implement an `HNSWIndex` matching the `VectorIndex` interface and hook it up by mapping `index_type = "hnsw"` in the configuration.
3. **Metadata Filtering**: When metadata indexes are introduced, the `Retriever` can intercept the `metadata_filter` attribute in `SearchRequest` and pass it to the index search query or apply pre-filtering.
