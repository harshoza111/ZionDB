# ZionDB Vector Index Architecture

This document describes the design, abstractions, and future indexing support for the **ZionDB Vector Index Layer (Version 1)**.

---

## Architecture Overview

The vector index layer abstracts vector similarity searches, separating the search index layout from raw data persistence.

```
                  ┌──────────────────────┐
                  │    Retriever / Query │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ VectorIndex (ABC)    │
                  └──────────┬───────────┘
                             │
                     ┌───────┴───────┐
                     ▼               ▼
             [BruteForceIndex]   [Future HNSWIndex]
                     │               │
                     └───────┬───────┘
                             │ (Iterate and load)
                             ▼
                  ┌──────────────────────┐
                  │ RecordProvider (ABC) │
                  └──────────┬───────────┘
                             │ (Implemented by)
                             ▼
                  ┌──────────────────────┐
                  │   InMemoryStorage    │
                  └──────────────────────┘
```

---

## Core Design Principles & Abstractions

### 1. Read-Only Storage Access (`RecordProvider`)

Rather than letting the `VectorIndex` depend directly on the full read-write `Storage` class, we introduced the read-only `RecordProvider` interface.

- **Interface Segregation**: The index only needs to retrieve, iterate, check, and count records—it does not need to mutate records (`insert`, `update`, `delete`, or `clear`).
- **Safety**: Restricting the index to a read-only provider contract prevents code defects like accidental record modification or deletion during indexing operations.
- **Flexibility**: Any backend can implement `RecordProvider` (e.g. SQLite database read-only views, flat binary files) allowing the index to operate without modification.

### 2. Search Result Isolation (`SearchResult`)

The search method `VectorIndex.search()` returns a list of `SearchResult` objects containing only:
- `record_id: str`
- `score: float`

- **Decoupling**: The index layer does not return `ChunkRecord` objects. The index only manages numerical vector relationships and IDs.
- **Memory Optimization**: Avoids loading heavy text payloads or metadata on the heap during similarity sorting.
- **Single Source of Truth**: The storage layer remains the single source of truth for text and metadata. The retrieval orchestrator (retriever) receives IDs from the index and queries storage only for the specific matching chunks requested by the user.

### 3. Similarity Metric Abstraction (`SimilarityMetric`)

Calculating vector similarity is decoupled from the index query logic:
- `SimilarityMetric` defines the standard contract `calculate(v1, v2)`.
- It defines a `greater_is_better` boolean property specifying if scores should sort descending (like Cosine Similarity or Dot Product) or ascending (like Euclidean Distance).
- This design satisfies the **Open/Closed Principle (OCP)**; new metrics (e.g., Manhattan distance, dot product) can be hot-swapped into `BruteForceIndex` without changing the search logic.

---

## Algorithmic Walkthrough: `BruteForceIndex`

Version 1 uses flat brute-force calculation. While brute-force does not scale to sub-linear time, it serves as the baseline for correctness:

1. **Rebuild**: Iterates over all records from `RecordProvider` and caches the 1D embeddings in an in-memory map.
2. **Search**:
   1. Loops through all cached vectors.
   2. Calls the injected `SimilarityMetric` to compute scores.
   3. Wraps matches in `SearchResult` models.
   4. Sorts the results based on the metric's sorting preference.
   5. Yields the top-$K$ scoring items.

---

## Future Graph Index Integration: HNSW

The `VectorIndex` interface is designed to support graph-based Approximate Nearest Neighbors (ANN) like **HNSW (Hierarchical Navigable Small World)** in upcoming versions:

- **Incremental Sync**: An `HNSWIndex` will implement `insert(record)`, `remove(record_id)`, and `update(record)` to build and prune graph edges incrementally as storage mutates.
- **Bulk Build**: `rebuild()` will iterate through the `RecordProvider` to load all records and execute bulk graph entry insertion.
- **Search Routing**: `search(query_vector, top_k)` will execute graph search algorithms (routing from top layers down to base layers) instead of linear scans, returning the identical `SearchResult` models.

---

## Index Persistence & Serializers

Vector search indices may require storing complex graph or bucket structures to support durability:

- **`IndexSerializer`**: Defines the abstract interface for index-specific persistence routines.
- **`BruteForceIndexSerializer`**: The default concrete implementation. Because brute force indices contain no custom structural files (they evaluate raw vectors in real-time), `serialize` is a no-op. During `deserialize`, it triggers `index.rebuild()` to reconstruct the in-memory cache from record storage.
- **Extensibility**: When new indexes (like HNSW or IVF) are implemented, they will define dedicated serializers (e.g. `HNSWIndexSerializer` saving details to `hnsw.index`), keeping persistence isolated from retrieval runtime business logic.

