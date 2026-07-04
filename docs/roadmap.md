# ZionDB Long-Term Roadmap

This document outlines the milestones and roadmap for future iterations of ZionDB.

---

## Completed: Version 1 (Core Indexing Pipeline & Storage Base)
- [x] Interface-driven modular pipeline.
- [x] Standard `@dataclass(slots=True)` models.
- [x] HF download and CPU-optimized ONNX model management (no PyTorch).
- [x] Blank spaCy `sentencizer` sentence splitting.
- [x] Greg Kamradt's semantic similarity boundary detection.
- [x] YAML pipeline configuration.
- [x] Storage Interface and high-performance InMemoryStorage driver.

---

## Upcoming Milestones

### Version 2: Local Persistence & Retrieval
* **Objective**: Save/load indexes to disk and perform exact searches.
* **Scope**:
  - Implement SQLite/Disk storage driver (SQLiteStorage / DiskStorage).
  - Implement exact nearest neighbor search (brute-force flat L2 / cosine distance search).
  - Write index load/save APIs.

### Version 3: Approximate Nearest Neighbors (ANN) Indexing
* **Objective**: Scale lookup queries to sub-linear time.
* **Scope**:
  - Implement **HNSW (Hierarchical Navigable Small World)** graphs from scratch using numpy.
  - Implement **IVF (Inverted File Index)** partitioning.
  - Implement Product Quantization (PQ) for vector compression.

### Version 4: Query Engine & Hybrid Search
* **Objective**: Perform complex queries combining keyword and vector search.
* **Scope**:
  - Sparse text representation (TF-IDF or BM25).
  - Dense-sparse hybrid search fusion (e.g. Reciprocal Rank Fusion - RRF).
  - Metadata filtering (pre-filtering and post-filtering).

### Version 5: RAG Integration, APIs & CLI
* **Objective**: Complete end-to-end user-facing interfaces.
* **Scope**:
  - Fast API REST endpoints for query and indexing.
  - CLI utility to ingest files and run search queries.
  - Simple local web dashboard (using Streamlit or Next.js) showing chunk layouts and search results.
  - Integration interface for LLM calls (e.g. Gemini, OpenAI) to generate final answers.

### Version 6: Evaluation & Benchmarking
* **Objective**: Test performance and quality.
* **Scope**:
  - Ingest standard retrieval datasets.
  - Measure Recall@K, MRR, and NDCG.
  - Benchmark QPS (Queries Per Second) and build latency.
