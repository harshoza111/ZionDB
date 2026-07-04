# ZionDB Long-Term Roadmap

This document outlines the milestones and roadmap for future iterations of ZionDB.

---

## Completed: Version 1 (Core Indexing, Storage & Indexing Base, Collections & Public API)
- [x] Interface-driven modular pipeline.
- [x] Standard `@dataclass(slots=True)` models.
- [x] HF download and CPU-optimized ONNX model management (no PyTorch).
- [x] Blank spaCy `sentencizer` sentence splitting.
- [x] Greg Kamradt's semantic similarity boundary detection.
- [x] YAML pipeline configuration.
- [x] Storage Interface and high-performance InMemoryStorage driver.
- [x] RecordProvider read-only interface abstraction.
- [x] Similarity metric abstraction and baseline BruteForceIndex driver.
- [x] Collection management layer and named Collections partition.
- [x] Retriever orchestrator mapping queries to index results and storage records.
- [x] Public-facing `ZionDB` API entry point.
- [x] Versioned binary record serializer (`records.bin`) using Length-Value layout.
- [x] Human-readable YAML configuration serializer (`config.yaml`).
- [x] Dynamic IndexSerializer interfaces and registry.
- [x] CollectionPersistence save/load orchestrator.
- [x] Collection API save and load integration.

---

## Upcoming Milestones

### Version 2: Alternative Storage Drivers & SQLite
* **Objective**: Introduce SQLite/Disk-backed storage drivers for scalable database operations.
* **Scope**:
  - Implement SQLite/Disk storage driver (`SQLiteStorage`).
  - Implement `SQLiteRecordSerializer` mapping records to database rows.



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
