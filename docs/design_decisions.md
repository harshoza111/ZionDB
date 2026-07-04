# ZionDB Architectural Decision Records (ADRs)

This document explains the major architectural choices made during the bootstrapping and construction of ZionDB.

---

## 1. Dataclasses with `slots=True` vs Pydantic

* **Decision**: Use native Python `@dataclass(slots=True)` rather than `Pydantic` models.
* **Rationale**: 
  - ZionDB is an educational vector database focused on low latency and transparency. Pydantic adds a heavy runtime dependency and overhead due to input validation.
  - Using `slots=True` (introduced in Python 3.10) provides faster attribute access and significant memory savings by preventing the creation of `__dict__` and `__weakref__` for instances. Since a database may process millions of sentence and chunk representations, minimizing memory footprint is critical.
* **Alternatives Considered**:
  - *Pydantic*: Rejected due to library weight and computational overhead.
  - *Standard Dicts*: Rejected as they lack type safety, autocompletion support, and fail to prevent attribute typos.

---

## 2. No PyTorch Runtime Dependency for Embeddings

* **Decision**: Load and run embedding models directly via `onnxruntime` and `transformers` tokenizers without installing PyTorch (`torch`).
* **Rationale**:
  - PyTorch is extremely heavy (several gigabytes) and is optimized for GPU training. For local indexing and CPU inference (essential for a lightweight vector DB), it is highly redundant.
  - ONNX Runtime is lightweight, loads quickly, and performs highly optimized CPU vector operations (often faster than raw PyTorch on CPU).
  - The Hugging Face repo `sentence-transformers/all-MiniLM-L6-v2` contains a pre-exported ONNX version under `onnx/model.onnx`. We download it directly using `huggingface_hub` snapshotting.
* **Alternatives Considered**:
  - *Standard PyTorch Pipeline*: Rejected due to package bloat and slow cold-starts.

---

## 3. Blank spaCy English Model with Sentencizer Component

* **Decision**: Default to a blank spaCy English pipeline with the `sentencizer` pipe, falling back to a custom `RegexSentenceSplitter`.
* **Rationale**:
  - A full spaCy model (like `en_core_web_sm`) is robust but requires downloads at runtime (12MB+) and a POS tagger, which is slow.
  - A blank spaCy pipeline with only the `sentencizer` component uses spaCy's optimized rule-based sentence segmentation. It does not require downloading external files, executes instantly, and is highly robust.
  - If users prefer rule-less regex partitioning, `RegexSentenceSplitter` is fully self-contained.
* **Alternatives Considered**:
  - *NLTK*: Rejected because `nltk.download('punkt')` requires manual workspace bootstrapping and runtime network requests.

---

## 4. Double-Method Kamradt Boundary Detection (Text Re-Embedding vs Vector Averaging)

* **Decision**: Implement a double-method approach where the boundary detector can either:
  1. Group sentences as text and call the embedding provider to re-embed (official Kamradt method).
  2. Average individual sentence vectors mathematically (high-performance approximation).
* **Rationale**:
  - The official Kamradt method concatenates text windows (e.g. sentences $i-1, i, i+1$) and feeds them to the embedding model. This allows the model to compute context-aware embeddings of the combined text but requires $2k+1$ times more text embeddings, which is slow.
  - For large documents, mathematical averaging of individual sentence embeddings (weighted by length or mean) is extremely fast (zero additional ONNX runs).
  - Offering both methods through the `KamradtBoundaryDetector` constructor provides maximum educational and optimization flexibility.

---

## 5. Read-Only `RecordProvider` Interface for Vector Index

* **Decision**: Introduce the `RecordProvider` interface to represent a read-only view of the storage layer and make `VectorIndex` depend on it instead of the full read-write `Storage` class.
* **Rationale**:
  - **Interface Segregation Principle**: The index only needs to retrieve data during initialization or rebuilds—it has no business performing storage insertions, updates, deletions, or clearing data.
  - **Security & Separation**: Segregating read operations protects database state integrity and prevents the indexing logic from accidentally changing storage states.
  - **Hot-Swappable Back-ends**: Since `Storage` inherits from `RecordProvider`, any storage implementation (e.g. `InMemoryStorage`, SQLite, disk files) automatically implements `RecordProvider`.
* **Alternatives Considered**:
  - *Direct Storage Dependency*: Rejected because it violates interface segregation and couples the index to write-mutation APIs.

---

## 6. Decoupled Similarity Metrics and `SearchResult` Models

* **Decision**: Abstract similarity calculations using a `SimilarityMetric` interface and return search matches via a decoupled `SearchResult` model containing only ID and score.
* **Rationale**:
  - **Open/Closed Principle**: Decoupling distance calculations from the search algorithm allows adding metrics (like Euclidean Distance or Dot Product) without changing index code.
  - **Sorting Abstraction**: The `greater_is_better` property on `SimilarityMetric` cleanly delegates sort directions (cosine/dot product values sort descending; Euclidean distances sort ascending) back to the metric itself.
  - **Memory Decoupling**: Returning `SearchResult` containing only the ID and score avoids moving heavy document strings and metadata payloads onto the heap during similarity sorting.

---

## 7. Collection-Level Partitioning and Component Ownership

* **Decision**: Make each `Collection` an independent entity that owns its local `Storage`, `VectorIndex`, `Retriever`, `EmbeddingProvider`, and `DocumentIndexingPipeline` instances.
* **Rationale**:
  - **Modular Decoupling**: This allows different collections to have completely different model setups, distance metrics, and chunking parameters.
  - **Simple Multi-Tenancy**: Collections are isolated in memory. A query on Collection A has zero exposure to data in Collection B, preventing leakage.
  - **Single Responsibility**: `CollectionManager` is responsible only for lifecycles, and `Collection` is responsible for document/search logic.

---

## 8. Public-Facing SearchResult Model

* **Decision**: Create a distinct public-facing `SearchResult` class in the `retrieval` package containing the full `ChunkRecord` and score, instead of returning raw IDs.
* **Rationale**:
  - **Encapsulation**: Prevents leaking the database's internal record ID schemes (e.g. `{doc_id}#chunk_{chunk_index}`) to users.
  - **Ease of Use**: Users immediately receive the chunk text and user metadata, without needing to make subsequent database lookup calls to get the text of the matched chunk.

---

## 9. Versioned Binary Record Serializer and Decoupled Persistence

* **Decision**: Implement record persistence using a versioned, little-endian binary Length-Value format (`records.bin`), keeping serialization completely decoupled from core database classes.
* **Rationale**:
  - **Single Responsibility Principle**: Core business logic classes (`Collection`, `Storage`, `VectorIndex`) do not know anything about binary formatting, file layouts, or disk paths. They delegate persistence to `CollectionPersistence` and its specialized serializers.
  - **Open-Closed Principle (Dynamic Serializer Registry)**: Using a dynamic registry to map `CollectionConfig.index_type` to concrete `IndexSerializer` classes allows hot-swapping or adding serializers (e.g. `HNSWIndexSerializer`) without editing orchestration logic.
  - **Serialization Performance**: Embeddings are written directly as raw IEEE-754 `float32` bytes (not text/JSON), avoiding parsing/string conversion overhead and keeping file sizes small.
  - **Backward Compatibility**: A magic byte header (`b'ZION'`) and a file format version (`uint16`) ensure we can support file migrations in future releases.

---

## 10. Retrieval Architecture Refactoring (Phase 1)

* **Decision**: Decouple query parameters from internal execution using `SearchContext`, introduce a structured `FilterExpression` AST for logical/comparison metadata filters, and delegate filtering strategy to `VectorIndex` implementations.
* **Rationale**:
  - **SearchRequest (Public API) vs SearchContext (Internal Execution)**: `SearchRequest` represents what the user asked for (query string, top_k, raw dict filters). `SearchContext` contains what the retrieval engine actually needs to execute the search (pre-computed vector embeddings, parsed `FilterExpression` AST, resolved metric overrides). This separation keeps embedding generation and AST translation isolated in the `Retriever` layer, ensuring future parameters (like candidate IDs, search planning overrides, timeouts) can be added to `SearchContext` without changing the public-facing API.
  - **FilterExpression AST**: Representing metadata filters as an abstract syntax tree (`EqualFilter`, `AndFilter`, `OrFilter`, `NotFilter`) rather than raw dictionaries allows complex boolean logic (AND, OR, NOT nested recursively) and comparison operations to be evaluated in a type-safe, extensible structure.
  - **Index-Side Metadata Filtering**:
    - Conceptually, metadata filtering must happen *before* or *during* similarity computation to avoid return set depletion (which happens if vector similarity is run first and results are filtered post-search).
    - However, how metadata is stored and evaluated differs drastically across index layouts. For example, `BruteForceIndex` maintains an in-memory metadata cache and does simple O(1) checks. A future `HNSWIndex` might check filters during graph traversal (single-stage payload-aware traversal) or use separate bitmap/metadata indices to pre-select entry points.
    - Thus, the Retriever must NOT own filtering logic. It passes the `FilterExpression` inside the `SearchContext` to the `VectorIndex`, allowing each index implementation to optimize and execute filters natively.
  - **Future Preparedness**: This architecture establishes direct extension points:
    - **HNSW / BM25**: Future vector/keyword search indices can expose the same `search(context: SearchContext)` signature, accepting filters and applying them custom-tailored to their storage layouts.
    - **Hybrid Search**: Result fusion (like RRF) can merge separate candidate streams produced by indices using the same context.
    - **Reranking**: An optional reranking stage can easily intercept the hydrated results list after the index search completes.

---

## 11. Retrieval Pipeline Refactoring (Phase 2)

* **Decision**: Decompose the `Retriever` into Query Embedding, Vector Search, Hydration services, and a PostProcessor interface, avoiding a generic pipeline framework.
* **Rationale**:
  - **Single Responsibility Principle (SRP)**: The `Retriever` is now a pure orchestrator containing no business logic. The retrieval lifecycle stages are cleanly separated:
    - `QueryEmbeddingService`: Pre-processing (embedding and context building).
    - `VectorSearchService`: Index interaction.
    - `HydrationService`: Storage interaction.
    - `PostProcessor`: Post-search result mutation (initially `IdentityPostProcessor`).
  - **Avoiding Framework Bloat**: We intentionally rejected building a generic pipeline runner or stage graph framework. Such frameworks add unnecessary layers of abstraction, are hard to debug, and make the code less readable for learning and research. Hardcoded, domain-specific service references inside a dedicated coordinator (`Retriever`) are highly transparent, self-documenting, and maintain strong compiler/IDE typing checks.
  - **Future Extensibility**:
    - **HNSW / BM25**: Future vector indexes can plug directly into the `VectorSearchService` or subclass it (e.g. `HybridSearchService`), query multiple index layouts, and merge results before returning them to `Retriever`.
    - **Reranking, Pagination, Thresholding**: These future stages fit naturally as custom `PostProcessor` implementations (e.g. `RerankingPostProcessor`) without changing the `Retriever`'s core orchestrating pipeline.

---

## 12. SearchExecutor Abstraction (Phase 3)

* **Decision**: Introduce a `SearchExecutor` interface and move the retrieval execution logic out of the `Retriever` and into the concrete `DenseSearchExecutor`.
* **Rationale**:
  - **Strategy Pattern / Open-Closed Principle**: The `Retriever` class is now completely decoupled from specific retrieval workflows. By depending only on the abstract `SearchExecutor` interface, the `Retriever` class does not change when we introduce alternative retrieval strategies (e.g., hybrid dense/sparse search, multi-stage retrieval, or query routing).
  - **DenseSearchExecutor**: Encapsulates the standard dense vector similarity search pipeline. It coordinates the pre-retrieval embedding and parsing services, queries the vector index, hydrates the matching candidate records from storage, and applies any post-retrieval processing steps.
  - **Extensibility**: Adding new retrieval strategies like hybrid search will simply involve implementing a `HybridSearchExecutor` (which executes both vector and keyword queries, fuses their outputs, hydrates matches, and optionally reranks them) and injecting it into the `Retriever` constructor.


