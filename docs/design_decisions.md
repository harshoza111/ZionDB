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



