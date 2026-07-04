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
