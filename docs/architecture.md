# ZionDB Architecture

ZionDB is a modular, custom vector database and retrieval engine designed from scratch for educational and production portfolios. This document describes the architecture of **Version 1: The Indexing Pipeline**.

---

## Design Philosophy

1. **Separation of Concerns**: Each stage in the pipeline performs exactly one task.
2. **Interface-Driven Design**: The orchestrator depends exclusively on abstract interfaces. Implementations are fully swappable.
3. **Data Integrity**: Structured domain models (`dataclasses` with `slots=True`) are used for type safety and memory efficiency instead of loose dictionaries or strings.

---

## High-Level Component Layout

```mermaid
graph TD
    doc[TextDocument] --> splitter[SentenceSplitter]
    splitter --> sentences[list[Sentence]]
    sentences --> embedder1[EmbeddingProvider]
    embedder1 --> sent_embs[list[SentenceEmbedding]]
    
    sentences --> detector[BoundaryDetector]
    sent_embs --> detector
    detector --> decisions[list[BoundaryDecision]]
    
    sentences --> builder[ChunkBuilder]
    decisions --> builder
    builder --> chunks[list[Chunk]]
    
    chunks --> embedder2[ChunkEmbedder]
    embedder2 --> chunk_embs[list[ChunkEmbedding]]
    
    chunk_embs[list[ChunkEmbedding]] --> output[Indexable Objects]
```

---

## Component Responsibilities

### 1. Document Indexing Pipeline (`DocumentIndexingPipeline`)
- Coordinates the stages of the pipeline.
- Handles edge cases (such as empty texts or single sentences) before invoking heavy components.

### 2. Sentence Splitters (`SentenceSplitter`)
- **`RegexSentenceSplitter`**: Split sentences using punctuation rules and regex patterns.
- **`SpacySentenceSplitter`**: Splits sentences using spaCy's optimized rule-based `sentencizer`.
- Both extract exact character ranges (`start_char`, `end_char`) matching the original source text.

### 3. Model Manager & Embedding Providers (`ModelManager`, `EmbeddingProvider`)
- **`ModelManager`**: Handles downloading model files, local cache directories (`models/`), and initializing the CPU-bound ONNX session.
- **`ONNXEmbeddingProvider`**: Tokenizes input text strings, runs the ONNX model, performs Mean Pooling (incorporating attention mask), applies L2 Normalization, and yields normalized embeddings of size 384.

### 4. Boundary Detector (`BoundaryDetector`)
- **`KamradtBoundaryDetector`**: Groups consecutive sentences with a buffer size $k$, calculates cosine distances between adjacent groups, and flags semantic topic transitions exceeding a computed percentile or standard deviation threshold.

### 5. Chunk Builder (`ChunkBuilder`)
- **`SemanticChunkBuilder`**: Assembles sentences into final `Chunk` objects using boundary splits, generating unified chunk texts and copying character spans and source indices into metadata.

### 6. Chunk Embedder (`ChunkEmbedder`)
- **`ONNXChunkEmbedder`**: Computes the final embedding vectors for the generated chunks to prepare them for downstream indexing.
