# ZionDB Persistence Layer

This document details the architecture, design decisions, and binary file layouts of the ZionDB Persistence Layer (introduced in Version 1).

---

## Design Philosophy

The persistence layer is designed to be **completely decoupled from vector search and retrieval logic**. 

Persistence is responsible *only* for saving, loading, serializing, and deserializing collection state. It converts active runtime structures (like collections, configurations, and chunk records) to persistent files on disk and restores them back without knowing how similarity math or indexes operate under the hood.

---

## Persistence Architecture

ZionDB divides serialization responsibilities across dedicated, small classes orchestrated by a central coordinator:

```
                            Collection
                                │ (save() / load())
                                ▼
                      CollectionPersistence
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
   ConfigSerializer      RecordSerializer      IndexSerializer
    (config.yaml)         (records.bin)         (e.g., no-op)
```

### 1. `Collection` API
- Exposes high-level `save(path)` and `load(path)` methods.
- Delegates all disk operations directly to `CollectionPersistence`.

### 2. `CollectionPersistence` (Orchestrator)
- Coordinates directory setup and coordinates save/load operations.
- Interrogates `CollectionConfig` to resolve index and storage requirements.
- Uses a **Dynamic Serializer Registry** to look up the correct `IndexSerializer` class matching `config.index_type`, allowing the system to scale to HNSW, IVF, or PQ indexes without changes to this coordinator class.

### 3. `ConfigSerializer`
- Reads and writes `CollectionConfig` objects to a human-readable `config.yaml` using YAML format.
- Records system contexts including the collection name, persistence formatting version, and embedding dimension.

### 4. `RecordSerializer`
- Writes and parses the collections of `ChunkRecord` objects using a highly optimized, versioned binary format.

### 5. `IndexSerializer`
- Abstract interface defining vector index serialization.
- **`BruteForceIndexSerializer`**: The default concrete implementation. Because brute force indexes possess no persistent index structures (they scan raw records directly), saving is a no-op, and loading simply triggers an index reconstruction cache rebuild from the populated record storage.

---

## Collection Layout on Disk

Each collection is saved inside its own folder, facilitating simple database backups and isolation:

```text
my_database/
    books/
        config.yaml      # Human-readable collection configurations
        records.bin      # Binary serialized chunk records & embeddings
    papers/
        config.yaml
        records.bin
```

---

## Binary Record Format Specification

The `records.bin` file uses a custom, deterministic binary layout packing values using **little-endian (`<`)** formatting for CPU-native deserialization speed:

### File Header
| Field Name | Data Type | Size (Bytes) | Description |
| :--- | :--- | :--- | :--- |
| `magic` | `char[4]` | 4 | Magic file identifier bytes (`b'ZION'`) |
| `version` | `uint16` | 2 | Persistence format version (currently `1`) |
| `record_count` | `uint32` | 4 | Total number of chunk records in the file |

### Record Block (Repeated `record_count` times)
For each chunk record, fields are packed sequentially using a Length-Value (LV) pattern:

| Field Name | Data Type | Size (Bytes) | Description |
| :--- | :--- | :--- | :--- |
| `id_len` | `uint32` | 4 | Length of `record_id` string |
| `record_id` | `char[]` | `id_len` | UTF-8 encoded unique record ID |
| `text_len` | `uint32` | 4 | Length of chunk text string |
| `text` | `char[]` | `text_len` | UTF-8 encoded chunk text contents |
| `emb_len` | `uint32` | 4 | Number of floating-point values in embedding array |
| `embedding` | `float32[]`| `emb_len * 4` | Raw binary IEEE-754 float32 values |
| `meta_len` | `uint32` | 4 | Length of metadata JSON string |
| `metadata` | `char[]` | `meta_len` | UTF-8 JSON-serialized metadata dictionary |
| `doc_id_len` | `uint32` | 4 | Length of document ID string |
| `doc_id` | `char[]` | `doc_id_len` | UTF-8 encoded source document ID |
| `model_len` | `uint32` | 4 | Length of embedding model name string |
| `model_name` | `char[]` | `model_len` | UTF-8 encoded embedding model name |
| `method_len` | `uint32` | 4 | Length of chunking method name string |
| `chunk_method`| `char[]` | `method_len` | UTF-8 encoded chunking method name |
| `token_count` | `uint32` | 4 | Estimated count of tokens in chunk |
| `created_at` | `float64` | 8 | Creation date as a Unix epoch timestamp |

---

## Future Extensibility

This architecture is built for gradual expansion of ZionDB:

1. **HNSW / IVF Index Serializers**:
   Future indices like HNSW require graph node tables and adjacency list serialization. To support this, we implement `HNSWIndexSerializer` inheriting from `IndexSerializer` and register it in the registry `_INDEX_SERIALIZERS["hnsw"]`. It will output `hnsw.index` files in the collection folder.
2. **Metadata & Inverted Index Serializers**:
   For metadata filters, we can implement secondary inverted indexes and register `MetadataIndexSerializer` to manage saving/loading those maps (e.g., `metadata.index`).
3. **Write-Ahead Log (WAL)**:
   For transactional safety, a WAL serializer can serialize in-flight changes to `wal.log`.
4. **SQLite Record Serializer**:
   If we migrate from in-memory record storage to SQLite, we can introduce `SQLiteRecordSerializer` which dumps database rows or manages SQL migrations, without changes to `CollectionPersistence`.
