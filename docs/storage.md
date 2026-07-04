# ZionDB Storage Layer Architecture

This document describes the design, mechanics, and future compatibility of the **ZionDB Storage Layer (Version 1)**.

---

## Architecture Overview

The storage layer acts as a decoupled store for processed text chunks and their corresponding embedding vectors. It has zero knowledge of chunking strategies, embedding algorithms, or vector similarity math.

```
                  ┌──────────────────────┐
                  │    Indexing Pipeline │
                  └──────────┬───────────┘
                             │ (Chunk + vector output)
                             ▼
                  ┌──────────────────────┐
                  │     ChunkRecord      │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Storage Interface   │
                  └──────────┬───────────┘
                             │
                     ┌───────┴───────┐
                     ▼               ▼
             [InMemoryStorage]   [Future drivers...]
```

---

## Responsibilities

1. **Record Lifecycle Management**: Facilitates inserting, retrieving, updating, deleting, and checking the existence of `ChunkRecord` objects.
2. **Resource Tracking**: Monitors storage metrics such as total record count (`size()`).
3. **Data Integrity**: Enforces uniqueness constraints on record IDs and raises typed domain exceptions (e.g. `RecordAlreadyExistsError`, `RecordNotFoundError`) rather than generic Python errors.

---

## Design Decisions

### Why a `list` + `dict` was chosen for `InMemoryStorage`

`InMemoryStorage` is implemented using two internal data structures:
* `_records: list[ChunkRecord]`: Serves as the primary flat list of stored records.
* `_id_to_index: dict[str, int]`: Maps record IDs to their current position indices in the `_records` list.

This design provides optimal time complexities for all primary operations:

| Operation | Time Complexity | Notes |
| :--- | :--- | :--- |
| `insert()` | $O(1)$ | Append to the list and write the index to the dictionary. |
| `get()` | $O(1)$ | Direct dictionary lookup of the list index, then array access. |
| `update()` | $O(1)$ | Direct dictionary lookup and replacement. |
| `exists()` | $O(1)$ | Simple hash map check on the index keys. |
| `delete()` | $O(1)$ | Custom swap-and-pop implementation (see below). |
| `size()` | $O(1)$ | Returns length of the internal list. |
| `iterate()` | $O(1)$ amortized | Returns an iterator over the internal list. |

### Optimized O(1) Deletes: The Swap-and-Pop Technique

In Python, deleting an item from the middle of a list (`list.pop(index)` or `list.remove(item)`) is an $O(N)$ operation because all subsequent elements must be shifted left to maintain contiguity. This shift would also invalidate the index mapping in `_id_to_index`, requiring an $O(N)$ dictionary rebuild.

To maintain strict $O(1)$ deletion complexity and keep data structures in sync, `InMemoryStorage` uses the **swap-and-pop** technique:

1. **Check Position**: If the element to delete is already the last element in the list, pop it directly ($O(1)$).
2. **Swap**: If it is in the middle of the list, swap the target element with the last element in the list.
3. **Update Index Map**: Update the dictionary entry for the swapped last element to point to its new index position.
4. **Pop**: Pop the target element (now at the end of the list) in $O(1)$ time.
5. **Clean Dictionary**: Delete the target record's ID key from the dictionary.

*Trade-off*: This approach is not order-preserving. The sequential order of records in `_records` changes dynamically during intermediate deletions. Since vector search and indexing operate on the set of records as a whole rather than a strict sequence, order instability is a negligible trade-off for $O(1)$ deletion speeds.

---

## Future Compatibility

The Storage abstract base class interface makes it straightforward to extend ZionDB without changing pipeline or query code:

### 1. SQLite Persistence (`SQLiteStorage`)
A future driver can implement the `Storage` interface by writing `ChunkRecord` text and metadata to an SQLite database, and storing serialized 384-dimensional float arrays (as BLOBs or JSON strings).

### 2. Disk Storage (`DiskStorage`)
For larger datasets, a disk-based storage driver can serialize records to binary files (e.g. Protocol Buffers, MessagePack, or custom binary layout) and maintain index mappings on disk using a B-Tree structure.

### 3. HNSW and IVF Indexes
Future indexing algorithms (like Hierarchical Navigable Small World graphs or Inverted File partitioning) will wrap or subscribe to Storage insertion/deletion events to construct and update query acceleration graphs asynchronously.

### 4. Product Quantization (PQ)
Storage drivers can integrate compression back-ends to store codebooks and quantized vector representations, saving significant memory as index size scales.
