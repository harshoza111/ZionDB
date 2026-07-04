import struct
import numpy as np
import pytest

from ziondb import ZionDB, CollectionConfig
from ziondb.persistence.exceptions import CorruptedCollectionError, InvalidPersistenceVersion


def test_persistence_save_load_search(tmp_path, mock_embedder):
    db = ZionDB(default_embedding_provider=mock_embedder)

    # 1. Create a collection with custom configuration
    config = CollectionConfig(
        splitter_type="regex",
        regex_pattern=r"(?<=[.!?]) +",
        buffer_size=1,
        threshold_value=50.0
    )
    coll = db.create_collection("test_persistence", config=config)

    # 2. Add records
    coll.add_document(
        text="This is sentence one. This is sentence two! Sentence three is here?",
        metadata={"category": "general"},
        id="doc_1"
    )

    original_count = coll.count()
    assert original_count > 0

    # Run a similarity search before saving
    query = "sentence one"
    original_results = coll.search(query=query, top_k=2)
    assert len(original_results) > 0

    # 3. Save the collection to disk
    save_dir = tmp_path / "test_persistence_dir"
    coll.save(save_dir)

    # Assert directory files exist
    assert (save_dir / "config.yaml").exists()
    assert (save_dir / "records.bin").exists()

    # 4. Load the collection into a completely fresh/empty collection
    coll_loaded = db.create_collection("loaded_coll", config=CollectionConfig())
    coll_loaded.load(save_dir)

    # 5. Verify config preservation
    assert coll_loaded.name == "test_persistence"
    assert coll_loaded.config.splitter_type == "regex"
    assert coll_loaded.config.regex_pattern == r"(?<=[.!?]) +"
    assert coll_loaded.config.buffer_size == 1
    assert coll_loaded.config.threshold_value == 50.0

    # 6. Verify count and record preservation
    assert coll_loaded.count() == original_count

    original_records = list(coll.storage.iterate())
    loaded_records = list(coll_loaded.storage.iterate())

    assert len(original_records) == len(loaded_records)
    for orig, load in zip(original_records, loaded_records):
        assert orig.id == load.id
        assert orig.text == load.text
        assert np.allclose(orig.embedding, load.embedding)
        assert orig.metadata == load.metadata
        assert orig.system_metadata.document_id == load.system_metadata.document_id
        assert orig.system_metadata.embedding_model == load.system_metadata.embedding_model
        assert orig.system_metadata.chunking_method == load.system_metadata.chunking_method
        assert orig.system_metadata.token_count == load.system_metadata.token_count
        assert orig.system_metadata.created_at == load.system_metadata.created_at

    # 7. Verify search query consistency (Saving -> Loading -> Searching produces identical results)
    loaded_results = coll_loaded.search(query=query, top_k=2)
    assert len(loaded_results) == len(original_results)
    for r_orig, r_load in zip(original_results, loaded_results):
        assert r_orig.record.id == r_load.record.id
        assert r_orig.record.text == r_load.record.text
        assert np.isclose(r_orig.score, r_load.score)


def test_persistence_empty_collection(tmp_path, mock_embedder):
    db = ZionDB(default_embedding_provider=mock_embedder)
    coll = db.create_collection("empty_coll")

    save_dir = tmp_path / "empty_coll_dir"
    coll.save(save_dir)

    assert (save_dir / "config.yaml").exists()
    assert (save_dir / "records.bin").exists()

    # Load back
    loaded_coll = db.create_collection("loaded_empty")
    loaded_coll.load(save_dir)
    assert loaded_coll.count() == 0
    assert loaded_coll.search(query="test", top_k=5) == []


def test_persistence_overwrite(tmp_path, mock_embedder):
    db = ZionDB(default_embedding_provider=mock_embedder)
    coll = db.create_collection("overwrite_coll")
    coll.add_document("Content A", id="d1")

    save_dir = tmp_path / "overwrite_dir"
    coll.save(save_dir)
    assert coll.count() == 1

    # Overwrite by saving a different collection state to the same folder
    coll2 = db.create_collection("overwrite_coll2")
    coll2.add_document("Content B, different and longer", id="d2")
    coll2.add_document("Content C", id="d3")
    coll2.save(save_dir)

    # Verify the load retrieves the new state
    loaded_coll = db.create_collection("loaded_overwrite")
    loaded_coll.load(save_dir)
    assert loaded_coll.name == "overwrite_coll2"
    assert loaded_coll.count() == 2
    assert any("Content B" in r.text for r in loaded_coll.storage.iterate())


def test_corrupted_header_raises_error(tmp_path, mock_embedder):
    db = ZionDB(default_embedding_provider=mock_embedder)
    coll = db.create_collection("corrupt_coll")
    coll.add_document("Some content")

    save_dir = tmp_path / "corrupt_dir"
    coll.save(save_dir)

    # Corrupt the records.bin file magic header (first 4 bytes)
    records_file = save_dir / "records.bin"
    with open(records_file, "r+b") as f:
        f.seek(0)
        f.write(b"CORR")  # wrong magic header

    loaded_coll = db.create_collection("loaded_corrupt")
    with pytest.raises(CorruptedCollectionError, match="Invalid file magic header"):
        loaded_coll.load(save_dir)


def test_invalid_version_raises_error(tmp_path, mock_embedder):
    db = ZionDB(default_embedding_provider=mock_embedder)
    coll = db.create_collection("version_coll")
    coll.add_document("Some content")

    save_dir = tmp_path / "version_dir"
    coll.save(save_dir)

    # Modify version in records.bin (bytes 4-5) to 9999
    records_file = save_dir / "records.bin"
    with open(records_file, "r+b") as f:
        f.seek(4)
        f.write(struct.pack("<H", 9999))

    loaded_coll = db.create_collection("loaded_version")
    with pytest.raises(InvalidPersistenceVersion, match="Unsupported persistence version: 9999"):
        loaded_coll.load(save_dir)
