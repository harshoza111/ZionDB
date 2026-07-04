from datetime import datetime, timezone
import json
from pathlib import Path
import struct
from typing import List
import numpy as np

from ziondb.storage.record import ChunkRecord, SystemMetadata
from ziondb.persistence.exceptions import (
    SerializationError,
    InvalidPersistenceVersion,
    CorruptedCollectionError
)


class RecordSerializer:
    """Handles binary serialization and deserialization of ChunkRecords."""

    def serialize(self, file_path: Path, records: List[ChunkRecord], version: int = 1) -> None:
        """
        Serializes a list of ChunkRecords to a binary file.
        """
        try:
            with open(file_path, "wb") as f:
                # Write file header: Magic (4 bytes) + Version (2 bytes) + Record Count (4 bytes)
                f.write(b"ZION")
                f.write(struct.pack("<H", version))
                f.write(struct.pack("<I", len(records)))

                for record in records:
                    # 1. Record ID
                    id_bytes = record.id.encode("utf-8")
                    f.write(struct.pack("<I", len(id_bytes)))
                    f.write(id_bytes)

                    # 2. Text
                    text_bytes = record.text.encode("utf-8")
                    f.write(struct.pack("<I", len(text_bytes)))
                    f.write(text_bytes)

                    # 3. Embedding (Count of float elements + raw float32 bytes)
                    embedding_arr = np.ascontiguousarray(record.embedding, dtype=np.float32)
                    f.write(struct.pack("<I", embedding_arr.size))
                    f.write(embedding_arr.tobytes())

                    # 4. Metadata (JSON serialized)
                    metadata_str = json.dumps(record.metadata)
                    meta_bytes = metadata_str.encode("utf-8")
                    f.write(struct.pack("<I", len(meta_bytes)))
                    f.write(meta_bytes)

                    # 5. SystemMetadata
                    sys = record.system_metadata
                    
                    doc_id_bytes = sys.document_id.encode("utf-8")
                    f.write(struct.pack("<I", len(doc_id_bytes)))
                    f.write(doc_id_bytes)

                    model_bytes = sys.embedding_model.encode("utf-8")
                    f.write(struct.pack("<I", len(model_bytes)))
                    f.write(model_bytes)

                    method_bytes = sys.chunking_method.encode("utf-8")
                    f.write(struct.pack("<I", len(method_bytes)))
                    f.write(method_bytes)

                    f.write(struct.pack("<I", sys.token_count))
                    
                    timestamp = sys.created_at.timestamp()
                    f.write(struct.pack("<d", timestamp))

        except Exception as e:
            raise SerializationError(f"Failed to serialize records: {e}") from e

    def deserialize(self, file_path: Path) -> List[ChunkRecord]:
        """
        Deserializes a list of ChunkRecords from a binary file.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Records file not found at {file_path}")

        try:
            with open(file_path, "rb") as f:
                # 1. Read and verify Magic
                magic = f.read(4)
                if magic != b"ZION":
                    raise CorruptedCollectionError(f"Invalid file magic header: {magic!r}. File may be corrupted.")

                # 2. Read and verify Version
                version_bytes = f.read(2)
                if len(version_bytes) < 2:
                    raise CorruptedCollectionError("EOF reached while reading version header.")
                version = struct.unpack("<H", version_bytes)[0]
                if version != 1:
                    raise InvalidPersistenceVersion(f"Unsupported persistence version: {version}. Expected 1.")

                # 3. Read record count
                count_bytes = f.read(4)
                if len(count_bytes) < 4:
                    raise CorruptedCollectionError("EOF reached while reading record count header.")
                record_count = struct.unpack("<I", count_bytes)[0]

                records: List[ChunkRecord] = []
                for i in range(record_count):
                    # Helper to read exact count of bytes
                    def read_exact(n: int, field_name: str) -> bytes:
                        data = f.read(n)
                        if len(data) < n:
                            raise CorruptedCollectionError(
                                f"Unexpected EOF reading record {i+1} field '{field_name}'. Expected {n} bytes, got {len(data)}."
                            )
                        return data

                    # Read Record ID
                    id_len = struct.unpack("<I", read_exact(4, "id_len"))[0]
                    record_id = read_exact(id_len, "record_id").decode("utf-8")

                    # Read Text
                    text_len = struct.unpack("<I", read_exact(4, "text_len"))[0]
                    text = read_exact(text_len, "text").decode("utf-8")

                    # Read Embedding
                    emb_len = struct.unpack("<I", read_exact(4, "emb_len"))[0]
                    emb_bytes = read_exact(emb_len * 4, "embedding")
                    embedding = np.frombuffer(emb_bytes, dtype=np.float32).copy()

                    # Read Metadata
                    meta_len = struct.unpack("<I", read_exact(4, "meta_len"))[0]
                    meta_bytes = read_exact(meta_len, "metadata")
                    metadata = json.loads(meta_bytes.decode("utf-8"))

                    # Read SystemMetadata
                    doc_id_len = struct.unpack("<I", read_exact(4, "doc_id_len"))[0]
                    doc_id = read_exact(doc_id_len, "doc_id").decode("utf-8")

                    model_len = struct.unpack("<I", read_exact(4, "model_len"))[0]
                    model_name = read_exact(model_len, "model_name").decode("utf-8")

                    method_len = struct.unpack("<I", read_exact(4, "method_len"))[0]
                    chunking_method = read_exact(method_len, "chunking_method").decode("utf-8")

                    token_count = struct.unpack("<I", read_exact(4, "token_count"))[0]
                    created_at_ts = struct.unpack("<d", read_exact(8, "created_at"))[0]
                    created_at = datetime.fromtimestamp(created_at_ts, timezone.utc)

                    sys_meta = SystemMetadata(
                        document_id=doc_id,
                        embedding_model=model_name,
                        chunking_method=chunking_method,
                        token_count=token_count,
                        created_at=created_at
                    )

                    record = ChunkRecord(
                        id=record_id,
                        text=text,
                        embedding=embedding,
                        metadata=metadata,
                        system_metadata=sys_meta
                    )
                    records.append(record)

                return records

        except (CorruptedCollectionError, InvalidPersistenceVersion) as e:
            raise e
        except Exception as e:
            raise SerializationError(f"Failed to deserialize records: {e}") from e
