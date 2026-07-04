from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

from ziondb.collection.collection_config import CollectionConfig
from ziondb.index.vector_index import VectorIndex
from ziondb.storage.record import ChunkRecord
from ziondb.persistence.serializers.index_serializer import IndexSerializer
from ziondb.persistence.serializers.brute_force_serializer import BruteForceIndexSerializer
from ziondb.persistence.serializers.config_serializer import ConfigSerializer
from ziondb.persistence.serializers.record_serializer import RecordSerializer
from ziondb.persistence.exceptions import PersistenceError


class CollectionPersistence:
    """Orchestrates serialization and deserialization across collection components."""

    # Dynamic registry mapping index types to their respective serializer classes
    _INDEX_SERIALIZERS: Dict[str, Type[IndexSerializer]] = {
        "brute_force": BruteForceIndexSerializer,
        # Extensible: future index serializers can be registered here (e.g. HNSW)
    }

    def __init__(
        self,
        config_serializer: Optional[ConfigSerializer] = None,
        record_serializer: Optional[RecordSerializer] = None,
    ) -> None:
        """
        Initialize CollectionPersistence.
        """
        self.config_serializer = config_serializer or ConfigSerializer()
        self.record_serializer = record_serializer or RecordSerializer()

    def _get_index_serializer(self, index_type: str) -> IndexSerializer:
        serializer_cls = self._INDEX_SERIALIZERS.get(index_type.lower())
        if not serializer_cls:
            raise PersistenceError(f"No IndexSerializer registered for index type: '{index_type}'")
        return serializer_cls()

    def save(self, dir_path: Path, collection) -> None:
        """
        Saves a Collection's state to disk.
        """
        records = list(collection.storage.iterate())
        self.save_raw(
            dir_path=dir_path,
            name=collection.name,
            config=collection.config,
            records=records,
            index=collection.index
        )

    def save_raw(
        self,
        dir_path: Path,
        name: str,
        config: CollectionConfig,
        records: List[ChunkRecord],
        index: VectorIndex
    ) -> None:
        """
        Saves raw collection configurations, records, and index state.
        """
        try:
            dir_path.mkdir(parents=True, exist_ok=True)

            # 1. Config
            config_path = dir_path / "config.yaml"
            embedding_dim = 0
            if records:
                embedding_dim = len(records[0].embedding)
            self.config_serializer.serialize(
                file_path=config_path,
                name=name,
                config=config,
                embedding_dim=embedding_dim,
                version=1
            )

            # 2. Records
            records_path = dir_path / "records.bin"
            self.record_serializer.serialize(
                file_path=records_path,
                records=records,
                version=1
            )

            # 3. Index (dynamically resolved)
            idx_serializer = self._get_index_serializer(config.index_type)
            idx_serializer.serialize(dir_path, index)

        except Exception as e:
            if not isinstance(e, PersistenceError):
                raise PersistenceError(f"Failed to save collection: {e}") from e
            raise e

    def load(self, dir_path: Path) -> Tuple[str, CollectionConfig, List[ChunkRecord]]:
        """
        Loads the configurations and records from the specified directory.
        """
        config_path = dir_path / "config.yaml"
        records_path = dir_path / "records.bin"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        if not records_path.exists():
            raise FileNotFoundError(f"Records file not found at {records_path}")

        try:
            name, config, embedding_dim, version = self.config_serializer.deserialize(config_path)
            records = self.record_serializer.deserialize(records_path)
            return name, config, records
        except Exception as e:
            if not isinstance(e, PersistenceError) and not isinstance(e, FileNotFoundError):
                raise PersistenceError(f"Failed to load collection: {e}") from e
            raise e

    def load_index(self, dir_path: Path, index: VectorIndex, index_type: str) -> None:
        """
        Deserializes index-specific data into the VectorIndex.
        """
        idx_serializer = self._get_index_serializer(index_type)
        idx_serializer.deserialize(dir_path, index)
