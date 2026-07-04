from pathlib import Path
from typing import Tuple
import yaml

from ziondb.collection.collection_config import CollectionConfig
from ziondb.persistence.exceptions import SerializationError


class ConfigSerializer:
    """Responsible for serializing and deserializing CollectionConfig objects to and from YAML."""

    def serialize(
        self,
        file_path: Path,
        name: str,
        config: CollectionConfig,
        embedding_dim: int,
        version: int = 1
    ) -> None:
        """
        Serializes CollectionConfig to a YAML file.
        """
        try:
            data = {
                "collection_name": name,
                "persistence_version": version,
                "embedding_dimension": embedding_dim,
                "embedding_model": config.embedding_model,
                "chunking_method": config.chunking_method,
                "similarity_metric": config.similarity_metric,
                "index_type": config.index_type,
                "storage_type": config.storage_type,
                "cache_dir": config.cache_dir,
                "max_length": config.max_length,
                "splitter_type": config.splitter_type,
                "regex_pattern": config.regex_pattern,
                "spacy_model": config.spacy_model,
                "buffer_size": config.buffer_size,
                "threshold_type": config.threshold_type,
                "threshold_value": config.threshold_value,
                "use_text_embedding": config.use_text_embedding
            }
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False)
        except Exception as e:
            raise SerializationError(f"Failed to serialize config to YAML: {e}") from e

    def deserialize(self, file_path: Path) -> Tuple[str, CollectionConfig, int, int]:
        """
        Deserializes a YAML file back into collection configurations.

        Returns:
            Tuple[str, CollectionConfig, int, int]:
                (collection_name, CollectionConfig, embedding_dimension, persistence_version)
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                raise SerializationError("Invalid config YAML file: root is not a dictionary.")

            name = data.get("collection_name")
            if not name:
                raise SerializationError("Missing 'collection_name' field in config.")

            version = data.get("persistence_version", 1)
            embedding_dim = data.get("embedding_dimension", 0)

            config = CollectionConfig(
                embedding_model=data.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
                chunking_method=data.get("chunking_method", "kamradt"),
                similarity_metric=data.get("similarity_metric", "cosine"),
                index_type=data.get("index_type", "brute_force"),
                storage_type=data.get("storage_type", "in_memory"),
                cache_dir=data.get("cache_dir", "models"),
                max_length=data.get("max_length", 256),
                splitter_type=data.get("splitter_type", "regex"),
                regex_pattern=data.get("regex_pattern"),
                spacy_model=data.get("spacy_model", "en_sentencizer"),
                buffer_size=data.get("buffer_size", 1),
                threshold_type=data.get("threshold_type", "percentile"),
                threshold_value=data.get("threshold_value", 50.0),
                use_text_embedding=data.get("use_text_embedding", False)
            )

            return name, config, embedding_dim, version
        except Exception as e:
            if not isinstance(e, SerializationError):
                raise SerializationError(f"Failed to deserialize config from YAML: {e}") from e
            raise e
