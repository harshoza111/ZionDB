from abc import ABC, abstractmethod
from pathlib import Path
from ziondb.index.vector_index import VectorIndex


class IndexSerializer(ABC):
    """Abstract interface for serializing and deserializing vector indexes."""

    @abstractmethod
    def serialize(self, dir_path: Path, index: VectorIndex) -> None:
        """
        Serializes the vector index state to files inside the directory.

        Args:
            dir_path: The directory path where index state should be saved.
            index: The active VectorIndex to serialize.
        """
        pass

    @abstractmethod
    def deserialize(self, dir_path: Path, index: VectorIndex) -> None:
        """
        Deserializes the vector index state from files inside the directory.

        Args:
            dir_path: The directory path where index state is saved.
            index: The active VectorIndex to load state into.
        """
        pass
