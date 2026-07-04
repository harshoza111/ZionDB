from pathlib import Path
from ziondb.index.vector_index import VectorIndex
from ziondb.persistence.serializers.index_serializer import IndexSerializer


class BruteForceIndexSerializer(IndexSerializer):
    """
    Serializer for BruteForceIndex.
    Since brute-force indexes do not store index-specific structural state,
    saving is a no-op and loading initiates a rebuild from the underlying storage.
    """

    def serialize(self, dir_path: Path, index: VectorIndex) -> None:
        """No-op for brute-force index."""
        pass

    def deserialize(self, dir_path: Path, index: VectorIndex) -> None:
        """Triggers index rebuild from its storage provider."""
        index.rebuild()
