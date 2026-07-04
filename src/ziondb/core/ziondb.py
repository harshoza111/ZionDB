from typing import List, Optional

from ziondb.collection.collection_manager import CollectionManager
from ziondb.collection.collection import Collection
from ziondb.collection.collection_config import CollectionConfig
from ziondb.core.interfaces import EmbeddingProvider


class ZionDB:
    """The public entry point for ZionDB."""

    def __init__(self, default_embedding_provider: Optional[EmbeddingProvider] = None) -> None:
        """
        Initialize ZionDB.

        Args:
            default_embedding_provider: Optional default embedding provider for all collections.
        """
        self._manager = CollectionManager(default_embedding_provider=default_embedding_provider)

    def create_collection(
        self,
        name: str,
        config: Optional[CollectionConfig] = None,
        embedding_provider: Optional[EmbeddingProvider] = None
    ) -> Collection:
        """
        Creates a new collection.

        Args:
            name: Unique name for the collection.
            config: Optional config settings for the collection.
            embedding_provider: Optional custom embedding provider.

        Returns:
            Collection: The created Collection object.
        """
        return self._manager.create_collection(
            name=name,
            config=config,
            embedding_provider=embedding_provider
        )

    def get_collection(self, name: str) -> Collection:
        """
        Retrieves a collection by its name.

        Args:
            name: The name of the collection to fetch.

        Returns:
            Collection: The requested Collection object.
        """
        return self._manager.get_collection(name)

    def delete_collection(self, name: str) -> None:
        """
        Deletes a collection by its name.

        Args:
            name: The name of the collection to delete.
        """
        self._manager.delete_collection(name)

    def list_collections(self) -> List[str]:
        """
        Lists names of all collections managed.

        Returns:
            List[str]: Active collection names.
        """
        return self._manager.list_collections()

    def exists(self, name: str) -> bool:
        """
        Checks if a collection with the given name exists.

        Args:
            name: The name of the collection.

        Returns:
            bool: True if it exists, False otherwise.
        """
        return self._manager.exists(name)
