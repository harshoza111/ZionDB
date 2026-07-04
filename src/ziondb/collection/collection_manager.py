from typing import Dict, List, Optional

from ziondb.collection.collection import Collection
from ziondb.collection.collection_config import CollectionConfig
from ziondb.collection.exceptions import CollectionAlreadyExistsError, CollectionNotFoundError
from ziondb.core.interfaces import EmbeddingProvider


class CollectionManager:
    """Manages the creation, retrieval, and deletion of independent collections in ZionDB."""

    def __init__(self, default_embedding_provider: Optional[EmbeddingProvider] = None) -> None:
        """
        Initialize the CollectionManager.

        Args:
            default_embedding_provider: Optional fallback embedding provider for newly created collections.
        """
        self._collections: Dict[str, Collection] = {}
        self._default_embedding_provider = default_embedding_provider

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

        Raises:
            CollectionAlreadyExistsError: If a collection with the given name already exists.
        """
        if name in self._collections:
            raise CollectionAlreadyExistsError(f"Collection '{name}' already exists.")

        col_config = config or CollectionConfig()
        provider = embedding_provider or self._default_embedding_provider

        collection = Collection(
            name=name,
            config=col_config,
            embedding_provider=provider
        )
        self._collections[name] = collection
        return collection

    def get_collection(self, name: str) -> Collection:
        """
        Retrieves a collection by its name.

        Args:
            name: The name of the collection to fetch.

        Returns:
            Collection: The requested Collection object.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
        """
        if name not in self._collections:
            raise CollectionNotFoundError(f"Collection '{name}' not found.")
        return self._collections[name]

    def delete_collection(self, name: str) -> None:
        """
        Deletes a collection by its name.

        Args:
            name: The name of the collection to delete.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
        """
        if name not in self._collections:
            raise CollectionNotFoundError(f"Collection '{name}' not found.")
        del self._collections[name]

    def list_collections(self) -> List[str]:
        """
        Lists names of all collections managed.

        Returns:
            List[str]: Active collection names.
        """
        return list(self._collections.keys())

    def exists(self, name: str) -> bool:
        """
        Checks if a collection with the given name exists.

        Args:
            name: The name of the collection.

        Returns:
            bool: True if it exists, False otherwise.
        """
        return name in self._collections
