from abc import ABC, abstractmethod
from typing import Iterator
from ziondb.storage.record import ChunkRecord

class Storage(ABC):
    """Abstract interface defining the operations contract for a ZionDB storage driver."""

    @abstractmethod
    def insert(self, record: ChunkRecord) -> None:
        """
        Inserts a new record into storage.

        Args:
            record: The ChunkRecord to store.

        Raises:
            RecordAlreadyExistsError: If a record with the same ID already exists.
        """
        pass

    @abstractmethod
    def get(self, record_id: str) -> ChunkRecord:
        """
        Retrieves a record by its unique ID.

        Args:
            record_id: The ID of the record to fetch.

        Returns:
            ChunkRecord: The stored record.

        Raises:
            RecordNotFoundError: If the ID does not exist in storage.
        """
        pass

    @abstractmethod
    def update(self, record: ChunkRecord) -> None:
        """
        Updates an existing record in storage.

        Args:
            record: The updated ChunkRecord.

        Raises:
            RecordNotFoundError: If the record ID does not exist.
        """
        pass

    @abstractmethod
    def delete(self, record_id: str) -> None:
        """
        Deletes a record from storage by its ID.

        Args:
            record_id: The ID of the record to delete.

        Raises:
            RecordNotFoundError: If the ID does not exist in storage.
        """
        pass

    @abstractmethod
    def exists(self, record_id: str) -> bool:
        """
        Checks if a record with the specified ID exists in storage.

        Args:
            record_id: The ID to check.

        Returns:
            bool: True if it exists, False otherwise.
        """
        pass

    @abstractmethod
    def iterate(self) -> Iterator[ChunkRecord]:
        """
        Returns an iterator over all stored records.

        Returns:
            Iterator[ChunkRecord]: An iterator yielding ChunkRecords.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Removes all records from storage."""
        pass

    @abstractmethod
    def size(self) -> int:
        """
        Returns the total number of records currently stored.

        Returns:
            int: The record count.
        """
        pass
